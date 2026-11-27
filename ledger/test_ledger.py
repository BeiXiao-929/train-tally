"""ledger/test_ledger.py —— 账本自检（纯 CPU，无卡也能跑）。

这些测试守的不是「代码能不能跑」，而是「公式有没有写错」：
账本一旦算错，后面所有『误差 < 10%』的验收都失去意义。

覆盖：
1. 参数量解析值与公开值的偏差 < 5%（口径错了要先修配置）
2. ZeRO 档位越高、显存越少（单调性）
3. DDP 的 AllReduce 字节数 = 2×P×(N-1)/N（4 卡时应为 1.5P）
4. TP 的通信时间随 tp 上升而上升（同一链路）
5. 链路由 NVLink 换 PCIe，TP 通信时间上升一个数量级
6. MFU 上界恒 ≤ 1，且实测值不得超过上界
"""
from __future__ import annotations

import json
import pytest

from common import io
from ledger.comm_ledger import (
    dp_bytes,
    infer_effective_bandwidth,
    ledger as comm_ledger,
    solve_break_even_bw,
    tp_break_even,
    tp_bytes,
)
from ledger.memory_ledger import total_gb
from ledger.mfu_ledger import measured_mfu, theoretical_mfu_upper_bound
from ledger.model_configs import ModelConfig, ParallelConfig, get as get_model
from trace_parser.report import summarize_csv_dir


def test_param_count_matches_published():
    for name in ("qwen2.5-7b", "llama3-8b"):
        m = get_model(name)
        dev = m.deviation_pct()
        assert dev is not None
        assert abs(dev) < 5.0, f"{name} 参数量偏差 {dev:.2f}% 超过 5%，先查配置"


def test_zero_stage_monotonic():
    m = get_model("qwen2.5-7b")
    totals = []
    for s in (0, 1, 2, 3):
        pc = ParallelConfig(dp=4, zero_stage=s, batch=1, seq=2048)
        totals.append(total_gb(m, pc)["total_gb"])
    assert totals[0] > totals[1] > totals[2] > totals[3], \
        f"ZeRO 显存未单调递减：{totals}"


def test_activation_checkpointing_reduces_memory():
    m = get_model("qwen2.5-7b")
    off = total_gb(m, ParallelConfig(dp=4, zero_stage=3, activation_checkpointing=False))
    on = total_gb(m, ParallelConfig(dp=4, zero_stage=3, activation_checkpointing=True))
    assert on["activations_gb"] < off["activations_gb"] * 0.5


def test_ddp_allreduce_bytes_ring_formula():
    m = get_model("tiny-debug")
    pc = ParallelConfig(dp=4, zero_stage=0, dtype="bf16")
    p = m.total_params() * 1e9 * 2
    expect = 2 * p * (4 - 1) / 4
    assert dp_bytes(m, pc) == pytest.approx(expect, rel=1e-9)


def test_zero3_has_extra_allgather():
    m = get_model("tiny-debug")
    ddp = dp_bytes(m, ParallelConfig(dp=4, zero_stage=0))
    z3 = dp_bytes(m, ParallelConfig(dp=4, zero_stage=3))
    assert z3 == pytest.approx(ddp * 1.5, rel=1e-9)


def test_tp_bytes_zero_when_tp1_and_grows_with_tp():
    m = get_model("qwen2.5-7b")
    assert tp_bytes(m, ParallelConfig(tp=1)) == 0.0
    t2 = tp_bytes(m, ParallelConfig(tp=2, batch=1, seq=2048))
    t4 = tp_bytes(m, ParallelConfig(tp=4, batch=1, seq=2048))
    assert t4 > t2 > 0


def test_pcie_is_much_worse_than_nvlink_for_tp():
    m = get_model("qwen2.5-7b")
    pc = ParallelConfig(dp=1, tp=4, batch=1, seq=2048)
    nv = comm_ledger(m, pc, link="nvlink")
    pcie = comm_ledger(m, pc, link="pcie4")
    assert pcie["comm_ms"] > nv["comm_ms"] * 5, \
        f"PCIe/NVLink 的 TP 通信时间差不足 5 倍：{pcie['comm_ms']} vs {nv['comm_ms']}"
    assert pcie["comm_ratio"] > nv["comm_ratio"]


def test_mfu_upper_bound_and_measurement():
    m = get_model("qwen2.5-7b")
    pc = ParallelConfig(dp=4, zero_stage=0, batch=1, seq=2048)
    ub = theoretical_mfu_upper_bound(m, pc, link="nvlink")["upper_bound_with_comm"]
    assert 0 < ub <= 1.0
    # 4 卡 × 4000 tok/s 的假设值仅用于检查「实测不得超上界」这条不变式
    meas = measured_mfu(m, tokens_per_s=4000.0, n_gpus=4)
    assert meas <= ub + 1e-6, f"实测 MFU {meas:.3f} 超过上界 {ub:.3f}，统计有误"


def test_tp_break_even_and_critical_bw():
    m = get_model("qwen2.5-7b")
    crit = solve_break_even_bw(m, tp=2, latency_us=10.0, gpu="a100", batch=1, seq=2048)
    assert 10.5 <= crit <= 11.5, f"预期 TP=2 临界带宽约 11.0 GB/s，实际 {crit}"

    pcie_measured = tp_break_even(m, b_eff_gbps=8.0, tp_values=(2, 4))
    assert all(not r["profitable"] and r["net_gain_pct"] < 0 for r in pcie_measured)

    pcie_theory = tp_break_even(m, b_eff_gbps=25.0, tp_values=(2, 4))
    assert all(r["profitable"] and r["net_gain_pct"] > 0 for r in pcie_theory)


def test_comm_overlap_and_infer_bw():
    m = get_model("qwen2.5-7b")
    pc = ParallelConfig(dp=4, zero_stage=0, batch=1, seq=2048)
    no_ov = comm_ledger(m, pc, link="pcie4", overlap=0.0)
    half_ov = comm_ledger(m, pc, link="pcie4", overlap=0.5)
    assert half_ov["exposed_comm_ms"] == pytest.approx(no_ov["comm_ms"] * 0.5, rel=1e-3)
    assert half_ov["comm_ratio"] < no_ov["comm_ratio"]

    b_eff = infer_effective_bandwidth(total_bytes=1e9, comm_ms=100.0)
    assert b_eff == pytest.approx(10.0, rel=1e-6)


def test_from_hf_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "num_hidden_layers": 28,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": True,
    }), encoding="utf-8")
    m = ModelConfig.from_hf_config(cfg, name="test-qwen3")
    assert m.name == "test-qwen3"
    assert m.n_kv_heads == 8 and m.head_dim == 128
    assert 0.5 < m.total_params() < 0.8


def test_summarize_csv_dir(tmp_path):
    io.write_csv(
        tmp_path / "nccl_summary_20261127_000000.csv",
        [{"nccl_share_pct": 25.0, "overlap_ratio": 0.4, "eff_bw_gbps": 12.5}],
        {"source": "run1"},
    )
    rows = summarize_csv_dir(tmp_path)
    assert len(rows) == 1
    assert rows[0]["kind"] == "nsys_nccl"
    assert float(rows[0]["eff_bw_gbps"]) == pytest.approx(12.5)
