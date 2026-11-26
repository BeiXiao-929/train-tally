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

import pytest

from ledger.comm_ledger import dp_bytes, ledger as comm_ledger, tp_bytes
from ledger.memory_ledger import total_gb
from ledger.mfu_ledger import measured_mfu, theoretical_mfu_upper_bound
from ledger.model_configs import ParallelConfig, get as get_model


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
