"""ledger/comm_ledger.py —— 理论通信量账本（输入配置，输出每步字节数与耗时）。

文件职责
--------
按并行维度分别算每步的通信量，再除以链路有效带宽得到「通信时间」，
与计算时间相比得到「通信占比」。这个占比是 MFU 损失的第一解释变量。

公式（每步每卡）
----------------
DP AllReduce（环形）：
    2 × P_bytes × (N-1)/N               N = dp
ZeRO-1 / ZeRO-2（ReduceScatter + AllGather，字节数同 DDP，但两次集合通信）：
    2 × P_bytes × (N-1)/N
ZeRO-3 / FSDP（前向 AllGather + 反向 AllGather + 梯度 ReduceScatter）：
    3 × P_bytes × (N-1)/N
TP（每层前向 2 次 + 反向 2 次环形 AllReduce，共 4L 次，单次 2·b×s×h×bytes·(t-1)/t）：
    8 × L × b×s×h × bytes × (t-1)/t      ← 次数多、单次小，对延迟敏感
PP（每个 stage 边界每次收发激活）：
    2 × b×s×h × bytes × (p-1)（fwd+bwd 各一次）

**为什么 TP 在 PCIe 上不划算**：TP 的消息数是 `4×L`（7B/28 层 = 112 次/步），
单次只有 b×s×h×2 字节。带宽还没跑满，延迟已经把时间吃光了。
这就是 docs/selection_boundary.md 要量化的临界点。

TODO
----
[x] 增加通信与计算的重叠建模（--overlap，默认 0.0 为完全不重叠上界）
[ ] 增加 hierarchical AllReduce（多机时 ring 会跨机，实际带宽更低）
[ ] 支持 --nccl-algo LL/LL128/Simple 的带宽修正系数
[x] 从实测 trace 反推有效带宽（--nccl-csv 与 parse_nccl 结果对账）+ TP 临界带宽求解器

验收标准
--------
- 4 卡 DDP 的 AllReduce 字节数 = 2×P×(3/4)，与手算一致（误差 < 0.1%）
- TP=1 时 TP 项为 0；TP=4 的通信时间应显著高于 TP=2（同一链路）
- 换链路（pcie4 → nvlink）时 TP 通信时间下降一个数量级
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import io
from common.env import collect
from ledger.model_configs import ModelConfig, ParallelConfig, get as get_model

# 链路预设：单向有效带宽（GB/s，1e9）与单次消息延迟（us）
# 数值取工程经验区间，用 --bw/--latency 覆盖；报告里必须注明用的是哪一档。
LINK_PROFILES: Dict[str, Dict[str, float]] = {
    "nvlink": {"bw_gbps": 300.0, "latency_us": 5.0},
    "pcie4": {"bw_gbps": 25.0, "latency_us": 10.0},
    "pcie5": {"bw_gbps": 50.0, "latency_us": 8.0},
    "ib100": {"bw_gbps": 12.5, "latency_us": 20.0},
    "ib200": {"bw_gbps": 25.0, "latency_us": 15.0},
}

# 单卡峰值算力（TFLOPS，dense，bf16/fp16 with fp32 accumulate）
PEAK_TFLOPS: Dict[str, float] = {
    "a100": 312.0,
    "a100-40g": 312.0,
    "h100": 989.0,
    "4090": 165.0,
}


def param_bytes(model: ModelConfig, pc: ParallelConfig) -> float:
    """参与通信的参数字节数（每卡视角，已按 tp/pp 切分）。"""
    return model.total_params() * 1e9 / (pc.tp * pc.pp) * pc.dtype_bytes


def dp_bytes(model: ModelConfig, pc: ParallelConfig) -> float:
    n = max(pc.dp, 1)
    if n == 1:
        return 0.0
    p = param_bytes(model, pc)
    if pc.zero_stage == 3:
        return 3.0 * p * (n - 1) / n
    return 2.0 * p * (n - 1) / n


def tp_bytes(model: ModelConfig, pc: ParallelConfig) -> float:
    t = max(pc.tp, 1)
    if t == 1:
        return 0.0
    msg = pc.batch * pc.seq * model.hidden * pc.dtype_bytes
    # 4*L 次 AllReduce × 每次环形 AllReduce 搬运 2*msg*(t-1)/t（与 docs/selection_boundary.md 一致）
    return 8.0 * model.n_layers * msg * (t - 1) / t


def tp_messages(model: ModelConfig, pc: ParallelConfig) -> int:
    if pc.tp == 1:
        return 0
    return 4 * model.n_layers


def pp_bytes(model: ModelConfig, pc: ParallelConfig) -> float:
    p = max(pc.pp, 1)
    if p == 1:
        return 0.0
    msg = pc.batch * pc.seq * model.hidden * pc.dtype_bytes
    return 2.0 * msg * (p - 1)


def compute_flops_per_rank(model: ModelConfig, pc: ParallelConfig) -> float:
    """每步每卡的计算量（6PD 近似：前向 2 + 反向 4）。"""
    tokens = pc.batch * pc.seq
    return 6.0 * model.total_params() * 1e9 * tokens / (pc.tp * pc.pp)


def ledger(model: ModelConfig, pc: ParallelConfig, link: str = "nvlink",
           bw_gbps: Optional[float] = None, latency_us: Optional[float] = None,
           gpu: str = "a100", overlap: float = 0.0) -> Dict[str, Any]:
    prof = LINK_PROFILES.get(link, LINK_PROFILES["nvlink"])
    bw = bw_gbps if bw_gbps else prof["bw_gbps"]
    lat = latency_us if latency_us is not None else prof["latency_us"]
    overlap = max(0.0, min(1.0, float(overlap)))

    d_bytes = dp_bytes(model, pc)
    t_bytes = tp_bytes(model, pc)
    p_bytes = pp_bytes(model, pc)
    n_msg = tp_messages(model, pc) + (2 * max(pc.pp - 1, 0)) + (1 if pc.dp > 1 else 0)

    total_bytes = d_bytes + t_bytes + p_bytes
    comm_ms = total_bytes / (bw * 1e9) * 1e3 + n_msg * lat / 1e3
    exposed_comm_ms = comm_ms * (1.0 - overlap)
    peak = PEAK_TFLOPS.get(gpu, 312.0) * 1e12
    compute_ms = compute_flops_per_rank(model, pc) / peak * 1e3
    ratio = exposed_comm_ms / (exposed_comm_ms + compute_ms) if (exposed_comm_ms + compute_ms) > 0 else 0.0

    return {
        "model": model.name,
        "config": pc.describe(),
        "zero_stage": pc.zero_stage if pc.zero_stage else "DDP",
        "dp": pc.dp, "tp": pc.tp, "pp": pc.pp,
        "link": link, "bw_gbps": bw, "latency_us": lat,
        "overlap_assumed": overlap,
        "dp_mb": round(d_bytes / 1e6, 2),
        "tp_mb": round(t_bytes / 1e6, 2),
        "pp_mb": round(p_bytes / 1e6, 2),
        "total_mb": round(total_bytes / 1e6, 2),
        "messages": n_msg,
        "comm_ms": round(comm_ms, 4),
        "exposed_comm_ms": round(exposed_comm_ms, 4),
        "compute_ms": round(compute_ms, 4),
        "comm_ratio": round(ratio, 4),
        "tp_msg_kb": round(
            (pc.batch * pc.seq * model.hidden * pc.dtype_bytes) / 1e3, 2
        ) if pc.tp > 1 else 0.0,
    }


def tp_break_even(
    model: ModelConfig,
    b_eff_gbps: float = 25.0,
    latency_us: float = 10.0,
    gpu: str = "a100",
    batch: int = 1,
    seq: int = 2048,
    dtype: str = "bf16",
    tp_values: Sequence[int] = (1, 2, 4, 8, 16),
) -> List[Dict[str, Any]]:
    """计算不同 TP 度数下的节省计算时间、新增通信时间与净收益（对应 docs/selection_boundary.md）。"""
    base_pc = ParallelConfig(dp=1, tp=1, pp=1, batch=batch, seq=seq, dtype=dtype)
    peak = PEAK_TFLOPS.get(gpu, 312.0) * 1e12
    c_base_ms = compute_flops_per_rank(model, base_pc) / peak * 1e3
    rows: List[Dict[str, Any]] = []
    for t in tp_values:
        if t <= 1:
            rows.append({
                "tp": 1, "b_eff_gbps": b_eff_gbps, "save_ms": 0.0,
                "comm_ms": 0.0, "net_gain_pct": 0.0, "profitable": True,
            })
            continue
        pc = ParallelConfig(dp=1, tp=t, pp=1, batch=batch, seq=seq, dtype=dtype)
        n_msg = tp_messages(model, pc)
        t_bytes = tp_bytes(model, pc)
        comm_ms = t_bytes / (b_eff_gbps * 1e9) * 1e3 + n_msg * latency_us / 1e3
        save_ms = c_base_ms * (1.0 - 1.0 / t)
        net_pct = (save_ms - comm_ms) / c_base_ms * 100.0 if c_base_ms > 0 else 0.0
        rows.append({
            "tp": t,
            "b_eff_gbps": b_eff_gbps,
            "save_ms": round(save_ms, 2),
            "comm_ms": round(comm_ms, 2),
            "net_gain_pct": round(net_pct, 2),
            "profitable": save_ms > comm_ms,
        })
    return rows


def solve_break_even_bw(
    model: ModelConfig,
    tp: int = 2,
    latency_us: float = 10.0,
    gpu: str = "a100",
    batch: int = 1,
    seq: int = 2048,
    dtype: str = "bf16",
    lo: float = 0.1,
    hi: float = 1000.0,
) -> float:
    """二分求解使 save_ms == comm_ms 的临界有效带宽 B_eff (GB/s)。"""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        res = tp_break_even(model, b_eff_gbps=mid, latency_us=latency_us,
                            gpu=gpu, batch=batch, seq=seq, dtype=dtype, tp_values=(tp,))
        if res[0]["save_ms"] > res[0]["comm_ms"]:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2.0, 2)


def infer_effective_bandwidth(
    total_bytes: float,
    comm_ms: float,
    n_msgs: int = 0,
    latency_us: float = 0.0,
) -> Optional[float]:
    """从实测通信时长反推有效带宽 B_eff (GB/s)，用于与 selection_boundary 对账。"""
    net_ms = comm_ms - n_msgs * latency_us / 1e3
    if total_bytes <= 0 or net_ms <= 0:
        return None
    return round(total_bytes / (net_ms * 1e-3) / 1e9, 2)


def default_configs(dp: int, tp: int, pp: int, stages: List[int],
                    batch: int, seq: int, dtype: str) -> List[ParallelConfig]:
    return [ParallelConfig(dp=dp, tp=tp, pp=pp, zero_stage=s, batch=batch,
                           seq=seq, dtype=dtype) for s in stages]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="理论通信账本与 TP 选型边界求解器")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--zero", default="0,1,2,3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--link", default="nvlink", choices=sorted(LINK_PROFILES))
    ap.add_argument("--bw", type=float, default=None, help="覆盖链路带宽 GB/s")
    ap.add_argument("--latency", type=float, default=None, help="覆盖消息延迟 us")
    ap.add_argument("--overlap", type=float, default=0.0, help="假设通信计算重叠率 [0, 1]")
    ap.add_argument("--gpu", default="a100", choices=sorted(PEAK_TFLOPS))
    ap.add_argument("--tp-sweep", action="store_true", help="输出 TP=1,2,4,8,16 盈亏平衡扫描与临界带宽")
    ap.add_argument("--nccl-csv", default="", help="可选：trace_parser.parse_nccl 产物，反推实测 B_eff")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    model = get_model(args.model)
    if args.tp_sweep:
        bw = args.bw if args.bw else LINK_PROFILES[args.link]["bw_gbps"]
        lat = args.latency if args.latency is not None else LINK_PROFILES[args.link]["latency_us"]
        crit_bw = solve_break_even_bw(model, tp=2, latency_us=lat, gpu=args.gpu,
                                      batch=args.batch, seq=args.seq, dtype=args.dtype)
        sweep = tp_break_even(model, b_eff_gbps=bw, latency_us=lat, gpu=args.gpu,
                              batch=args.batch, seq=args.seq, dtype=args.dtype)
        print(f"模型 {model.name}  B_eff={bw} GB/s  lat={lat} us  TP=2 盈亏平衡带宽 ≈ {crit_bw} GB/s")
        print(f"\n{'tp':>4}{'save ms':>10}{'comm ms':>10}{'net gain':>11}  profitable")
        print("-" * 48)
        for r in sweep:
            print(f"{r['tp']:>4}{r['save_ms']:>10.2f}{r['comm_ms']:>10.2f}{r['net_gain_pct']:>10.1f}%  {r['profitable']}")
        return 0

    if args.nccl_csv:
        p = Path(args.nccl_csv)
        if not p.exists():
            print(f"[SKIP] 找不到 {p}")
            return 0
        _, n_rows = io.read_csv(p)
        if not n_rows:
            print("[SKIP] nccl CSV 为空")
            return 0
        pc = ParallelConfig(dp=args.dp, tp=args.tp, pp=args.pp, zero_stage=0,
                            batch=args.batch, seq=args.seq, dtype=args.dtype)
        expected_bytes = dp_bytes(model, pc) + tp_bytes(model, pc) + pp_bytes(model, pc)
        busy_raw = n_rows[0].get("nccl_busy_ms") or n_rows[0].get("nccl_ms")
        try:
            busy_ms = float(str(busy_raw))
        except (TypeError, ValueError):
            print("[SKIP] nccl CSV 缺少有效的 nccl_busy_ms / nccl_ms（无 trace 数据）")
            return 0
        b_eff = infer_effective_bandwidth(expected_bytes, busy_ms)
        crit = solve_break_even_bw(model, tp=max(args.tp, 2), gpu=args.gpu,
                                   batch=args.batch, seq=args.seq, dtype=args.dtype)
        print(f"理论总通信量 {expected_bytes / 1e6:.1f} MB，实测 NCCL 时长 {busy_ms:.2f} ms"
              f" → 反推 B_eff = {b_eff} GB/s（TP=2 临界带宽 {crit} GB/s）")
        return 0

    stages = [int(s) for s in args.zero.split(",")]
    pcs = default_configs(args.dp, args.tp, args.pp, stages, args.batch,
                          args.seq, args.dtype)
    rows = [ledger(model, pc, args.link, args.bw, args.latency, args.gpu, args.overlap)
            for pc in pcs]

    env = collect(extra={"module": "ledger.comm_ledger", "model": args.model,
                         "link": args.link, "gpu": args.gpu})
    out = io.write_csv(io.result_path("comm_ledger", args.out), rows, env,
                       extra_meta={"batch": args.batch, "seq": args.seq,
                                   "overlap": args.overlap})
    print(f"模型 {model.name}  链路 {args.link}  "
          f"bw={rows[0]['bw_gbps']} GB/s  lat={rows[0]['latency_us']} us  overlap={args.overlap:.2f}")
    print(f"\n{'config':<28}{'dp MB':>9}{'tp MB':>9}{'msgs':>6}"
          f"{'comm ms':>9}{'compute ms':>11}{'ratio':>7}")
    print("-" * 79)
    for r in rows:
        print(f"{r['config']:<28}{r['dp_mb']:>9.1f}{r['tp_mb']:>9.1f}"
              f"{r['messages']:>6}{r['comm_ms']:>9.3f}{r['compute_ms']:>11.3f}"
              f"{r['comm_ratio']:>7.2%}")
    print(f"\n写出 {out}")
    if args.overlap == 0.0:
        print("注意：ratio 按「完全不重叠」计算，是上界；实际重叠率要用 trace 算"
              "（trace_parser.report）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
