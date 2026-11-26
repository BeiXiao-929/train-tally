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
TP（每层前向 2 次 + 反向 2 次 AllReduce）：
    4 × L × b×s×h × bytes × (t-1)/t      ← 次数多、单次小，对延迟敏感
PP（每个 stage 边界每次收发激活）：
    2 × b×s×h × bytes × (p-1)（fwd+bwd 各一次）

**为什么 TP 在 PCIe 上不划算**：TP 的消息数是 `4×L`（7B/28 层 = 112 次/步），
单次只有 b×s×h×2 字节。带宽还没跑满，延迟已经把时间吃光了。
这就是 docs/selection_boundary.md 要量化的临界点。

TODO
----
[ ] 增加通信与计算的重叠建模（当前 ratio 按完全不重叠算，是上界）
[ ] 增加 hierarchical AllReduce（多机时 ring 会跨机，实际带宽更低）
[ ] 支持 --nccl-algo LL/LL128/Simple 的带宽修正系数
[ ] 从实测 trace 反推有效带宽（与 parse_nccl 的结果对账）

验收标准
--------
- 4 卡 DDP 的 AllReduce 字节数 = 2×P×(3/4)，与手算一致（误差 < 0.1%）
- TP=1 时 TP 项为 0；TP=4 的通信时间应显著高于 TP=2（同一链路）
- 换链路（pcie4 → nvlink）时 TP 通信时间下降一个数量级
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

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
    return 4.0 * model.n_layers * msg * (t - 1) / t


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
           gpu: str = "a100") -> Dict[str, Any]:
    prof = LINK_PROFILES.get(link, LINK_PROFILES["nvlink"])
    bw = bw_gbps if bw_gbps else prof["bw_gbps"]
    lat = latency_us if latency_us is not None else prof["latency_us"]

    d_bytes = dp_bytes(model, pc)
    t_bytes = tp_bytes(model, pc)
    p_bytes = pp_bytes(model, pc)
    n_msg = tp_messages(model, pc) + (2 * max(pc.pp - 1, 0)) + (1 if pc.dp > 1 else 0)

    total_bytes = d_bytes + t_bytes + p_bytes
    comm_ms = total_bytes / (bw * 1e9) * 1e3 + n_msg * lat / 1e3
    peak = PEAK_TFLOPS.get(gpu, 312.0) * 1e12
    compute_ms = compute_flops_per_rank(model, pc) / peak * 1e3
    ratio = comm_ms / (comm_ms + compute_ms) if (comm_ms + compute_ms) > 0 else 0.0

    return {
        "model": model.name,
        "config": pc.describe(),
        "zero_stage": pc.zero_stage if pc.zero_stage else "DDP",
        "dp": pc.dp, "tp": pc.tp, "pp": pc.pp,
        "link": link, "bw_gbps": bw, "latency_us": lat,
        "dp_mb": round(d_bytes / 1e6, 2),
        "tp_mb": round(t_bytes / 1e6, 2),
        "pp_mb": round(p_bytes / 1e6, 2),
        "total_mb": round(total_bytes / 1e6, 2),
        "messages": n_msg,
        "comm_ms": round(comm_ms, 4),
        "compute_ms": round(compute_ms, 4),
        "comm_ratio": round(ratio, 4),
        "tp_msg_kb": round(
            (pc.batch * pc.seq * model.hidden * pc.dtype_bytes) / 1e3, 2
        ) if pc.tp > 1 else 0.0,
    }


def default_configs(dp: int, tp: int, pp: int, stages: List[int],
                    batch: int, seq: int, dtype: str) -> List[ParallelConfig]:
    return [ParallelConfig(dp=dp, tp=tp, pp=pp, zero_stage=s, batch=batch,
                           seq=seq, dtype=dtype) for s in stages]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="理论通信账本")
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
    ap.add_argument("--gpu", default="a100", choices=sorted(PEAK_TFLOPS))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    model = get_model(args.model)
    stages = [int(s) for s in args.zero.split(",")]
    pcs = default_configs(args.dp, args.tp, args.pp, stages, args.batch,
                          args.seq, args.dtype)
    rows = [ledger(model, pc, args.link, args.bw, args.latency, args.gpu)
            for pc in pcs]

    env = collect(extra={"module": "ledger.comm_ledger", "model": args.model,
                         "link": args.link, "gpu": args.gpu})
    out = io.write_csv(io.result_path("comm_ledger", args.out), rows, env,
                       extra_meta={"batch": args.batch, "seq": args.seq})
    print(f"模型 {model.name}  链路 {args.link}  "
          f"bw={rows[0]['bw_gbps']} GB/s  lat={rows[0]['latency_us']} us")
    print(f"\n{'config':<28}{'dp MB':>9}{'tp MB':>9}{'msgs':>6}"
          f"{'comm ms':>9}{'compute ms':>11}{'ratio':>7}")
    print("-" * 79)
    for r in rows:
        print(f"{r['config']:<28}{r['dp_mb']:>9.1f}{r['tp_mb']:>9.1f}"
              f"{r['messages']:>6}{r['comm_ms']:>9.3f}{r['compute_ms']:>11.3f}"
              f"{r['comm_ratio']:>7.2%}")
    print(f"\n写出 {out}")
    print("注意：ratio 按「完全不重叠」计算，是上界；实际重叠率要用 trace 算"
          "（trace_parser.report）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
