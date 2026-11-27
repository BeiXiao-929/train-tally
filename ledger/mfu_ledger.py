"""ledger/mfu_ledger.py —— 理论 MFU 上界与实测差距。

文件职责
--------
1. 算「理论上界」：假设算力 100% 利用、通信完全不重叠时，MFU 最多是多少
2. 用 comm_ledger 的通信占比给出**通信约束下的上界**：`1/(1+comm_ratio)`
3. 用实测吞吐算实际 MFU，输出 gap，并把它拆成「通信吃掉」与「其他」两块

MFU 定义（6PD 近似）
--------------------
    MFU = 6 × P × tokens_per_sec / (peak_flops × n_gpus)
前向 2PD、反向 4PD 是标准近似。报告里必须注明这是近似值：
精确值要按实际 FLOPs 统计（含 attention 的 2·s²·d 项与未计入的 norm/激活）。

为什么要算上界而不是直接报 MFU
-----------------------------
只报 MFU=42% 会被追问「42% 是好还是坏」。给出上界（例如 58%）之后，
「离上界还差 16 个点，其中 11 个点来自通信、5 个点来自负载不均」才是结论。

TODO
----
[ ] 增加 attention 的 FLOPs 修正项（长序列时 2·b·s²·d 不可忽略）
[ ] 增加「负载不均」的估计（PP 的 bubble 比例 = (p-1)/(m+p-1)）
[ ] 支持从多个实测 CSV 批量算 gap 表

验收标准
--------
- 单卡、无通信时上界 = 100%（1.0），有 TP/DP 时严格下降
- 实测 MFU 不得超过上界；若超过说明 tokens/s 或卡数统计错了
- 输出的 gap 分解中，通信块 + 其他块 ≈ 总 gap（误差 < 2 个百分点）
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from common import io
from common.env import collect
from ledger.comm_ledger import PEAK_TFLOPS, ledger as comm_ledger
from ledger.model_configs import ModelConfig, ParallelConfig, get as get_model


def theoretical_mfu_upper_bound(model: ModelConfig, pc: ParallelConfig,
                                link: str = "nvlink", gpu: str = "a100",
                                overlap: float = 0.0) -> Dict[str, float]:
    """返回 {pure_compute, with_comm}。

    pure_compute：通信完全不发生时（单卡或完美重叠）的上界，恒为 1.0
    with_comm   ：通信完全不重叠时的上界 = compute/(compute+comm)
    overlap     ：0 表示完全不重叠，1 表示完美重叠；中间值按线性折算
    """
    c = comm_ledger(model, pc, link=link, gpu=gpu)
    comm_ms, compute_ms = c["comm_ms"], c["compute_ms"]
    total = compute_ms + comm_ms * (1.0 - overlap)
    with_comm = compute_ms / total if total > 0 else 0.0
    return {
        "comm_ms": comm_ms,
        "compute_ms": compute_ms,
        "comm_ratio": c["comm_ratio"],
        "upper_bound_pure": 1.0,
        "upper_bound_with_comm": round(with_comm, 4),
        "overlap_assumed": overlap,
    }


def measured_mfu(model: ModelConfig, tokens_per_s: float, n_gpus: int,
                 gpu: str = "a100") -> float:
    peak = PEAK_TFLOPS.get(gpu, 312.0) * 1e12
    return 6.0 * model.total_params() * 1e9 * tokens_per_s / (peak * n_gpus)


def rows_for(model: ModelConfig, pcs: List[ParallelConfig], link: str,
             gpu: str, measured_csv: str = "") -> List[Dict[str, Any]]:
    measured: Dict[str, float] = {}
    if measured_csv:
        _, data = io.read_csv(measured_csv)
        for r in data:
            key = str(r.get("config", "")).strip()
            try:
                measured[key] = float(r["tokens_per_s"])
            except (KeyError, ValueError):
                continue

    rows = []
    for pc in pcs:
        ub = theoretical_mfu_upper_bound(model, pc, link=link, gpu=gpu)
        row: Dict[str, Any] = {
            "model": model.name,
            "config": pc.describe(),
            "gpu": gpu,
            "link": link,
            "comm_ratio": ub["comm_ratio"],
            "upper_bound_mfu": ub["upper_bound_with_comm"],
        }
        tps = measured.get(pc.describe())
        if tps is None:
            row["measured_tokens_per_s"] = io.FILL
            row["measured_mfu"] = io.FILL
            row["gap_pp"] = io.FILL
            row["gap_from_comm_pp"] = io.FILL
            row["gap_other_pp"] = io.FILL
        else:
            m = measured_mfu(model, tps, pc.world_size, gpu)
            gap = ub["upper_bound_with_comm"] - m
            row["measured_tokens_per_s"] = round(tps, 2)
            row["measured_mfu"] = round(m, 4)
            row["gap_pp"] = round(gap * 100, 2)
            row["gap_from_comm_pp"] = round(ub["comm_ratio"] * 100, 2)
            row["gap_other_pp"] = round(gap * 100 - ub["comm_ratio"] * 100, 2)
        rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MFU 上界与实测差距")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--zero", default="0,1,2,3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--link", default="nvlink")
    ap.add_argument("--gpu", default="a100", choices=sorted(PEAK_TFLOPS))
    ap.add_argument("--measured-csv", default="",
                    help="含 config 与 tokens_per_s 列的实测 CSV（experiments 产出）")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    model = get_model(args.model)
    stages = [int(s) for s in args.zero.split(",")]
    pcs = [ParallelConfig(dp=args.dp, tp=args.tp, pp=args.pp, zero_stage=s,
                          batch=args.batch, seq=args.seq) for s in stages]
    rows = rows_for(model, pcs, args.link, args.gpu, args.measured_csv)

    env = collect(extra={"module": "ledger.mfu_ledger", "model": args.model,
                         "link": args.link, "gpu_target": args.gpu})
    out = io.write_csv(io.result_path("mfu_ledger", args.out), rows, env)
    print(f"模型 {model.name}  GPU={args.gpu}  链路={args.link}")
    print(f"{'config':<28}{'comm ratio':>12}{'UB MFU':>9}   {'meas MFU':>9} {'gap pp':>7}")
    print("-" * 67)
    for r in rows:
        meas = r["measured_mfu"]
        meas_str = meas if isinstance(meas, str) else f"{meas:.2%}"
        gap_str = str(r["gap_pp"])
        # 每列之间显式留空格：__FILL__ 正好等于列宽，不留空格会粘成 __FILL____FILL__
        print(f"{r['config']:<28}{r['comm_ratio']:>12.2%}{r['upper_bound_mfu']:>9.2%}"
              f"   {meas_str:>9} {gap_str:>7}")
    print(f"\n写出 {out}")
    print("判读：实测 MFU 超过上界 → 统计错了（卡数或 tokens/s）；"
          "gap 远大于通信占比 → 查负载不均、显存受限、dataloader。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
