"""experiments/run_zero.py —— 实验 C：ZeRO-1 / 2 / 3 的显存—通信权衡曲线。

文件职责
--------
同一模型、同一 batch，只换 ZeRO 档位，画出「显存下降 vs 吞吐下降」的曲线。
这张曲线是选型边界的第一手数据。

实现选择（必须写进报告，否则会被认为是「没跑过真的 ZeRO」）
--------------------------------------------------------
- **ZeRO-1**：`torch.distributed.optim.ZeroRedundancyOptimizer`，真 ZeRO-1
  （只切优化器状态，每步 AllGather 参数分片）
- **ZeRO-2**：`fully_shard(reshard_after_forward=False)`，等价语义
  （梯度分片 + 前向后不释放参数分片）
- **ZeRO-3**：`fully_shard(reshard_after_forward=True)`，等价 FSDP 全分片

不用 DeepSpeed 的理由：它会把「框架行为」混进「并行策略」的对比里
（bucket 大小、offload 默认值、通信 hook 全都不一样），而且 4 卡小模型上
DS 的 eager 模式收益不稳定。**本仓库的结论只用于趋势与量级，不声称复现 DeepSpeed。**

跑法
----
    python -m experiments.run_zero --dry-run --steps 3 --stages 1,2,3
    torchrun --nproc_per_node=4 -m experiments.run_zero --steps 100 --stages 1,2,3

TODO
----
[ ] 增加 DeepSpeed 分支（--backend deepspeed），把两套实现放在同一张图上对比
[ ] 增加 `--offload` 对照（ZeRO-3 + CPU offload 的显存/吞吐曲线）
[ ] 记录每档的 AllGather 次数（用 trace 统计，当前只记理论值）

验收标准
--------
- 三档显存严格递减：ZeRO-1 > ZeRO-2 > ZeRO-3
- 三档 loss 曲线互相对齐（同 seed 下前 100 步相对偏差 < 1e-3）
- 每档输出一行 CSV，可被 ledger.mfu_ledger 与 trace_parser 消费
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import torch

from common import io
from common.env import collect, set_seed
from experiments.tiny_model import (build_model, cleanup, setup_dist, summarize,
                                    train_steps)
from ledger.memory_ledger import total_gb
from ledger.model_configs import ParallelConfig, get as get_model

STAGE_NAME = {1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


def wrap_zero1(model, lr: float):
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=lr,
        overlap_with_ddp=False,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="实验 C：ZeRO-1/2/3")
    ap.add_argument("--model", default="tiny-debug")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        args.steps = min(args.steps, 5)
        args.seq = min(args.seq, 64)
        args.model = "tiny-debug"

    set_seed(args.seed)
    info = setup_dist(args.dry_run)
    cfg = get_model(args.model)
    stages = [int(s) for s in args.stages.split(",")]

    all_stats: List[Dict[str, Any]] = []
    try:
        for stage in stages:
            if info.is_main:
                print(f"\n=== {STAGE_NAME.get(stage, stage)} ===")
            # 每档重建模型，避免上一档的优化器状态/分片残留影响显存峰值
            model = build_model(cfg, tp=1, device=info.device)
            wrapped = False
            if info.world_size > 1 and info.device.type == "cuda":
                if stage == 1:
                    optimizer = wrap_zero1(model, args.lr)
                else:
                    from experiments.run_fsdp2 import wrap_fsdp2

                    model = wrap_fsdp2(model,
                                       reshard_after_forward=(stage == 3),
                                       bf16=(args.dtype == "bf16"))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                wrapped = True
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                if info.is_main:
                    print("[dry-run] 单进程/CPU：不包装（ZeRO-1 需 dist，ZeRO-2/3 需 CUDA），"
                          "只验证循环")

            rows = train_steps(model, optimizer, steps=args.steps,
                               batch=args.batch, seq=args.seq, vocab=cfg.vocab,
                               device=info.device, seed=args.seed,
                               warmup=min(2, max(args.steps // 3, 1)),
                               log=info.is_main)
            stats = summarize(rows, info.world_size)

            pc = ParallelConfig(dp=info.world_size, zero_stage=stage,
                                dtype=args.dtype, batch=args.batch, seq=args.seq)
            predicted = total_gb(cfg, pc)["total_gb"]
            measured = stats["peak_mem_gb"]
            stats.update({
                "exp": f"zero{stage}",
                "config": pc.describe() + ("" if wrapped else "(unwrapped)"),
                "world_size": info.world_size,
                "wrapped": wrapped,
                "predicted_gb": round(predicted, 3),
                "error_pct": (round((measured - predicted) / predicted * 100, 2)
                              if measured and predicted else io.FILL),
            })
            all_stats.append(stats)

            if info.is_main:
                env = collect(extra={"module": "experiments.run_zero",
                                     "model": cfg.name,
                                     "stage": stage})
                out = io.write_csv(io.result_path(f"zero{stage}", args.out),
                                   [stats], env,
                                   extra_meta={"steps": args.steps})
                print(f"[summary] {stats}\n写出 {out}")

        if info.is_main and len(all_stats) > 1:
            mem = [s["peak_mem_gb"] for s in all_stats]
            decreasing = all(mem[i] > mem[i + 1] for i in range(len(mem) - 1))
            print(f"\n显存单调递减检查：{mem} -> {'PASS' if decreasing else 'FAIL'}"
                  "（未包装时该项不可判）")
        if args.dry_run and info.is_main:
            print("dry-run: PASS")
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
