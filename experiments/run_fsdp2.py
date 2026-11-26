"""experiments/run_fsdp2.py —— 实验 B：FSDP2 全分片（参数分片的通信增量）。

文件职责
--------
在 FSDP2（`fully_shard`）下跑同一套训练循环，观察：
1. 参数被切成 1/N 后，显存下降幅度是否与账本一致（ZeRO-3 口径）
2. AllGather / ReduceScatter 的通信增量，以及它对 step 时间的贡献
3. 与 DDP 基线的 loss 是否对齐（前 200 步相对偏差 < 1e-3）

FSDP2 与 FSDP1 的关键差异
------------------------
FSDP2 用 `fully_shard(module)` 逐模块包装（per-module sharding），
不再有 FlatParameter；`reshard_after_forward=False` 等价于 ZeRO-2 的语义
（前向后不立即释放分片）。这也是 run_zero.py 复用它的原因。

跑法
----
    python -m experiments.run_fsdp2 --dry-run --steps 3
    torchrun --nproc_per_node=4 -m experiments.run_fsdp2 --steps 100

TODO
----
[ ] 增加 `--mp-policy bf16` 的完整对照（当前只传 param/reduce dtype，未测 cast 开销）
[ ] 增加 `--cpu-offload` 对照（FSDP2 支持 CPUOffloadPolicy）
[ ] 记录每层的 AllGather 时间（需要 trace，当前只看端到端）

验收标准
--------
- 4 卡 FSDP2 的 peak_mem 与账本 ZeRO-3 预测误差 < 10%
- loss 与 DDP 基线在前 200 步相对偏差 < 1e-3
- dry-run 在 CPU 单进程下跑通（跳过 FSDP 包装，只验证循环与落盘）
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

import torch

from common import io
from common.env import collect, set_seed
from experiments.tiny_model import (build_model, cleanup, setup_dist, summarize,
                                    train_steps)
from ledger.memory_ledger import total_gb
from ledger.model_configs import ParallelConfig, get as get_model


def wrap_fsdp2(model, reshard_after_forward: bool = True,
               bf16: bool = False):
    """逐 block 包装再包根模块（FSDP2 的推荐顺序）。"""
    from torch.distributed.fsdp import fully_shard

    policy = None
    if bf16:
        try:
            from torch.distributed.fsdp import MixedPrecisionPolicy

            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                          reduce_dtype=torch.float32)
        except Exception:
            policy = None
    kwargs: dict = {"reshard_after_forward": reshard_after_forward}
    if policy is not None:
        kwargs["mp_policy"] = policy
    for blk in model.blocks:
        fully_shard(blk, **kwargs)
    fully_shard(model, **kwargs)
    return model


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="实验 B：FSDP2 全分片")
    ap.add_argument("--model", default="tiny-debug")
    ap.add_argument("--steps", type=int, default=100)
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

    try:
        model = build_model(cfg, tp=1, device=info.device)
        wrapped = False
        if info.world_size > 1 and info.device.type == "cuda":
            model = wrap_fsdp2(model, reshard_after_forward=True,
                               bf16=(args.dtype == "bf16"))
            wrapped = True
        elif info.is_main:
            print("[dry-run] 单进程/CPU：跳过 FSDP2 包装（FSDP2 需 CUDA + 多进程），"
                  "仅验证训练循环与落盘")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rows = train_steps(model, optimizer, steps=args.steps, batch=args.batch,
                           seq=args.seq, vocab=cfg.vocab, device=info.device,
                           seed=args.seed, warmup=min(2, max(args.steps // 3, 1)),
                           log=info.is_main)
        stats = summarize(rows, info.world_size)

        pc = ParallelConfig(dp=info.world_size, tp=1, pp=1, zero_stage=3,
                            dtype=args.dtype, batch=args.batch, seq=args.seq)
        predicted = total_gb(cfg, pc)["total_gb"]
        measured = stats["peak_mem_gb"]
        stats.update({
            "exp": "fsdp2",
            "config": pc.describe() + ("" if wrapped else "(unwrapped)"),
            "world_size": info.world_size,
            "fsdp_wrapped": wrapped,
            "predicted_gb": round(predicted, 3),
            "error_pct": (round((measured - predicted) / predicted * 100, 2)
                          if measured and predicted else io.FILL),
        })

        if info.is_main:
            env = collect(extra={"module": "experiments.run_fsdp2",
                                 "model": cfg.name})
            out = io.write_csv(io.result_path("fsdp2", args.out), [stats], env,
                               extra_meta={"steps": args.steps})
            print(f"\n[summary] {stats}")
            print(f"写出 {out}")
        if args.dry_run and info.is_main:
            print("dry-run: PASS")
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
