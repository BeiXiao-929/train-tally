"""experiments/run_ddp.py —— 实验 A：纯数据并行（通信基线）。

文件职责
--------
建立**通信基线**：DDP 的 AllReduce 是所有并行策略里最少通信的一种，
后续 FSDP/ZeRO/TP 的通信增量都要拿它当分母。

观察重点
--------
1. `peak_mem_gb` 与账本预测的误差（目标 < 10%）
2. `tokens_per_s_global` 随卡数的增长是否次线性、拐点能否被通信量解释
3. `grad_norm` 与单卡基线的偏差（< 1e-2）

跑法
----
    # 单进程验证脚本逻辑（CPU 也能跑，3 步）
    python -m experiments.run_ddp --dry-run --steps 3

    # 4 卡
    torchrun --nproc_per_node=4 -m experiments.run_ddp --steps 100

TODO
----
[ ] 增加 `--bucket-cap-mb` 扫描（DDP 的桶大小直接影响重叠率）
[ ] 增加 `static_graph=True` 对照（影响 AllReduce 的调度时机）
[ ] 与梯度累积配合，验证「等效 batch 下的通信占比是否下降」

验收标准
--------
- 4 卡与单卡的 loss 在前 200 步相对偏差 < 1e-3（同 seed、同 batch 定义）
- peak_mem 与 memory_ledger 预测误差 < 10%
- 输出 CSV 含 config 与 tokens_per_s 列，可被 ledger.mfu_ledger 直接消费
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from common import io
from common.env import collect, set_seed
from experiments.tiny_model import (build_model, cleanup, setup_dist, summarize,
                                    train_steps)
from ledger.memory_ledger import total_gb
from ledger.model_configs import ParallelConfig, get as get_model


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="实验 A：DDP 通信基线")
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
        if info.world_size > 1:
            ddp_kwargs: Dict[str, Any] = {}
            if info.device.type == "cuda":
                ddp_kwargs["device_ids"] = [info.local_rank]
            model = DDP(model, **ddp_kwargs)
            if info.is_main:
                print(f"[ddp] world={info.world_size} backend={info.backend}")
        elif args.dry_run and info.is_main:
            print("[dry-run] world_size=1：不包 DDP，只验证训练循环与落盘")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rows = train_steps(model, optimizer, steps=args.steps, batch=args.batch,
                           seq=args.seq, vocab=cfg.vocab, device=info.device,
                           seed=args.seed, warmup=min(2, max(args.steps // 3, 1)),
                           log=info.is_main)
        stats = summarize(rows, info.world_size)

        pc = ParallelConfig(dp=info.world_size, tp=1, pp=1, zero_stage=0,
                            dtype=args.dtype, batch=args.batch, seq=args.seq)
        predicted = total_gb(cfg, pc)["total_gb"]
        measured = stats["peak_mem_gb"]
        stats.update({
            "exp": "ddp",
            "config": pc.describe(),
            "world_size": info.world_size,
            "predicted_gb": round(predicted, 3),
            "error_pct": (round((measured - predicted) / predicted * 100, 2)
                          if measured and predicted else io.FILL),
        })

        if info.is_main:
            env = collect(extra={"module": "experiments.run_ddp",
                                 "model": cfg.name})
            out = io.write_csv(io.result_path("ddp", args.out), [stats], env,
                               extra_meta={"steps": args.steps,
                                           "batch": args.batch,
                                           "seq": args.seq})
            print(f"\n[summary] {stats}")
            print(f"写出 {out}")
        if args.dry_run and info.is_main:
            print("dry-run: PASS")
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
