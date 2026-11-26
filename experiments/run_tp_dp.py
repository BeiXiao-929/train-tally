"""experiments/run_tp_dp.py —— 实验 D：2-way TP × 2-way DP（张量并行的通信特征）。

文件职责
--------
这是全仓库最关键的实验：它直接产出「TP 超过几卡不划算」的一手数据。
TP 的通信特征是「次数多、单次小」——每层前向 2 次、反向 2 次 AllReduce，
单次只有 b·s·h·2 字节。带宽通常没跑满，**延迟先把时间吃光了**。

因此本实验必须记录两个东西：
1. 每步耗时随 tp 的变化（tp=1/2/4 的曲线）
2. 链路类型（NVLink / PCIe）—— 同一配置在两种链路上的临界 tp 差一个数量级

跑法
----
    python -m experiments.run_tp_dp --dry-run --steps 3 --tp 2 --dp 2
    torchrun --nproc_per_node=4 -m experiments.run_tp_dp --steps 100 --tp 2 --dp 2
    torchrun --nproc_per_node=8 -m experiments.run_tp_dp --steps 100 --tp 4 --dp 2

TODO
----
[ ] 增加 tp=8 与 pp 组合（需要 8 卡，云预算内单独排期）
[ ] 记录每层 AllReduce 的实测耗时（需要 trace + NVTX 标注）
[ ] 增加 sequence parallel（TP 的 SP 变体，能减少 30% 的激活通信）

验收标准
--------
- tp×dp 必须等于 world_size，否则直接报错并给出可用组合
- tp=1 时行为与 run_ddp 完全一致（同 seed 同 loss）
- 输出的 step_ms 随 tp 上升而上升（PCIe 上应非常明显），这是选型边界的原始证据
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import io
from common.env import collect, set_seed, topology
from experiments.tiny_model import (build_model, cleanup, setup_dist, summarize,
                                    train_steps)
from ledger.comm_ledger import ledger as comm_ledger
from ledger.memory_ledger import total_gb
from ledger.model_configs import ParallelConfig, get as get_model


def build_mesh(dp: int, tp: int, device_type: str):
    from torch.distributed.device_mesh import DeviceMesh

    world = dp * tp
    ranks = torch.arange(world).reshape(dp, tp)
    return DeviceMesh(device_type, ranks)


def get_group(mesh, dim: int):
    """兼容 2.4 的 get_group 与更早的 get_dim_groups。"""
    if hasattr(mesh, "get_group"):
        return mesh.get_group(dim)
    return mesh.get_dim_groups()[dim]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="实验 D：TP × DP")
    ap.add_argument("--model", default="tiny-debug")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--link", default="", help="覆盖链路判定（nvlink/pcie4），空则自动探测")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        args.steps = min(args.steps, 5)
        args.seq = min(args.seq, 64)
        args.model = "tiny-debug"
        args.tp, args.dp = 1, 1

    set_seed(args.seed)
    info = setup_dist(args.dry_run)

    if args.tp * args.dp != info.world_size:
        print(f"[FAIL] tp({args.tp}) × dp({args.dp}) = {args.tp * args.dp} "
              f"≠ world_size({info.world_size})。4 卡可用：(tp,dp) ∈ "
              f"{{(1,4),(2,2),(4,1)}}", file=sys.stderr)
        cleanup()
        return 2

    cfg = get_model(args.model)
    if cfg.n_kv_heads % args.tp != 0:
        print(f"[FAIL] kv_heads={cfg.n_kv_heads} 不能被 tp={args.tp} 整除，"
              f"GQA 下 kv 头数也要能切", file=sys.stderr)
        cleanup()
        return 2

    link = args.link or topology()["interconnect"]
    try:
        mesh = None
        tp_group = None
        if info.world_size > 1:
            mesh = build_mesh(args.dp, args.tp, "cuda" if info.device.type == "cuda"
                              else "cpu")
            tp_group = get_group(mesh, 1)
            dp_group = get_group(mesh, 0)

        model = build_model(cfg, tp=args.tp, tp_group=tp_group, device=info.device)

        if info.world_size > 1:
            model = DDP(model, process_group=dp_group,
                        device_ids=[info.local_rank] if info.device.type == "cuda" else None)
        elif info.is_main:
            print("[dry-run] 单进程：tp=dp=1，只验证模型可切分与循环可跑")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rows = train_steps(model, optimizer, steps=args.steps, batch=args.batch,
                           seq=args.seq, vocab=cfg.vocab, device=info.device,
                           seed=args.seed, warmup=min(2, max(args.steps // 3, 1)),
                           log=info.is_main)
        stats = summarize(rows, info.world_size)

        pc = ParallelConfig(dp=args.dp, tp=args.tp, zero_stage=0,
                            dtype=args.dtype, batch=args.batch, seq=args.seq)
        predicted = total_gb(cfg, pc)["total_gb"]
        comm = comm_ledger(cfg, pc, link=link if link in
                           ("nvlink", "pcie4", "pcie5", "ib100", "ib200") else "nvlink")
        measured = stats["peak_mem_gb"]
        stats.update({
            "exp": "tp_dp",
            "config": pc.describe(),
            "world_size": info.world_size,
            "link": link,
            "tp_msgs": comm["messages"],
            "tp_msg_kb": comm["tp_msg_kb"],
            "pred_comm_ratio": comm["comm_ratio"],
            "predicted_gb": round(predicted, 3),
            "error_pct": (round((measured - predicted) / predicted * 100, 2)
                          if measured and predicted else io.FILL),
        })

        if info.is_main:
            env = collect(extra={"module": "experiments.run_tp_dp",
                                 "model": cfg.name, "link": link})
            out = io.write_csv(io.result_path("tp_dp", args.out), [stats], env,
                               extra_meta={"steps": args.steps})
            print(f"\n[summary] {stats}")
            print(f"写出 {out}")
        if args.dry_run and info.is_main:
            print("dry-run: PASS")
        if info.world_size > 1:
            dist.barrier()
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
