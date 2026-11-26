"""ledger/memory_ledger.py —— 理论显存账本（输入配置，输出每卡显存分解）。

文件职责
--------
事前算出「每张卡该占多少显存」，拆成四项：参数 / 梯度 / 优化器状态 / 激活。
这是副项目两个核心交付物之一：跑通 FSDP 网上教程遍地都是，
能事前算准显存才是壁垒。

四字节口径（混合精度 + Adam，每参数）
------------------------------------
    bf16 参数  2  |  bf16 梯度  2  |  fp32 主权重 4  |  Adam m 4  |  Adam v 4
                                                        → 全量 16 B/param
    ZeRO-1: 优化器 12/N  |  ZeRO-2: 优化器 12/N + 梯度 2/N
    ZeRO-3: 全部 16/N

激活的算法（这里与「除以 sqrt(L)」那种手wave不同）
------------------------------------------------
- 不做重计算：存每层所有中间张量，`acts = L/pp × 每层元素数 × b × bytes`
- 做重计算（AC）：只存每层**输入**（block 边界），外加一层重算时的峰值，
  所以 `acts ≈ L/pp × b×s×h × bytes + 单层峰值`。这就是 30%–60% 显存下降的来源。

误差预期：< 10%（验收线）。主要来源是 caching allocator 的碎片与临时 workspace，
所以引入 `--fragmentation`（默认 1.05），并用实测值反推校准。

TODO
----
[ ] 增加 FP8 训练（主权重仍 fp32，但参数/激活 1 字节）
[ ] 增加 MoE 的专家参数与 all2all 缓冲
[ ] 支持 --calibrate：从实测反推 fragmentation，而不是用固定 1.05
[ ] 把 attention 的 flash 实现差异做成显式开关（当前按 eager 计 scores 张量）

验收标准
--------
- qwen2.5-7b + ZeRO-3 + 4 卡：理论值 37.483 GB（含 1.05 碎片系数），与实测误差 < 10%
- 同一配置下 ZeRO-1 > ZeRO-2 > ZeRO-3 的显存必须严格递减
- AC 开关打开后激活项下降 50% 以上（相对不开）
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from common import io
from common.env import collect
from ledger.model_configs import ModelConfig, ParallelConfig, get as get_model

GB = 1024 ** 3


def state_bytes(model: ModelConfig, pc: ParallelConfig,
                master_weights: bool = True) -> Dict[str, float]:
    """返回每卡的参数/梯度/优化器字节数（未含激活）。"""
    params = model.total_params() * 1e9
    # TP 与 PP 先把参数切走：TP 切矩阵、PP 切层（近似按比例）
    p_rank = params / (pc.tp * pc.pp)
    b = pc.dtype_bytes

    param = p_rank * b
    grad = p_rank * (2 if pc.dtype != "fp32" else 4)
    opt = p_rank * (12 if master_weights else 8)

    n = max(pc.dp, 1)
    if pc.zero_stage == 0:
        pass
    elif pc.zero_stage == 1:
        opt /= n
    elif pc.zero_stage == 2:
        opt /= n
        grad /= n
    elif pc.zero_stage == 3:
        opt /= n
        grad /= n
        param /= n
    else:
        raise ValueError(f"不支持的 zero_stage={pc.zero_stage}")

    return {"params": param, "grads": grad, "optimizer": opt}


def activation_elems_per_layer(model: ModelConfig, pc: ParallelConfig,
                               attn_impl: str = "flash") -> float:
    """单层单样本的激活元素数（已按 TP 切分）。"""
    s = pc.seq
    h = model.hidden
    kv = model.kv_dim
    ffn = model.ffn / pc.tp
    heads = model.n_heads / pc.tp

    qkv_out = s * (h + 2 * kv)          # q/k/v 的输出（k/v 是 GQA，按真实维度算）
    attn_out = s * h
    ffn_up = s * 2 * ffn                # gate + up
    ffn_act = s * ffn                   # silu 输出
    residual = s * h                    # 残差流
    scores = s * s * heads if attn_impl == "eager" else 0.0
    return qkv_out + attn_out + ffn_up + ffn_act + residual + scores


def activation_bytes(model: ModelConfig, pc: ParallelConfig,
                     attn_impl: str = "flash") -> float:
    layers_per_rank = model.n_layers / pc.pp
    per_layer = activation_elems_per_layer(model, pc, attn_impl)
    if pc.activation_checkpointing:
        # 只存 block 边界 + 一层重算峰值
        stored = layers_per_rank * pc.seq * model.hidden + per_layer
    else:
        stored = layers_per_rank * per_layer
    return stored * pc.batch * pc.dtype_bytes


def total_gb(model: ModelConfig, pc: ParallelConfig, attn_impl: str = "flash",
             fragmentation: float = 1.05) -> Dict[str, float]:
    st = state_bytes(model, pc)
    act = activation_bytes(model, pc, attn_impl)
    total = (sum(st.values()) + act) * fragmentation
    return {
        "params_gb": st["params"] / GB,
        "grads_gb": st["grads"] / GB,
        "optimizer_gb": st["optimizer"] / GB,
        "activations_gb": act / GB,
        "total_gb": total / GB,
        "state_gb": sum(st.values()) / GB,
    }


def rows_for(model: ModelConfig, pcs: List[ParallelConfig],
             attn_impl: str = "flash", fragmentation: float = 1.05) -> List[Dict[str, Any]]:
    rows = []
    for pc in pcs:
        r = total_gb(model, pc, attn_impl, fragmentation)
        rows.append({
            "model": model.name,
            "zero_stage": pc.zero_stage if pc.zero_stage else "DDP",
            "dp": pc.dp, "tp": pc.tp, "pp": pc.pp,
            "batch": pc.batch, "seq": pc.seq, "dtype": pc.dtype,
            "activation_ckpt": pc.activation_checkpointing,
            "params_gb": round(r["params_gb"], 3),
            "grads_gb": round(r["grads_gb"], 3),
            "optimizer_gb": round(r["optimizer_gb"], 3),
            "activations_gb": round(r["activations_gb"], 3),
            "total_gb": round(r["total_gb"], 3),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="理论显存账本")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--dp", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--zero", default="0,1,2,3", help="逗号分隔的 ZeRO stage")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ac", action="store_true", help="开启激活重计算")
    ap.add_argument("--attn-impl", default="flash", choices=["flash", "eager"])
    ap.add_argument("--fragmentation", type=float, default=1.05)
    ap.add_argument("--measured-csv", default="", help="含 measured_gb 列的实测 CSV，用于算误差")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    model = get_model(args.model)
    stages = [int(s) for s in args.zero.split(",")]
    pcs = [ParallelConfig(dp=args.dp, tp=args.tp, pp=args.pp, zero_stage=s,
                          activation_checkpointing=args.ac, dtype=args.dtype,
                          batch=args.batch, seq=args.seq) for s in stages]

    rows = rows_for(model, pcs, args.attn_impl, args.fragmentation)

    if args.measured_csv:
        _, measured = io.read_csv(args.measured_csv)
        by_key = {(r.get("zero_stage"), str(r.get("dp"))): r for r in measured}
        for r in rows:
            m = by_key.get((str(r["zero_stage"]), str(r["dp"])))
            if not m:
                r["measured_gb"] = io.FILL
                r["error_pct"] = io.FILL
                continue
            try:
                mg = float(m["measured_gb"])
            except (KeyError, ValueError):
                r["measured_gb"] = io.FILL
                r["error_pct"] = io.FILL
                continue
            r["measured_gb"] = round(mg, 3)
            r["error_pct"] = round((r["total_gb"] - mg) / mg * 100, 2)

    env = collect(extra={"module": "ledger.memory_ledger",
                         "model": args.model,
                         "fragmentation": args.fragmentation})
    out = io.write_csv(io.result_path("memory_ledger", args.out), rows, env,
                       extra_meta={"attn_impl": args.attn_impl,
                                   "ac": args.ac})
    print(f"模型 {model.name}  解析参数量 {model.total_params():.3f} B"
          + (f"（与公开值偏差 {model.deviation_pct():+.2f}%）"
             if model.deviation_pct() is not None else ""))
    print(f"\n{'ZeRO':<6}{'params':>10}{'grads':>10}{'opt':>10}{'act':>10}{'total':>10}")
    print("-" * 56)
    for r in rows:
        print(f"{str(r['zero_stage']):<6}{r['params_gb']:>10.3f}{r['grads_gb']:>10.3f}"
              f"{r['optimizer_gb']:>10.3f}{r['activations_gb']:>10.3f}{r['total_gb']:>10.3f}")
    print(f"\n写出 {out}")
    print("验收：与实测 max_memory_allocated 的误差需 < 10%；"
          "若系统性偏低，先查 fragmentation 与 temp workspace。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
