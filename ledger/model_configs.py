"""ledger/model_configs.py —— 模型与并行配置的唯一定义来源。

文件职责
--------
1. 用结构超参解析式地算出参数量（不硬编码「7B」这种名字，避免与账本口径脱节）
2. 提供三档模型：真实（Qwen2.5-7B / Llama-3-8B）+ 调试用（tiny），
   tiny 用于在单卡/CPU 上验证脚本逻辑，避免把云卡时烧在调试上
3. 提供 ParallelConfig，集中表达 DP/TP/PP/ZeRO 与 batch/seq/dtype

为什么参数量要用算的而不是抄
----------------------------
MFU 的分母是「6 × 参数量 × tokens/s」，参数量抄错 5%，MFU 就错 5%，
而 MFU 的结论精度要求就是几个百分点。所以这里逐矩阵求和：
    embed + L×(q + k + v + o + gate + up + down + 2×norm) + final_norm + lm_head
并与公开值做偏差检查（<5% 视为口径差异，>5% 说明配置抄错了）。

注意：与公开参数量可能有 1%–4% 的差异（是否 tie embedding、是否计 bias、
norm 的实现差异）。**MFU 计算统一用本文件的解析值**，并在报告里注明这一口径。

TODO
----
[ ] 增加 MoE 配置（专家并行会让通信账本多一条 all2all）
[ ] 增加 MLA（DeepSeek）配置，KV 缓存公式不同
[ ] 增加 `--from-hf-config` 直接读 config.json，避免手抄出错

验收标准
--------
- qwen2.5-7b 解析参数量与公开值偏差 < 5%
- tiny 配置的 total_params < 0.1B，可在 CPU 上跑完一次前向
- 所有 dataclass 字段无默认歧义（dtype 用字符串，避免 torch 依赖）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden: int
    ffn: int
    vocab: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    tie_embeddings: bool = False
    published_params_b: Optional[float] = None  # 公开参数量，仅用于口径对照

    # ---------------- 参数分解（单位：个）---------------- #
    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def params_breakdown(self) -> Dict[str, float]:
        """返回各部分的参数量（单位：B = 1e9）。"""
        embed = self.vocab * self.hidden
        q = self.hidden * self.q_dim
        k = self.hidden * self.kv_dim
        v = self.hidden * self.kv_dim
        o = self.q_dim * self.hidden
        gate = self.hidden * self.ffn
        up = self.hidden * self.ffn
        down = self.ffn * self.hidden
        norms = 2 * self.hidden
        per_layer = q + k + v + o + gate + up + down + norms
        lm_head = 0 if self.tie_embeddings else self.vocab * self.hidden
        total = embed + self.n_layers * per_layer + self.hidden + lm_head
        return {
            "embedding": embed / 1e9,
            "attention_per_layer": (q + k + v + o) / 1e9,
            "mlp_per_layer": (gate + up + down) / 1e9,
            "norms_per_layer": norms / 1e9,
            "lm_head": lm_head / 1e9,
            "total": total / 1e9,
        }

    def total_params(self) -> float:
        return self.params_breakdown()["total"]

    def deviation_pct(self) -> Optional[float]:
        if not self.published_params_b:
            return None
        return (self.total_params() - self.published_params_b) / self.published_params_b * 100

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """每 token 每层的 KV 字节数 × 层数（不含 batch 与 seq）。"""
        return 2 * self.n_layers * self.kv_dim * dtype_bytes


@dataclass
class ParallelConfig:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    zero_stage: int = 0            # 0=DDP, 1/2/3=ZeRO
    activation_checkpointing: bool = False
    dtype: str = "bf16"            # bf16 / fp16 / fp32
    batch: int = 1                  # micro batch per rank
    seq: int = 2048

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp

    @property
    def dtype_bytes(self) -> int:
        return {"bf16": 2, "fp16": 2, "fp32": 4}[self.dtype]

    def describe(self) -> str:
        z = f"ZeRO-{self.zero_stage}" if self.zero_stage else "DDP"
        ac = "+AC" if self.activation_checkpointing else ""
        return (f"{z} dp{self.dp}×tp{self.tp}×pp{self.pp} "
                f"b{self.batch}s{self.seq}{ac}")


# --------------------------------------------------------------------------- #
# 模型注册表
# --------------------------------------------------------------------------- #
_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B-Instruct", n_layers=28, hidden=3584, ffn=18944,
        vocab=152064, n_heads=28, n_kv_heads=4, head_dim=128,
        # Qwen2.5-7B 不共享词嵌入：解析值 7.62B 与公开值一致（差异来自 lm_head）
        tie_embeddings=False, published_params_b=7.62,
    ),
    "llama3-8b": ModelConfig(
        name="Llama-3-8B", n_layers=32, hidden=4096, ffn=14336,
        vocab=128256, n_heads=32, n_kv_heads=8, head_dim=128,
        tie_embeddings=False, published_params_b=8.03,
    ),
    "tiny-1b": ModelConfig(
        name="Tiny-1B（单卡可跑，用于验证脚本）", n_layers=16, hidden=2048,
        ffn=5632, vocab=32000, n_heads=16, n_kv_heads=4, head_dim=128,
        tie_embeddings=True, published_params_b=0.79,
    ),
    "tiny-debug": ModelConfig(
        name="Tiny-Debug（CPU 可跑，仅验证逻辑）", n_layers=4, hidden=256,
        ffn=512, vocab=1024, n_heads=4, n_kv_heads=2, head_dim=64,
        tie_embeddings=True,
    ),
}


def get(name: str) -> ModelConfig:
    if name not in _MODELS:
        raise KeyError(f"未知模型 {name}，可用：{', '.join(_MODELS)}")
    return _MODELS[name]


def names() -> List[str]:
    return list(_MODELS)


def all_models() -> List[ModelConfig]:
    return list(_MODELS.values())


__all__ = ["ModelConfig", "ParallelConfig", "get", "names", "all_models"]
