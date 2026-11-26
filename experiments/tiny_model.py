"""experiments/tiny_model.py —— 四组实验共用的最小训练骨架。

文件职责
--------
只做四件事，确保四组实验的差异**只来自并行策略**：
1. `setup_dist()`：统一初始化（nccl/gloo、rank、device、mesh）
2. `TinyGPT`：可被 TP 切分的极简 decoder（RMSNorm + GQA attention + SwiGLU MLP）
3. `make_batch()`：合成 token（不用数据集，仓库里不存任何数据文件）
4. `train_steps()`：统一步循环与指标口径（step_ms、tokens/s、峰值显存、grad norm）

为什么用合成数据
----------------
数据管线会引入与并行策略无关的抖动（dataloader worker、IO）。本项目研究的是
「显存—通信—MFU」，合成 token 能把变量减到只剩并行策略本身，也避免了
把数据集放进仓库。

为什么 TP 要自己写两个 Linear
-----------------------------
`ColumnParallelLinear` / `RowParallelLinear` 是 TP 的最小完整单元：
前者按输出维切、后者按输入维切 + all_reduce。两个加起来不到 40 行，
比引入 megatron 依赖更适合「读懂机制」这个目标。

TODO
----
[ ] 支持 PP（需要切层 + send/recv，当前 pp 只在账本里建模，实验未跑）
[ ] attention 换成 flash-attn（当前用 SDPA，兼容性优先）
[ ] 增加 `--profile` 导出 chrome trace（trace_parser 需要）
[ ] 支持 bf16 autocast（当前 fp32，理由是 A100 上 bf16 的数值对齐要单独验证）

验收标准
--------
- `--dry-run` 在 CPU 单进程上跑通 3 步（gloo，world_size=1）
- 同一 seed 两次运行的 loss 序列一致（单进程）
- tp=1 时 Column/RowParallelLinear 与 nn.Linear 输出逐位一致
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ledger.model_configs import ModelConfig


# --------------------------------------------------------------------------- #
# 分布式初始化
# --------------------------------------------------------------------------- #
@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_dist(dry_run: bool = False) -> DistInfo:
    """初始化（或假装初始化）进程组。

    dry_run 时用 world_size=1 + gloo + cpu，保证没有 GPU、没有 torchrun 也能跑通，
    这是「上云前先在本地验证脚本」这条纪律的实现方式。
    """
    if dry_run and "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29531")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available() and not dry_run:
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistInfo(rank, local_rank, world_size, device, backend)


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# --------------------------------------------------------------------------- #
# 并行线性层（TP 的最小完整单元）
# --------------------------------------------------------------------------- #
class ColumnParallelLinear(nn.Module):
    """按输出维切分：每个 rank 只持有 out/tp 行。输入完整复制。"""

    def __init__(self, in_features: int, out_features: int, tp: int = 1,
                 process_group=None):
        super().__init__()
        assert out_features % tp == 0, f"out={out_features} 不能被 tp={tp} 整除"
        self.out_per_rank = out_features // tp
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    """按输入维切分：每个 rank 只持有 in/tp 列，输出需要 all_reduce 求和。"""

    def __init__(self, in_features: int, out_features: int, tp: int = 1,
                 process_group=None):
        super().__init__()
        assert in_features % tp == 0, f"in={in_features} 不能被 tp={tp} 整除"
        self.in_per_rank = in_features // tp
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.normal_(self.weight, std=0.02)
        self.process_group = process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)
        if dist.is_initialized() and dist.get_world_size(self.process_group) > 1:
            dist.all_reduce(out, group=self.process_group)
        return out


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: int, tp_group=None):
        super().__init__()
        assert cfg.n_heads % tp == 0 and cfg.n_kv_heads % tp == 0, \
            f"heads={cfg.n_heads}/kv={cfg.n_kv_heads} 不能被 tp={tp} 整除"
        self.n_heads = cfg.n_heads // tp
        self.n_kv_heads = cfg.n_kv_heads // tp
        self.head_dim = cfg.head_dim
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.qkv = ColumnParallelLinear(cfg.hidden, self.q_dim + 2 * self.kv_dim,
                                        tp, tp_group)
        self.o = RowParallelLinear(self.q_dim, cfg.hidden, tp, tp_group)
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # GQA：kv 头数少于 q 头数时复制，避免引入额外依赖
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.q_dim)
        return self.o(out)


class MLP(nn.Module):
    """SwiGLU：w13 输出 (gate, up)，w2 投影回 hidden。"""

    def __init__(self, cfg: ModelConfig, tp: int, tp_group=None):
        super().__init__()
        assert cfg.ffn % tp == 0, f"ffn={cfg.ffn} 不能被 tp={tp} 整除"
        self.ffn_per_rank = cfg.ffn // tp
        self.w13 = ColumnParallelLinear(cfg.hidden, 2 * cfg.ffn, tp, tp_group)
        self.w2 = RowParallelLinear(cfg.ffn, cfg.hidden, tp, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).split(self.ffn_per_rank, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: int, tp_group=None):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden)
        self.attn = Attention(cfg, tp, tp_group)
        self.ln2 = RMSNorm(cfg.hidden)
        self.mlp = MLP(cfg, tp, tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """极简 decoder：只保留与本项目相关的结构（GQA + SwiGLU + RMSNorm）。"""

    def __init__(self, cfg: ModelConfig, tp: int = 1, tp_group=None,
                 vocab: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab or cfg.vocab
        self.embed = nn.Embedding(self.vocab, cfg.hidden)
        self.blocks = nn.ModuleList([Block(cfg, tp, tp_group)
                                     for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, self.vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


def build_model(cfg: ModelConfig, tp: int = 1, tp_group=None,
                device: torch.device = torch.device("cpu")) -> TinyGPT:
    model = TinyGPT(cfg, tp=tp, tp_group=tp_group)
    return model.to(device)


# --------------------------------------------------------------------------- #
# 数据与训练步
# --------------------------------------------------------------------------- #
def make_batch(vocab: int, batch: int, seq: int, device: torch.device,
               seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, vocab, (batch, seq + 1), generator=g).to(device)
    return idx[:, :-1], idx[:, 1:]


def grad_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total = total + p.grad.detach().float().pow(2).sum()
    total = all_reduce_sum(total)
    return float(total.sqrt().item())


def train_steps(model: nn.Module, optimizer: torch.optim.Optimizer, *,
                steps: int, batch: int, seq: int, vocab: int,
                device: torch.device, seed: int = 1234,
                warmup: int = 2, log: bool = True) -> List[Dict[str, Any]]:
    """跑 steps 步，返回每步指标。前 warmup 步不计入吞吐（冷启动）。"""
    model.train()
    rows: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        x, y = make_batch(vocab, batch, seq, device, seed + step)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        loss.backward()
        gn = grad_norm(model)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mem = (torch.cuda.max_memory_allocated() / 2 ** 30
               if device.type == "cuda" else 0.0)

        rows.append({
            "step": step,
            "loss": round(float(loss.item()), 6),
            "grad_norm": round(gn, 6),
            "step_ms": round(dt * 1e3, 3),
            "tokens_per_s": round(batch * seq / dt, 1),
            "peak_mem_gb": round(mem, 3),
            "is_warmup": step <= warmup,
        })
        if log and step % max(1, steps // 5) == 0:
            print(f"  step {step:>4}/{steps} loss={rows[-1]['loss']:.4f} "
                  f"{rows[-1]['step_ms']:.1f} ms "
                  f"{rows[-1]['tokens_per_s']:.0f} tok/s "
                  f"mem={rows[-1]['peak_mem_gb']:.2f} GB", flush=True)
    return rows


def summarize(rows: List[Dict[str, Any]], world_size: int) -> Dict[str, Any]:
    """取稳态段（去掉 warmup）的中位数，口径与账本对齐。"""
    steady = [r for r in rows if not r["is_warmup"]] or rows
    ms = sorted(r["step_ms"] for r in steady)
    tps = sorted(r["tokens_per_s"] for r in steady)
    mid = len(ms) // 2
    return {
        "step_ms_median": round(ms[mid], 3),
        "tokens_per_s_per_rank": round(tps[mid], 1),
        "tokens_per_s_global": round(tps[mid] * world_size, 1),
        "peak_mem_gb": round(max(r["peak_mem_gb"] for r in rows), 3),
        "loss_first": steady[0]["loss"],
        "loss_last": steady[-1]["loss"],
        "grad_norm_last": steady[-1]["grad_norm"],
        "steps": len(rows),
    }
