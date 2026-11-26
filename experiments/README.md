# experiments —— 四组并行实验与「跑通判据」

## 定位

这里的价值不是「跑通 FSDP」（教程满地都是），而是**在可控变量下跑出可比数据**：
同一模型、同一 batch、同一 seed，只换并行策略，看显存、通信、MFU 怎么变。

| 实验 | 脚本 | 并行配置 | 目的 |
|---|---|---|---|
| A | `run_ddp.py` | DDP | 通信基线：AllReduce 与反向的重叠程度 |
| B | `run_fsdp2.py` | FSDP2 全分片 | 参数分片带来的通信增量 |
| C | `run_zero.py` | ZeRO-1/2/3 | 显存—通信权衡曲线 |
| D | `run_tp_dp.py` | TP×DP | 高频小消息对延迟的敏感性（选型边界的原始数据） |

`tiny_model.py` 是共用骨架（含 TP 的 Column/RowParallelLinear），
`matrix.yaml` 是唯一事实来源（配置 + 七条判据 + 卡时预算）。

## 一键命令

```bash
# 0) 本地验证脚本（无卡可跑，3–5 步，gloo 单进程）
make dry-run

# 1) 单组实验（4 卡）
torchrun --nproc_per_node=4 -m experiments.run_ddp    --model tiny-1b --steps 200
torchrun --nproc_per_node=4 -m experiments.run_fsdp2  --model tiny-1b --steps 200
torchrun --nproc_per_node=4 -m experiments.run_zero   --model tiny-1b --steps 200 --stages 1,2,3
torchrun --nproc_per_node=4 -m experiments.run_tp_dp  --model tiny-1b --steps 200 --tp 2 --dp 2
torchrun --nproc_per_node=4 -m experiments.run_tp_dp  --model tiny-1b --steps 200 --tp 4 --dp 1
```

## 七条跑通判据（缺一条都不算跑通）

| # | 判据 | 怎么验 |
|---|---|---|
| C1 | loss 与单卡基线前 200 步相对偏差 < 1e-3 | 比 `loss_first`/`loss_last` |
| C2 | 第 1/100 步 grad_norm 偏差 < 1e-2 | `grad_norm_last` |
| C3 | `max_memory_allocated` 与账本误差 < 10% | `error_pct` 列 |
| C4 | 单卡吞吐随卡数次线性，拐点可被通信量解释 | `tokens_per_s_global / world_size` |
| C5 | NCCL 字节与理论同量级（< 20%） | `trace_parser.parse_nccl` vs `comm_ledger` |
| C6 | checkpoint 重载 loss 连续 | 手工（TODO：未自动化） |
| C7 | 无 NaN/OOM，trace 完整导出 | 日志 + `traces/` |

**C3 是副项目的核心验收线**：账本算不准，整个「能诊断」的定位就不成立。
**C4 是选型结论的前提**：如果吞吐随卡数线性增长，说明通信根本不是瓶颈，
那 docs/selection_boundary.md 的结论就无从谈起。

## 结果表（占位）

| 实验 | world | step ms | tok/s/卡 | tok/s 合计 | 峰值 GB | 账本 GB | 误差 % | loss 首→末 |
|---|---|---|---|---|---|---|---|---|
| A DDP | 4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| B FSDP2 | 4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| C ZeRO-1/2/3 | 4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| D TP2×DP2 / TP4×DP1 | 4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |

> 整表 `N/A`，原因只有一条：**本机是单卡 RTX 4060 Laptop 8GB，四组实验都需要 4×A100。**
> 本机能验证的是脚本本身——`python -m experiments.run_<x> --dry-run --steps 5`
> 能跑通训练循环与落盘（单进程/CPU 会跳过 FSDP2 包装，这是设计而非 bug）。
> 上表每一格都需要真实 4 卡才有数字，**不做估算填充**。

## 实现选择（写进报告，别让面试官以为是 DeepSpeed 实测）

| 档位 | 本仓库实现 | 与原生实现的关系 |
|---|---|---|
| ZeRO-1 | `ZeroRedundancyOptimizer` | 真 ZeRO-1 |
| ZeRO-2 | `fully_shard(reshard_after_forward=False)` | 语义等价，非 DeepSpeed |
| ZeRO-3 | `fully_shard(reshard_after_forward=True)` | 语义等价 FSDP |

不用 DeepSpeed 的理由：它的 bucket/offload/通信 hook 默认值会把
「框架行为」混进「并行策略」的对比。**结论只用于趋势与量级。**

## 上云纪律（40 卡时是唯一真金白银的成本）

1. 先在本地单卡用 `tiny-debug` 跑 `--dry-run`，确认脚本无 bug
2. 再在云上单卡跑 10 步，确认显存与账本误差在看板上能看见
3. 才上 4 卡跑满 200 步
4. 每次实验结束**立刻导出 trace**（实例一关就没了）
5. 单次超预估两倍即停止——这条比任何优化都省钱

## 已知边界

- 合成数据（random token）：loss 没有语义，只能用于**数值对齐**（C1/C2），
  不能用来讨论收敛性。
- `tiny-1b` 的 embedding 与 lm_head 未做 TP 切分（replicated），
  所以 TP 的显存收益会被高估一点；报告里要注明。
- PP 只在账本里建模，实验未跑（`matrix.yaml` 里没有 PP 项）。
- 不保存 checkpoint（C6 手工验证），仓库不允许出现 `.pt`。
