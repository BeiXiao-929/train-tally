# dist-mfu-profiler

**面向 4×A100 的 DDP / FSDP2 / ZeRO-1·2·3 / TP×DP 实验设计，配一套「理论账本 +
trace 解析」，把「显存去哪了、通信占多少、MFU 为什么上不去」从感觉变成数据，
最后给出「TP 超过几卡不划算」的量化边界。**

> **当前实测范围**：账本三表与 trace 解析器**已在单卡 RTX 4060 Laptop 上跑通**
> （纯 CPU 计算，无需 GPU）；**四组多卡实验一次都没跑过**，全部标 `N/A`——
> 这是本仓库唯一需要硬件的部分，理由与替代方案见「边界与反例」。

一句话版本：**不是「我跑通了 FSDP」，而是「我能事前算准、事后定位、并给出选型边界」。**

| 能力 | 交付物 | 证据 |
|---|---|---|
| 事前算显存 | [ledger/memory_ledger.py](ledger/memory_ledger.py) | 账本已跑（126.8/59.8/48.7/37.5 GB）；**与实测误差 `N/A`（无 4 卡）** |
| 事前算通信 | [ledger/comm_ledger.py](ledger/comm_ledger.py) | 22.8 GB/步（PCIe 占比 75.3%）；与 trace 实测比对 `N/A` |
| 事前算 MFU 上界 | [ledger/mfu_ledger.py](ledger/mfu_ledger.py) | 上界 79.75% / 72.42%；**实测 MFU `N/A`（无 4 卡）** |
| 事后定位通信 | [trace_parser/](trace_parser/) | 解析器自检通过（重叠率 0.5/占比 100%/忙碌 90%）；真实 trace `N/A` |
| 选型边界 | [docs/selection_boundary.md](docs/selection_boundary.md) | 临界 B_eff ≈ 11 GB/s；PCIe(5–8) → TP=2 起负收益 |

---

## 环境与版本锁定

| 项 | 锁定值 | 说明 |
|---|---|---|
| 硬件 | 4×A100 40GB | 单机 NVLink 或 PCIe（链路类型是核心变量，必须记录） |
| PyTorch | `2.4.1` | FSDP2 `fully_shard` 自 2.4 可用；`MixedPrecisionPolicy` 细节在 2.5 更稳 |
| CUDA / NCCL | `12.1` / `2.20.x`（随 torch 内置） | 用 `torch.cuda.nccl.version()` 确认 |
| Python | `3.10+` | — |
| 其它 | numpy 1.26 / pyyaml / pandas / matplotlib | 见 `requirements.txt` |
| 预算 | 40 卡时 | 见 `experiments/matrix.yaml`，单次超预估两倍即停止 |

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -m common.env --topology     # 先确认是 NVLink 还是 PCIe
```

---

## 一键复现

```bash
make smoke          # 无卡自检：环境 + 账本公式 + trace 解析 + 单进程干跑
make ledger         # 三张理论表（显存/通信/MFU 上界）  -> results/
make dry-run        # 四组实验各跑 5 步（CPU/gloo，验证脚本无 bug）
make self-test      # trace 解析自检（合成数据，已知重叠率 50%）

# 4 卡（云上；先按 matrix.yaml 逐组过 C1–C7 判据）
make ddp            # A: DDP 基线
make fsdp2          # B: FSDP2 全分片
make zero           # C: ZeRO-1/2/3
make tpdp           # D: 2-way TP × 2-way DP

# trace
python -m trace_parser.parse_torch_profiler --trace traces/step.json
python -m trace_parser.parse_nccl --kern-sum profiling/raw/kern_sum.csv

# 一条命令跑完所有「不需要 4 卡」的部分（约 3 秒）
make run-all        # 等价于 bash tools/run_all.sh
```

`tools/run_all.sh` 按顺序跑：环境与拓扑 → 显存账本 → 通信/MFU 账本 → trace 解析自检，
每步的原始输出落到 `results/run_<时间戳>/`。加 `RUN_DRYRUN=1` 会追加四组实验的单进程
干跑，用来确认训练循环与落盘没坏——**上云之前先跑这个，别把 40 卡时烧在脚本 bug 上**。

---

## 核心结果（账本已跑；多卡实验全部 `N/A`，原因见下表最后一列）

### 1. 显存账本（Qwen2.5-7B / 4 卡 / bf16 / b=1 / s=2048 / 含 1.05 碎片系数）

| 配置 | 参数 GB | 梯度 GB | 优化器 GB | 激活 GB | 合计 | 实测 | 误差 |
|---|---|---|---|---|---|---|---|
| DDP | 14.185 | 14.185 | 85.110 | 7.328 | **126.848** | `N/A` | `N/A` |
| ZeRO-1 | 14.185 | 14.185 | 21.277 | 7.328 | **59.824** | `N/A` | `N/A` |
| ZeRO-2 | 14.185 | 3.546 | 21.277 | 7.328 | **48.654** | `N/A` | `N/A` |
| ZeRO-3 | 3.546 | 3.546 | 21.277 | 7.328 | **37.483** | `N/A` | `N/A` |

> 解析参数量 7.615 B，与公开值 7.62 B 偏差 **−0.06%**（说明配置没抄错，账本的分母可信）。
> 三个直接可用的结论：
> **① 7B 纯 DDP 在 40 GB 卡上根本放不下**（126.8 GB/卡），必须上分片；
> **② ZeRO-3 才进 37.5 GB**，小于 40 GB 但余量只有 2.5 GB，实测 batch 稍大就 OOM；
> **③ 优化器状态是最大头**（85.1 GB = 67%），这解释了为什么 ZeRO-1 一步就砍掉一半。
>
> 「实测」列待 4 卡跑完后回填（`experiments/run_zero.py` 会输出 `peak_mem_gb` 与 `error_pct`）。
>
> **读 `results/*.csv` 的元数据时注意两个字段**：`gpu=` 是**采集这张表的机器**（RTX 4060 Laptop，
> 账本是纯 CPU 计算），`gpu_target=` 是**账本假设的目标卡**（`a100`，只用来取峰值算力与链路参数）。
> 两者混看会把「在笔记本上算出来的 A100 账本」误读成「在 A100 上实测」。

### 2. 吞吐与 MFU

| 实验 | tok/s/卡 | tok/s 合计 | MFU 上界 | 实测 MFU | gap | 通信占比 | 重叠率 |
|---|---|---|---|---|---|---|---|
| A DDP（NVLink） | `N/A` | `N/A` | 79.75% | `N/A` | `N/A` | 20.25% | `N/A` |
| B FSDP2（NVLink） | `N/A` | `N/A` | 与 DDP 同量级 | `N/A` | `N/A` | 同 DDP | `N/A` |
| C ZeRO-3（NVLink） | `N/A` | `N/A` | 72.42% | `N/A` | `N/A` | 27.58% | `N/A` |
| D TP2×DP2 | `N/A` | `N/A` | 见 selection_boundary §3 | `N/A` | `N/A` | `N/A` | `N/A` |
| D TP4×DP1 | `N/A` | `N/A` | 见 selection_boundary §3 | `N/A` | `N/A` | `N/A` | `N/A` |

> 「MFU 上界」与「通信占比」由 `ledger/mfu_ledger.py` 算出（假定通信完全不重叠，是保守上界）；
> 其余列需 4 卡实测。**上界先给出，是为了让实测 MFU 有一个可比较的参照**——
> 只报 MFU=42% 没法判断好坏，报「上界 79.75%、实测 X%、其中 Y 个点被通信吃掉」才是结论。

### 3. 链路对 DDP 的直接影响（PCIe 4.0 / 25 GB/s / 4 卡）

| 配置 | 每步通信 MB | 通信 ms | 计算 ms | 通信占比 | 结论 |
|---|---|---|---|---|---|
| DDP | 22846.5 | 913.9 | 299.9 | **75.3%** | 通信是计算的 3 倍，PCIe 上不可用 |
| ZeRO-1 | 22846.5 | 913.9 | 299.9 | 75.3% | 字节数与 DDP 相同（RS+AG 两次集合通信） |
| ZeRO-2 | 22846.5 | 913.9 | 299.9 | 75.3% | 同上；差别在延迟而非带宽 |
| ZeRO-3 | 34269.7 | 1370.8 | 299.9 | 82.1% | 多一次 AllGather，字节数 ×1.5 |

> 这张表是「为什么 PCIe 上不要用 DDP 做 7B」的算术证据：22.8 GB / 25 GB/s = 914 ms，
> 而计算只要 300 ms。换成 NVLink（300 GB/s）同一配置的通信占比降到 20.25%。
> **注意这是「完全不重叠」的上界**，真实重叠率要用 trace 算（`trace_parser`），
> 但 PCIe 上重叠率通常也低，量级不变。

### 4. TP 临界值（本文档最有价值的一行）

| 链路 | B_eff | t=2 | t=4 | t=8 | 临界 t |
|---|---|---|---|---|---|
| NVLink（账本假设） | 300 GB/s | +47.8% | +71.9% | +83.9% | 16 卡 |
| PCIe 4.0（理论） | 25 GB/s | +27.7% | +41.8% | +48.8% | 16 卡 |
| **PCIe 4.0（实测区间）** | **5–8 GB/s** | −18.9% | −28.1% | −32.7% | **1 卡** |

> 单元格是净收益（省下的计算时间减新增通信时间，除以 C_base）。
> **TP=2 的盈亏平衡带宽 ≈ 11.0 GB/s**（二分求得），而 PCIe 4.0 上 14 MB 消息的实测
> 有效带宽通常只有 5–8 GB/s —— **因此 PCIe 上 TP 从 2 卡起就是负收益**，
> 不是"4 卡起"。这是本仓库对原设计文档的一处修正。
>
> `实测有效带宽` 这一格是 **`N/A`（无 4 卡）**。上表的 300 / 25 GB/s 是账本假设值与理论值，
> 5–8 GB/s 是设计文档给出的经验区间，**三者的来源不同，不能混着引用**。
> 云卡补齐后必须用实测 B_eff 重算本表并替换注释——这条纪律不能破。

---

## 证据：能给的，与要等 4 卡才能给的

| 图 | 说明 |
|---|---|
| `docs/img/timeline_ddp.png` | `N/A`：DDP 的 NCCL 与计算重叠情况（需 4 卡 trace） |
| `docs/img/timeline_tp2.png` | `N/A`：TP 的 112 次小消息（需 4 卡 trace），可见关键路径上的串行 |
| `docs/img/mem_vs_zero.png` | `N/A`：显存随 ZeRO 档位下降（需 4 卡 trace）、吞吐同步下降 |
| `docs/img/mfu_gap.png` | `N/A`：上界 vs 实测的 gap 分解（需 4 卡实测 MFU） |

截图纪律：timeline 必须同窗口、同缩放、同 step 区间，否则不可比。

---

## 边界与反例

**不做什么**
- 不做多机（只单机 4 卡），因此不验证分层 AllReduce。
- 不训练真实数据集（合成 token），只验数值对齐与性能。
- 不复现 DeepSpeed ZeRO：见下方「实现声明」。

**实现声明（必须写进报告，否则会被认为夸大）**
- ZeRO-1 = `ZeroRedundancyOptimizer`（真 ZeRO-1）
- ZeRO-2/3 = FSDP2 的 `reshard_after_forward` 近似
- 结论只用于**趋势与量级**

**已知失效边界**
- 合成数据无收敛语义，不能用来讨论训练效果
- embedding / lm_head 未 TP 切分，TP 的显存收益略有高估
- PP 只在账本里建模，实验未跑
- 账本误差主要来自 allocator 碎片与临时 workspace（用 `--fragmentation` 校准）

**反例（保留）**
1. **只看通信占比就断言瓶颈** → 重叠率 90% 时结论完全反了。
   用 `make self-test` 可以复现「占比 20%、重叠率 50%」这种容易被误读的组合。
2. **用理论带宽算 TP 临界值** → PCIe 实测往往只有理论的 1/3–1/5，结论会偏乐观一档。
3. **一次性跑完四组再分析** → 出错时无法定位是脚本、显存还是通信，必须逐组过 C1–C7。
4. **为对上实测而改公式** → 改口径参数可以，改公式就是造假。

---

## 目录与依赖方向

```
dist-mfu-profiler/
├── common/          # 环境指纹（卡数/NCCL/拓扑）、结果落盘
├── ledger/          # 理论账本：显存 / 通信 / MFU 上界（纯 CPU，可离线跑）
├── trace_parser/    # trace 解析：分类、占比、重叠率（含自检）
├── experiments/     # 四组并行实验 + matrix.yaml + 共用训练骨架
├── results/         # 只提交 CSV 与 Markdown
└── docs/            # 选型边界、技术报告
```

依赖单向：`experiments → ledger → common`；`trace_parser → common`。
账本不依赖任何实验脚本，所以能在拿到卡之前就产出结论——
这正是「事前算准」的意义。

## 文档索引

| 文档 | 讲什么 | 什么时候读 |
|---|---|---|
| [`docs/technical_report.md`](docs/technical_report.md) | 技术报告：三本账的公式与假设、通信占比、MFU gap 分解、边界 | 只看一份的话看这个 |
| [`docs/selection_boundary.md`](docs/selection_boundary.md) | TP 盈亏平衡带宽的推导与结论（B_eff ≈ 11 GB/s 怎么来的） | 关心「TP 超过几卡不划算」 |
| [`ledger/README.md`](ledger/README.md) | 三本账的口径：2(N−1)/N 通信量、ZeRO 四分量、MFU 上界的推导 | 想核对公式 |
| [`experiments/README.md`](experiments/README.md) | 四组实验的 C1–C7 判据、matrix.yaml 参数、卡时预算 | 上云前必读 |
| [`trace_parser/README.md`](trace_parser/README.md) | 通信分类规则、占比与重叠率的算法、自检数据怎么造 | 拿到 trace 之后 |

> 约定：账本公式改口径可以（`--fragmentation` 等），**改公式对上实测就是造假**；
> 没有 4 卡的地方一律 `N/A` 并写明原因，不用估算值填。

---

## 许可与数据

代码 MIT。仓库内不含模型权重、数据集、checkpoint、profiler 原始报告。
所有 `__FILL__` 都表示「未实测」，不允许用估算值替换。
