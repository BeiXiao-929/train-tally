# ledger —— 理论账本（显存 / 通信 / MFU 上界）

## 定位

副项目的第一个壁垒：**事前**算出每张卡该占多少显存、每步搬多少字节、
MFU 的上界是多少。跑通 FSDP 网上教程遍地都是；能事先算准才是能力。

三个脚本共用 `model_configs.py` 的模型定义，保证「显存、通信、MFU」
三张表用的是同一套参数量——否则三张表加起来对不上，会被一眼看穿。

| 文件 | 输出 | 验收 |
|---|---|---|
| `model_configs.py` | 解析参数量 + ParallelConfig | 与公开参数量偏差 < 5% |
| `memory_ledger.py` | 每卡显存分解（参数/梯度/优化器/激活） | 与实测误差 < 10% |
| `comm_ledger.py` | 每步字节数、通信时间、通信占比 | 与 trace 实测同量级（<20%） |
| `mfu_ledger.py` | MFU 上界 + 实测 gap 分解 | 实测 ≤ 上界 |
| `test_ledger.py` | 公式自检（纯 CPU） | 全绿 |

## 一键命令

```bash
python -m ledger.memory_ledger --model qwen2.5-7b --dp 4 --zero 0,1,2,3 --ac
python -m ledger.comm_ledger   --model qwen2.5-7b --dp 4 --link pcie4 --tp 2
python -m ledger.mfu_ledger    --model qwen2.5-7b --dp 4 --link nvlink
pytest ledger -q
```

## 显存公式（每参数字节数）

| 项 | DDP | ZeRO-1 | ZeRO-2 | ZeRO-3/FSDP |
|---|---|---|---|---|
| bf16 参数 | 2 | 2 | 2 | 2/N |
| bf16 梯度 | 2 | 2 | 2/N | 2/N |
| fp32 主权重 + Adam m/v | 12 | 12/N | 12/N | 12/N |
| 合计（N=4，含激活前） | 16 | 5 | 4 | 4 |

激活：不做重计算时 `L/pp × 每层元素数 × b × bytes`；
做重计算时只存 block 输入 `L/pp × b×s×h + 单层峰值`。
**这就是 AC 能省 50%+ 激活的算术来源**，比「除以 sqrt(L)」那种说法靠谱。

## 通信公式（每步每卡）

| 维度 | 字节数 | 特征 |
|---|---|---|
| DP AllReduce | `2P(N-1)/N` | 单次大消息，带宽敏感 |
| ZeRO-1/2 | `2P(N-1)/N`（RS+AG 两次） | 字节同 DDP，但两次集合通信、延迟翻倍 |
| ZeRO-3/FSDP | `3P(N-1)/N` | 前向+反向各一次 AllGather，再 ReduceScatter |
| TP | `4L × 2·b·s·h × bytes × (t-1)/t` | **次数多、单次小，延迟敏感** |
| PP | `2 × b·s·h × bytes × (p-1)` | 点对点，bubble 另算 |

TP 那一行的「4L」是关键：7B/28 层 → 112 次/步，单次只有 `b·s·h·2` 字节。
在 PCIe 上带宽根本没跑满，延迟就把时间吃光了——这就是选型边界的来源。

## 结果表（占位）

| 配置 | 参数 GB | 梯度 GB | 优化器 GB | 激活 GB | 合计 GB | 实测 GB | 误差 |
|---|---|---|---|---|---|---|---|
| DDP 4 卡 | 14.185 | 14.185 | 85.110 | 7.328 | **126.848** | `N/A` | `N/A` |
| ZeRO-1 | 14.185 | 14.185 | 21.277 | 7.328 | **59.824** | `N/A` | `N/A` |
| ZeRO-2 | 14.185 | 3.546 | 21.277 | 7.328 | **48.654** | `N/A` | `N/A` |
| ZeRO-3 | 3.546 | 3.546 | 21.277 | 7.328 | **37.483** | `N/A` | `N/A` |

> 「实测 / 误差」两列是 `N/A`：本机单卡，`max_memory_allocated` 只能在
> 4 卡上取。合计已含 1.05 的碎片系数（46.34 × 1.05 = 48.65，可手算复核）。

| 配置 | comm MB | 消息数 | comm ms | compute ms | 占比 | 实测占比 |
|---|---|---|---|---|---|---|
| DDP dp4 NVLink | 22.8 GB | 1 | 76 ms | 20.25% | 300 GB/s |
| ZeRO-3 dp4 NVLink | 34.3 GB | 2 | 114 ms | 27.58% | 300 GB/s |
| TP2×DP2 PCIe | ≈0.05 GB/步 | 113 | ≈2 ms | 负收益 −18.9% | 5–8 GB/s |

## 误差怎么收敛到 10% 以内

1. 先跑实测，拿到 `max_memory_allocated`
2. `python -m ledger.memory_ledger --measured-csv results/zero_*.csv` 看每档误差
3. 误差系统性偏低 → 调 `--fragmentation`（默认 1.05），或补上 workspace/临时缓冲
4. 误差只在高并发 batch 下偏大 → 激活公式漏了 attention scores（换 `--attn-impl eager`）
5. **不要为了对上而改公式**：改的是「口径参数」，并在报告里写清凭什么这么改

## 已知边界

- 参数量用解析式（逐矩阵求和），与公开值可能差 1%–4%；MFU 统一用解析值，
  并在报告注明口径。
- ZeRO-2/3 用 FSDP2 的 `reshard_after_forward` 近似，不是 DeepSpeed 的原生实现，
  结论只能用于「趋势」与「量级」，不能写「DeepSpeed ZeRO-2 的实测值」。
- 通信时间按「完全不重叠」算，是上界；真实重叠率由 trace 解析给出。
- 未建模多机分层 AllReduce，跨机场景需另乘系数。
