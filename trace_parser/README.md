# trace_parser —— 从 trace 里读出「通信吃掉了多少、被掩盖了多少」

## 定位

副项目的第二个壁垒。跑通 FSDP 之后，真正的问题是：**MFU 为什么上不去？**
答案通常藏在两个数字里：

```
通信占比（NCCL 时长 / 墙钟）      → 通信有没有占时间
重叠率（NCCL ∩ Compute / NCCL）   → 占的时间有没有被计算掩盖
```

通信占比 30% 但重叠率 90% 的训练，实际损失可能只有 3%。
**只看占比会得出「通信是瓶颈」的错误结论**——这是本项目最想纠正的误判。

## 文件职责

| 文件 | 输入 | 输出 |
|---|---|---|
| `classify_rules.yaml` | — | kernel 名 → 类别的正则规则（外置，可校准） |
| `parse_torch_profiler.py` | chrome trace JSON | 分类占比表 + 重叠率 + Markdown |
| `parse_nccl.py` | nsys stats CSV（kern_sum / gpu_trace） | NCCL 占比、实例数、平均单次耗时、重叠率 |
| `report.py` | 上述事件的统一汇总 | CSV + Markdown；`--self-test` 不依赖真实 trace |

## 一键命令

```bash
# 0) 先确认解析链路没坏（合成数据，已知重叠率 50%）
make self-test
#   等价于: python -m trace_parser.report --self-test

# 1) torch.profiler 路径
#    在训练脚本里：p.export_chrome_trace("traces/step.json")
python -m trace_parser.parse_torch_profiler --trace traces/step.json

# 2) nsys 路径
nsys stats --report cuda_gpu_kern_sum --format csv r.nsys-rep > profiling/raw/kern_sum.csv
nsys stats --report cuda_gpu_trace  --format csv r.nsys-rep > profiling/raw/gpu_trace.csv
python -m trace_parser.parse_nccl --kern-sum profiling/raw/kern_sum.csv \
    --trace profiling/raw/gpu_trace.csv
```

## 结果表（占位）

| 实验 | 墙钟 ms | NCCL 占比 | 重叠率 | Compute 占比 | GPU 忙碌率 | 实测通信量 MB | 理论通信量 MB |
|---|---|---|---|---|---|---|---|
| A: DDP dp4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| B: FSDP2 dp4 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| C: ZeRO-2 / ZeRO-3 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |
| D: TP2×DP2 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` |

> 全部 `N/A`：需要 4 卡的真实 trace。
> **但解析器本身已验证**：`python -m trace_parser.report --self-test` 用合成 trace
> （计算 [0,100] ms、通信 [50,150] ms）得到已知答案 —— 重叠率 0.5、占比合计 100%、
> GPU 忙碌率 90%、NCCL 占比 20%，四项全部通过。**工具可信，缺的只是真数据。**

> 「理论通信量」来自 `ledger.comm_ledger`；「实测通信量」nsys 不给，只能从
> NCCL 报告的 bytes 列取（版本相关）。取不到就写 `__FILL__`——**不要用时长反推字节数**。

## 三个必须写进报告的判读

1. **占比高 + 重叠率高 = 不是瓶颈**。所以报告里两个数字必须并排出现，
   只报一个就是选择性呈现。
2. **ZeRO-3 的通信量是 DDP 的 1.5 倍，但重叠率往往更高**（前向 AllGather 可与
   上一层计算重叠）。所以「ZeRO-3 通信更贵」不等于「ZeRO-3 更慢」。
3. **TP 的重叠率通常很低**。因为 TP 的 AllReduce 在每一层的关键路径上，
   没有别的计算可以掩盖它。这是 PCIe 上 TP 不划算的机制性原因，
   与 comm_ledger 的「消息数 4L」相互印证。

## 规则校准（换框架必做）

```bash
python -m trace_parser.parse_torch_profiler --trace traces/step.json
# 看输出的 other 类占比
```

- `other` > 2% → 有 kernel 没被规则覆盖，把名字前缀加进 `classify_rules.yaml`
- `compute` 占比异常低 → 检查是否把 CPU op 也算进来了（默认已过滤）
- 重叠率恒为 0 → 检查 NCCL 与 Compute 是否落在同一个 pid/stream 维度上

## 已知边界

- 重叠率用「全局区间并集」计算，不区分是哪个 step、哪条 stream。
  极短的交叉会被算进重叠，可能高估重叠率（TODO 里已标注改进方向）。
- 按整段 trace 统计，不做 step 切窗；长时间训练需先截取稳态段。
- nsys 与 torch.profiler 的时钟基准不同，跨工具对比要用相对占比，不要比绝对毫秒。
