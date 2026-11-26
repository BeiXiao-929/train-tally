"""common —— 环境指纹与结果落盘（纯 CPU 可用）。

多卡项目的环境指纹比单卡多三件事，缺了就没法解释 MFU 与通信差异：
    1. 卡数、逐卡型号（混卡会让 MFU 分母失真）
    2. NCCL 版本（不同版本的算法选择直接改变带宽利用率）
    3. 拓扑（NVLink / PCIe / SYS）——这是 docs/selection_boundary.md 的核心变量

注意：这里**不做任何 re-export**，避免 `python -m common.env` 触发
"found in sys.modules after import of package" 的 RuntimeWarning。
各处一律显式 `from common.env import ...` / `from common import io`。
"""
