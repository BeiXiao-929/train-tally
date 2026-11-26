"""common/env.py —— 环境指纹（多卡版）。

文件职责
--------
采集并返回一次训练的完整环境描述，写进任何结果 CSV 的头部：
    gpu_count / gpu / driver / cuda / torch / nccl / interconnect /
    clock_sm_mhz / seed

三点与单卡项目的差异：
1. **gpu_count 必须记录**：MFU 的分母是「峰值算力 × 卡数」，卡数错了结论就错了。
2. **nccl 版本必须记录**：NCCL 的算法/协议选择（LL/LL128/Simple）会改变
   实际可达带宽，跨版本比较通信量没有意义。
3. **interconnect 必须记录**：NVLink 与 PCIe 的临界 TP 值差一个数量级
   （见 docs/selection_boundary.md），这是本项目最有价值的结论的输入。

TODO
----
[ ] 采集逐卡时钟与 PCIe 链路代数/宽度（`nvidia-smi --query-gpu=pcie.link.gen.*`）
[ ] 采集 NIC 信息（IB/RoCE 速率），多机场景需要
[ ] 记录 CUDA_VISIBLE_DEVICES 原始值（复现时最常见的坑）

验收标准
--------
- 无 GPU 时 exit 0，gpu_count=0，不抛异常
- 有 GPU 时 gpu_count 与 torch.cuda.device_count() 一致
- `python -m common.env --topology` 能打印 nvidia-smi topo -m 的摘要
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_SEED = 1234

HEADER_KEYS = [
    "gpu_count", "gpu", "driver", "cuda", "torch", "nccl",
    "interconnect", "clock_sm_mhz", "seed",
]


def _run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except Exception:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"gpu_count": 0, "gpu": None, "driver": None,
                            "clock_sm_mhz": None}
    try:
        import torch

        info["gpu_count"] = torch.cuda.device_count()
        if info["gpu_count"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    raw = _run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm",
                "--format=csv,noheader,nounits"])
    if raw:
        lines = raw.splitlines()
        first = [p.strip() for p in lines[0].split(",")]
        info["gpu_count"] = len(lines)
        if len(first) >= 3:
            info["gpu"] = first[0]
            info["driver"] = first[1]
            try:
                info["clock_sm_mhz"] = int(first[2])
            except ValueError:
                pass
    return info


def nccl_version() -> Optional[str]:
    try:
        import torch

        v = torch.cuda.nccl.version()
        return ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
    except Exception:
        return None


def topology(raw: bool = False) -> Dict[str, Any]:
    """返回 {interconnect, detail}。

    interconnect 取值：nvlink / pcie / mixed / unknown / cpu
    判定依据：`nvidia-smi topo -m` 的 GPU-GPU 列里出现 NV 即视为有 NVLink。
    """
    out = _run(["nvidia-smi", "topo", "-m"])
    if not out:
        return {"interconnect": "unknown", "detail": ""}
    lines = [ln for ln in out.splitlines() if ln.strip()]
    gpu_lines = [ln for ln in lines if ln.strip().upper().startswith("GPU")]
    has_nv = any("NV" in ln for ln in gpu_lines)
    has_pci = any("PIX" in ln or "PHB" in ln or "PXB" in ln for ln in gpu_lines)
    has_sys = any("SYS" in ln for ln in gpu_lines)
    if has_nv and not has_sys:
        kind = "nvlink"
    elif has_nv and has_sys:
        kind = "mixed"
    elif has_pci:
        kind = "pcie"
    elif has_sys:
        kind = "cpu"      # 跨 NUMA，走主机内存，最差
    else:
        kind = "unknown"
    detail = "\n".join(lines[:12]) if raw else " | ".join(gpu_lines[:4])
    return {"interconnect": kind, "detail": detail}


def lock_clocks(sm_mhz: int = 1410, mem_mhz: int = 1215) -> bool:
    """锁定 GPU 时钟。

    MFU 的分母是「峰值算力 × 卡数」，但分子（实测吞吐）随频率漂移可达 5%+。
    多卡训练时不同卡的降频程度还不一样，会让「通信占比」出现无法解释的抖动。
    所以做 MFU 对比实验前必须锁频，并把实际频率写进结果头部。

    需要 root；失败只告警（WSL2 下 nvidia-smi 多数子命令不可用）。
    """
    ok_sm = _run(["nvidia-smi", f"--lock-gpu-clocks={sm_mhz},{sm_mhz}"], timeout=30)
    ok_mem = _run(["nvidia-smi", f"--lock-memory-clocks={mem_mhz},{mem_mhz}"], timeout=30)
    ok = ok_sm is not None and ok_mem is not None
    if not ok:
        print(f"[WARN] 时钟锁定失败（需要 root）。建议手工执行："
              f"nvidia-smi -lgc {sm_mhz} -lmc {mem_mhz}；"
              f"未锁频时吞吐差异可达 5%，MFU 对比结论要打折扣。", file=sys.stderr)
    return ok


def reset_clocks() -> None:
    _run(["nvidia-smi", "--reset-gpu-clocks"], timeout=30)
    _run(["nvidia-smi", "--reset-memory-clocks"], timeout=30)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False  # 与 MFU 的 FP16 峰值对齐
    except Exception:
        pass


def collect(extra: Optional[Dict[str, Any]] = None,
            seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info.update(gpu_info())
    info["cuda"] = None
    info["torch"] = _pkg_version("torch")
    info["nccl"] = nccl_version()
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if info["torch"] is None:
            info["torch"] = torch.__version__
    except Exception:
        pass
    info["interconnect"] = topology()["interconnect"]
    info["seed"] = seed

    ordered = {k: info.get(k) for k in HEADER_KEYS}
    ordered["world_size_env"] = os.environ.get("WORLD_SIZE", "1")
    ordered["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    # 主机名会随 results/*.csv 一起入库并公开，默认脱敏为 "local"。
    # 需要区分多台机器时设 BENCH_HOST_LABEL=<标签>（如 a100-node0），
    # 不要把真实主机名写进 CSV —— 配合 GPU 型号与版本足以指纹化一台机器。
    ordered["host"] = os.environ.get("BENCH_HOST_LABEL", "local")
    ordered["collected_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if extra:
        ordered.update(extra)
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="环境指纹（多卡）")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--topology", action="store_true", help="打印 nvidia-smi topo -m")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--lock-clocks", nargs=2, type=int, metavar=("SM", "MEM"),
                    help="尝试锁定时钟，例如 --lock-clocks 1410 1215")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    if args.lock_clocks:
        lock_clocks(*args.lock_clocks)
    if args.topology:
        topo = topology(raw=True)
        print(f"interconnect: {topo['interconnect']}")
        print(topo["detail"])
        if topo["interconnect"] == "pcie":
            print("\n[注意] 检测到 PCIe：TP 的临界值会比 NVLink 小一个数量级，"
                  "实验 D 的结论必须注明链路类型。")
        return 0

    info = collect(seed=args.seed)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("# env " + " | ".join(f"{k}={info.get(k)}" for k in HEADER_KEYS))
    if info["gpu_count"] == 0:
        print("[WARN] 未检测到 GPU：训练实验将 skip，账本仍可纯 CPU 运行",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
