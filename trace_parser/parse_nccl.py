"""trace_parser/parse_nccl.py —— 从 nsys stats 的 CSV 里量化 NCCL 通信。

文件职责
--------
1. 读 `cuda_gpu_kern_sum`：NCCL kernel 的次数、总时长、占全部 kernel 的比例
2. 读 `cuda_gpu_trace`：拿 NCCL 与 Compute 的时间区间，算**重叠率**
3. 与理论通信量对账（理论值来自 ledger.comm_ledger，本脚本不重复推导）

为什么必须读 trace 而不是只读 kern_sum
--------------------------------------
kern_sum 只知道「NCCL 占了多少时间」，不知道这些时间是否与计算重叠。
同样 30% 的通信占比，重叠率 0 与 90% 是天壤之别——后者几乎不损失 MFU。
这是本模块存在的唯一理由。

nsys 不给通信字节数
-------------------
`cuda_gpu_kern_sum` 没有 bytes 列，所以「NCCL 字节数」这一项只能：
    (a) 用理论账本（comm_ledger）给出预期值
    (b) 用 nccl 的 `--report nccl_gpu_kern_sum`（部分版本有 bytes）补充
本脚本对字节数一律写 `__FILL__`，绝不从时长反推。

用法
----
    nsys stats --report cuda_gpu_kern_sum --format csv r.nsys-rep > profiling/raw/kern.csv
    nsys stats --report cuda_gpu_trace  --format csv r.nsys-rep > profiling/raw/trace.csv
    python -m trace_parser.parse_nccl --kern-sum profiling/raw/kern.csv \
        --trace profiling/raw/trace.csv

TODO
----
[ ] 支持 nccl_gpu_kern_sum 的 bytes 列（版本相关），有则填入并与理论对账
[ ] 按 NCCL 算法（RING/TREE、LL/LL128）分类统计
[ ] 增加逐 step 的通信时间序列（定位抖动来源）

验收标准
--------
- NCCL 占比与 nsys GUI 里 `nccl_gpu_kern_sum` 的 Time(%) 一致（误差 < 1%）
- 重叠率与手工在 timeline 上量的结果同量级
- 缺 trace 文件时只输出占比，overlap 写 __FILL__，不猜测
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import io
from trace_parser.report import (Event, Interval, merge, overlap_len, summary,
                                 total_len)

NCCL_RE = re.compile(r"nccl|all_?reduce|all_?gather|reduce_?scatter|alltoall",
                     re.IGNORECASE)


def read_nsys_csv(path: Path) -> List[Dict[str, str]]:
    """容错读 nsys stats CSV（前面有若干元信息行）。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = None
    for i, line in enumerate(lines[:40]):
        if "Name" in line and ("Time" in line or "Duration" in line or "Start" in line):
            header = i
            break
    if header is None:
        raise ValueError(f"{path} 找不到表头，确认是 --format csv 的产物")
    return [{ (k or "").strip(): (v or "").strip() for k, v in r.items() }
            for r in csv.DictReader(lines[header:]) if r.get("Name")]


def _num(row: Dict[str, str], *prefixes: str) -> Optional[float]:
    for p in prefixes:
        for k in row:
            if k.lower().startswith(p.lower()):
                try:
                    return float(row[k].replace(",", ""))
                except ValueError:
                    continue
    return None


def nccl_intervals(trace_rows: List[Dict[str, str]]) -> List[Interval]:
    out = []
    for r in trace_rows:
        name = r.get("Name", "")
        if not NCCL_RE.search(name):
            continue
        start = _num(r, "Start")
        dur = _num(r, "Duration")
        if start is None or dur is None or dur <= 0:
            continue
        out.append((start / 1e9, (start + dur) / 1e9))
    return out


def compute_intervals(trace_rows: List[Dict[str, str]]) -> List[Interval]:
    out = []
    for r in trace_rows:
        name = r.get("Name", "")
        if NCCL_RE.search(name) or re.search(r"memcpy|memset", name, re.I):
            continue
        start = _num(r, "Start")
        dur = _num(r, "Duration")
        if start is None or dur is None or dur <= 0:
            continue
        out.append((start / 1e9, (start + dur) / 1e9))
    return out


def kern_summary(kern_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = 0.0
    nccl_total = 0.0
    nccl_count = 0
    for r in kern_rows:
        t = _num(r, "Total Time") or 0.0
        total += t
        if NCCL_RE.search(r.get("Name", "")):
            nccl_total += t
            nccl_count += int(_num(r, "Instances") or 1)
    return {
        "total_kernel_ms": round(total / 1e6, 3),
        "nccl_ms": round(nccl_total / 1e6, 3),
        "nccl_share_pct": round(nccl_total / total * 100, 2) if total else 0.0,
        "nccl_instances": nccl_count,
        "avg_nccl_us": round(nccl_total / nccl_count / 1e3, 2) if nccl_count else 0.0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="量化 NCCL 通信占比与重叠率")
    ap.add_argument("--kern-sum", default="profiling/raw/kern_sum.csv")
    ap.add_argument("--trace", default="profiling/raw/gpu_trace.csv")
    ap.add_argument("--expected-mb", type=float, default=None,
                    help="可选：comm_ledger 理论通信量 MB，用于近似反推 B_eff（注明非实测字节）")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    kern_path = Path(args.kern_sum)
    if not kern_path.exists():
        print(f"[SKIP] 找不到 {kern_path}。先用 nsys stats --report cuda_gpu_kern_sum 导出。")
        return 0

    rows = kern_summary(read_nsys_csv(kern_path))

    trace_path = Path(args.trace)
    if trace_path.exists():
        trace_rows = read_nsys_csv(trace_path)
        nccl_iv = nccl_intervals(trace_rows)
        comp_iv = compute_intervals(trace_rows)
        comm = total_len(nccl_iv)
        ov = overlap_len(nccl_iv, comp_iv)
        rows["overlap_ratio"] = round(ov / comm, 4) if comm > 0 else 0.0
        rows["nccl_busy_ms"] = round(comm * 1e3, 3)
        rows["compute_busy_ms"] = round(total_len(comp_iv) * 1e3, 3)
        rows["nccl_bytes"] = io.FILL   # nsys 不提供字节数，见文件头的说明
        if args.expected_mb and comm > 0:
            rows["eff_bw_gbps"] = round(args.expected_mb * 1e6 / comm / 1e9, 2)
            rows["eff_bw_source"] = "approx_from_comm_ledger"
        else:
            rows["eff_bw_gbps"] = io.FILL
            rows["eff_bw_source"] = io.FILL
    else:
        rows["overlap_ratio"] = io.FILL
        rows["nccl_busy_ms"] = io.FILL
        rows["compute_busy_ms"] = io.FILL
        rows["nccl_bytes"] = io.FILL
        rows["eff_bw_gbps"] = io.FILL
        rows["eff_bw_source"] = io.FILL
        print(f"[WARN] 找不到 {trace_path}：只输出占比，重叠率写 __FILL__", file=sys.stderr)

    env = {"source": kern_path.name, "trace": trace_path.name if trace_path.exists() else "none"}
    out = io.write_csv(io.result_path("nccl_summary", args.out), [rows], env)
    for k, v in rows.items():
        print(f"{k:>20}: {v}")
    print(f"\n写出 {out}")
    print("判读：NCCL 占比高但重叠率高 → 不是瓶颈；两者都高才是瓶颈。"
          "字节数请与 ledger.comm_ledger 的理论值对账。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
