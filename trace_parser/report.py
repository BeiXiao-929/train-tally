"""trace_parser/report.py —— Trace 解析的核心逻辑与汇报入口。

文件职责
--------
1. 分类规则（外置 YAML，换框架只改配置不改代码）
2. 时间占比统计：每类事件的 GPU 时长之和 / 墙钟时间
3. **重叠率**：NCCL 与 Compute 的时间交集 / NCCL 总时长
4. 汇总成 Markdown + CSV，供 docs/selection_boundary.md 与报告引用
5. `--self-test`：合成一份已知重叠率的 trace，验证解析链路（不需要真实 trace）

为什么重叠率比「通信占比」更重要
--------------------------------
通信占比 30% 但重叠率 90% 的训练，实际损失可能只有 3%。
只看占比会得出「通信是瓶颈」的错误结论——这也正是本项目要纠正的常见误判。
重叠率 = 0 才是「通信完全串行」，那才是真的瓶颈。

TODO
----
[ ] 增加「每类事件的时间线直方图」（按 1ms 桶），定位通信突发
[ ] 支持按 step 切窗（当前按整段 trace 统计）
[ ] 增加 dataloader / host 侧空隙统计（需要 cpu_op 事件）
[ ] 重叠率改为「按 NCCL 消息粒度」计算，而不是全局区间并集

验收标准
--------
- --self-test 通过：合成数据重叠率 50%，解析结果与期望差 < 1e-6
- 各类占比之和 = 100% ± 0.01
- 空 trace 不崩溃，输出全 0 并告警
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import io

Interval = Tuple[float, float]  # (start_s, end_s)

DEFAULT_RULES: List[Tuple[str, str]] = [
    ("nccl_allreduce", r"nccl.*all_?reduce|allreduce"),
    ("nccl_reducescatter", r"nccl.*reduce_?scatter|reducescatter"),
    ("nccl_allgather", r"nccl.*all_?gather|allgather"),
    ("nccl_other", r"nccl|alltoall|broadcast|sendrecv"),
    ("memcpy", r"memcpy|memset"),
    ("compute", r"."),          # 兜底：其余都算计算
]


@dataclass
class Event:
    name: str
    start_s: float
    dur_s: float
    cat: str = "kernel"

    @property
    def end_s(self) -> float:
        return self.start_s + self.dur_s


# --------------------------------------------------------------------------- #
# 规则
# --------------------------------------------------------------------------- #
def load_rules(path: Optional[str] = None) -> List[Tuple[str, "re.Pattern[str]"]]:
    """读 YAML 规则；没有 pyyaml 或文件时退回内置规则。"""
    labels: List[Tuple[str, str]] = DEFAULT_RULES
    if path and Path(path).exists():
        try:
            import yaml

            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            items = data.get("rules", [])
            if items:
                labels = [(str(i["label"]), str(i["pattern"])) for i in items]
        except Exception as exc:
            print(f"[WARN] 读取规则失败（{exc}），使用内置规则", file=sys.stderr)
    return [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in labels]


def classify(name: str, rules: Sequence[Tuple[str, "re.Pattern[str]"]] ) -> str:
    for label, pattern in rules:
        if pattern.search(name):
            return label
    return "other"


# --------------------------------------------------------------------------- #
# 区间运算
# --------------------------------------------------------------------------- #
def merge(intervals: Iterable[Interval]) -> List[Interval]:
    items = sorted((s, e) for s, e in intervals if e > s)
    if not items:
        return []
    merged = [list(items[0])]
    for s, e in items[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def total_len(intervals: Iterable[Interval]) -> float:
    return sum(e - s for s, e in merge(intervals))


def overlap_len(a: Iterable[Interval], b: Iterable[Interval]) -> float:
    """两个区间集合的交集总长度（先去重叠，避免重复计数）。"""
    A, B = merge(a), merge(b)
    i = j = 0
    total = 0.0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if e > s:
            total += e - s
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return total


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def summary(events: List[Event],
            rules: Sequence[Tuple[str, "re.Pattern[str]"]] ) -> Dict[str, Any]:
    if not events:
        return {"wall_s": 0.0, "rows": [], "totals": {}}
    wall_start = min(e.start_s for e in events)
    wall_end = max(e.end_s for e in events)
    wall = max(wall_end - wall_start, 1e-9)

    buckets: Dict[str, Dict[str, Any]] = {}
    for e in events:
        label = classify(e.name, rules)
        b = buckets.setdefault(label, {"label": label, "count": 0,
                                       "total_s": 0.0, "intervals": []})
        b["count"] += 1
        b["total_s"] += e.dur_s
        b["intervals"].append((e.start_s, e.end_s))

    compute = [iv for b in buckets.values() if b["label"] == "compute"
               for iv in b["intervals"]]
    nccl = [iv for b in buckets.values() if b["label"].startswith("nccl")
            for iv in b["intervals"]]

    rows = []
    for label, b in sorted(buckets.items(), key=lambda kv: -kv[1]["total_s"]):
        rows.append({
            "category": label,
            "count": b["count"],
            "total_ms": round(b["total_s"] * 1e3, 3),
            "share_pct": round(b["total_s"] / wall * 100, 2),
            "avg_us": round(b["total_s"] / b["count"] * 1e6, 2),
        })

    comm_total = sum(e - s for s, e in merge(nccl))
    overlap = overlap_len(nccl, compute)
    return {
        "wall_s": wall,
        "rows": rows,
        "totals": {
            "events": len(events),
            "nccl_ms": round(comm_total * 1e3, 3),
            "compute_ms": round(total_len(compute) * 1e3, 3),
            "nccl_share_pct": round(comm_total / wall * 100, 2),
            "overlap_ms": round(overlap * 1e3, 3),
            "overlap_ratio": round(overlap / comm_total, 4) if comm_total > 0 else 0.0,
            "gpu_busy_pct": round(
                total_len([(e.start_s, e.end_s) for e in events]) / wall * 100, 2),
        },
    }


def to_markdown(res: Dict[str, Any], env: Optional[Dict[str, Any]] = None) -> str:
    t = res["totals"]
    note = (f"墙钟 {res['wall_s'] * 1e3:.1f} ms，NCCL 占比 {t.get('nccl_share_pct')}%，"
            f"**重叠率 {t.get('overlap_ratio')}**（1=完全被计算掩盖），"
            f"GPU 忙碌率 {t.get('gpu_busy_pct')}%。")
    lines = ["# Trace 解析报告", ""]
    if env:
        lines += ["> " + "; ".join(f"{k}={v}" for k, v in env.items()), ""]
    lines += [note, "", "| 类别 | 次数 | 总时长 ms | 占比 % | 平均 us |",
              "|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['category']} | {r['count']} | {r['total_ms']} "
                     f"| {r['share_pct']} | {r['avg_us']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 自检：合成已知重叠率的 trace
# --------------------------------------------------------------------------- #
def _synthetic_events(overlap_frac: float = 0.5) -> Tuple[List[Event], float]:
    """构造 4 个 step 的假 trace：
    - compute 每 step 占 [i*10, i*10+8]
    - nccl    每 step 占 [i*10+4, i*10+6]，与 compute 交集 2/2 = 100%？
    为了让 overlap 可控，第二个 step 的 nccl 放在 [10+8, 10+10]（不重叠）。
    期望重叠率：(2+0+2+0)/4 = 0.5
    """
    events: List[Event] = []
    for i in range(4):
        base = i * 10.0
        events.append(Event("vectorized_elementwise_kernel", base, 8.0, "kernel"))
        start = base + 4.0 if i % 2 == 0 else base + 8.0
        events.append(Event("ncclKernel_AllReduce_RING_LL", start, 2.0, "kernel"))
    return events, overlap_frac


def self_test() -> int:
    events, expected = _synthetic_events()
    rules = load_rules()
    res = summary(events, rules)
    got = res["totals"]["overlap_ratio"]
    ok_overlap = abs(got - expected) < 1e-6
    shares = sum(r["share_pct"] for r in res["rows"])
    ok_shares = abs(shares - 100.0) < 0.5
    ok_busy = 0 < res["totals"]["gpu_busy_pct"] <= 100

    print(f"重叠率 期望 {expected:.4f} 实得 {got:.4f}  {'OK' if ok_overlap else 'FAIL'}")
    print(f"占比合计 {shares:.2f}%  {'OK' if ok_shares else 'FAIL'}")
    print(f"GPU 忙碌率 {res['totals']['gpu_busy_pct']}%  {'OK' if ok_busy else 'FAIL'}")
    print(f"NCCL 占比 {res['totals']['nccl_share_pct']}%  "
          f"（20ms/40ms 应为 20%）")
    if all([ok_overlap, ok_shares, ok_busy]):
        print("self-test: PASS")
        return 0
    print("self-test: FAIL")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Trace 解析与汇报")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--trace", default="", help="chrome trace json")
    ap.add_argument("--rules", default="trace_parser/classify_rules.yaml")
    ap.add_argument("--csv", default="", help="results 目录，汇总已有 CSV（占位实现）")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    rules = load_rules(args.rules)

    if args.trace:
        from trace_parser.parse_torch_profiler import parse_chrome_trace

        path = Path(args.trace)
        if not path.exists():
            print(f"[SKIP] 找不到 {path}。导出方式见 trace_parser/README.md")
            return 0
        events = parse_chrome_trace(path)
        res = summary(events, rules)
        env = {"source": path.name, "events": len(events)}
        out = io.write_csv(io.result_path("trace_summary", args.out),
                           res["rows"], env, extra_meta=res["totals"])
        md = Path(args.out) / (out.stem + ".md")
        md.write_text(to_markdown(res, env), encoding="utf-8")
        print(to_markdown(res, env))
        print(f"写出 {out}\n写出 {md}")
        return 0

    if args.csv:
        print("[INFO] --csv 汇总请用 `python -m trace_parser.report --trace ...` 逐份解析后合并；"
              "当前先支持单份 trace。")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
