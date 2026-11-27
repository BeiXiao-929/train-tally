"""trace_parser/parse_torch_profiler.py —— 解析 torch.profiler 导出的 chrome trace。

文件职责
--------
把 `torch.profiler` 的 chrome trace JSON 转成统一的 Event 列表（秒为单位），
交给 report.py 做分类、占比与重叠率统计。

导出方式（实验脚本里已内置 --profile 选项）：
    with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        ...
    p.export_chrome_trace("traces/step.json")

只取 GPU 侧事件
---------------
chrome trace 里同时有 CPU op（cat 为空或 "cpu_op"）与 GPU kernel（cat="kernel"）。
**只统计 GPU kernel**，因为我们要回答的是「GPU 时间去哪了」；
把 CPU op 混进来会让占比失去意义（一个 kernel 可能对应十几个 CPU op）。

TODO
----
[ ] 支持按 stream 区分（多流重叠时，GPU 时间的并集与求和不同）
[ ] 支持 NVTX range 过滤（只统计 forward 段）
[ ] 增加 memory 事件的峰值统计（cat=gpu_memcpy 的 args 里有 bytes）

验收标准
--------
- 任意 torch.profiler chrome trace 能被解析成 Event 列表，dur 为秒
- 事件按 start 升序；空 trace 返回空列表而不是抛异常
- 与 `key_averages()` 的 kernel 总时长对照，误差 < 1%（同一次采集）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import io
from trace_parser.report import Event, load_rules, summary, to_markdown

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "Kernel", "gpu_kernel"}


def parse_chrome_trace(path: str | Path, gpu_only: bool = True) -> List[Event]:
    data = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    raw = data.get("traceEvents", data if isinstance(data, list) else [])
    events: List[Event] = []
    for e in raw:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", ""))
        if gpu_only and cat not in GPU_CATS:
            continue
        dur = e.get("dur")
        ts = e.get("ts")
        if dur is None or ts is None:
            continue
        try:
            dur_s = float(dur) / 1e6   # chrome trace 的时间单位是微秒
            ts_s = float(ts) / 1e6
        except (TypeError, ValueError):
            continue
        if dur_s <= 0:
            continue
        name = str(e.get("name", "unknown"))
        events.append(Event(name=name, start_s=ts_s, dur_s=dur_s, cat=cat))
    events.sort(key=lambda x: x.start_s)
    return events


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="解析 torch.profiler chrome trace")
    ap.add_argument("--trace", default="traces/step.json")
    ap.add_argument("--rules", default="trace_parser/classify_rules.yaml")
    ap.add_argument("--all-events", action="store_true", help="包含 CPU 事件（默认只取 GPU）")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    path = Path(args.trace)
    if not path.exists():
        print(f"[SKIP] 找不到 {path}\n"
              "       导出方式：在训练脚本里 p.export_chrome_trace('traces/step.json')，"
              "traces/ 已被 .gitignore 忽略。")
        return 0

    events = parse_chrome_trace(path, gpu_only=not args.all_events)
    if not events:
        print("[WARN] 未解析到 GPU 事件：确认 profiler 开了 CUDA activity，"
              "且 cat 字段是 'kernel'", file=sys.stderr)
        return 1

    rules = load_rules(args.rules)
    res = summary(events, rules)
    env = {"source": path.name, "events": len(events)}
    out = io.write_csv(io.result_path("trace_summary", args.out), res["rows"], env,
                       extra_meta=res["totals"])
    md = Path(args.out) / (out.stem + ".md")
    md.write_bytes(to_markdown(res, env).encode("utf-8"))
    print(to_markdown(res, env))
    print(f"写出 {out}\n写出 {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
