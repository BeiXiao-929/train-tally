"""common/io.py —— 结果落盘（带环境头）。

约定与单卡项目一致：CSV 前若干行 `# key=value`，正文可被 pandas 直接读；
无法给出的指标写 `__FILL__`，不写 0、不写估算。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FILL = "__FILL__"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def result_path(kind: str, outdir: str | Path = "results", ext: str = "csv") -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{kind}_{stamp()}.{ext}"


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]],
              env_info: Optional[Dict[str, Any]] = None,
              extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        if env_info:
            f.write("# " + "; ".join(f"{k}={v}" for k, v in env_info.items()) + "\n")
        if extra_meta:
            f.write("# meta " + "; ".join(f"{k}={v}" for k, v in extra_meta.items()) + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=FILL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, FILL) for k in fieldnames})
    return path


def read_csv(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    meta: Dict[str, Any] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            payload = line.lstrip("# ").strip()
            if payload.lower().startswith("meta "):
                payload = payload[5:]
            for chunk in payload.split(";"):
                if "=" in chunk:
                    k, v = chunk.split("=", 1)
                    meta[k.strip()] = v.strip()
            body_start = i + 1
        else:
            break
    data = list(csv.DictReader(lines[body_start:])) if body_start < len(lines) else []
    return meta, data


def write_markdown(path: str | Path, title: str, rows: List[Dict[str, Any]],
                   env_info: Optional[Dict[str, Any]] = None,
                   note: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    lines = [f"# {title}", ""]
    if env_info:
        lines += ["> " + "; ".join(f"{k}={v}" for k, v in env_info.items()), ""]
    if note:
        lines += [note, ""]
    if cols:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, FILL)) for c in cols) + " |")
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path
