"""Load raw messages and group them into time-bursts.

Used by src.threads (thread reconstruction) and src.eval_knowledge — the
old chunk-based chunking pipeline that used to live here was superseded by
knowledge distillation (src.knowledge) and removed.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _load_raw(path: Path) -> list[dict]:
    msgs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                msgs.append(json.loads(line))
    # ascending id => chronological order
    msgs.sort(key=lambda m: m["msg_id"])
    return msgs


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _time_bursts(msgs: list[dict], gap_seconds: float) -> dict[int, list[dict]]:
    """Split messages into time-bursts (runs with gaps <= gap_seconds) and
    return a map msg_id -> the burst it belongs to."""
    bursts: list[list[dict]] = []
    cur: list[dict] = []
    prev_dt: datetime | None = None
    for m in msgs:
        dt = _parse_dt(m.get("date"))
        if cur and prev_dt and dt and (dt - prev_dt).total_seconds() > gap_seconds:
            bursts.append(cur)
            cur = []
        cur.append(m)
        prev_dt = dt
    if cur:
        bursts.append(cur)
    return {m["msg_id"]: b for b in bursts for m in b}
