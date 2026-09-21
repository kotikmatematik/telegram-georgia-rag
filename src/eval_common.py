"""Shared helper for the retrieval/RAG evaluation modules (src/eval_retrieval.py,
src/eval_rag.py) — loading the hand-curated golden query set.

The golden set lives in eval/golden_queries.jsonl (repo root, tracked in git),
NOT under data/ — data/ is entirely gitignored (it holds real chat content),
so a hand-curated file that must grow via normal commits can't live there.
"""
from __future__ import annotations

import json
from pathlib import Path

import config


def load_golden(*, category: str | None = None, path: Path | None = None) -> list[dict]:
    """Load eval/golden_queries.jsonl, optionally filtered to one category
    (multi_source / one_answer / no_answer / critical)."""
    path = path or (config.EVAL_DIR / "golden_queries.jsonl")
    if not path.exists():
        raise SystemExit(f"{path} not found")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if category is not None:
        rows = [r for r in rows if r["category"] == category]
    return rows
