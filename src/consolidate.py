"""Consolidate distilled knowledge: merge near-duplicate Q&A across threads.

The same recommendation/fact often appears in several threads. We cluster
knowledge units by embedding similarity and merge each cluster into one unit
that carries the consensus + recency signals used later for ranking:

  - support_count : how many sources said it (consensus)
  - sources       : all root links with their dates and types
  - date_start/end: time span of the supporting messages (recency)
  - type          : majority knowledge type (volatile/stable/evergreen)

The representative question/answer is taken from the MOST RECENT source, so a
volatile fact reflects the latest state.

Run:  uv run python -m src.consolidate
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

import config
from src.store import embed_texts


def _load_knowledge(username: str) -> list[dict]:
    path = config.KNOWLEDGE_DIR / f"{username}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cluster(emb: np.ndarray, threshold: float) -> list[list[int]]:
    """Greedy union-find clustering by cosine similarity (vectors pre-normalized)."""
    n = len(emb)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sim = emb @ emb.T
    for i in range(n):
        # only upper triangle
        for j in np.where(sim[i, i + 1 :] >= threshold)[0]:
            a, b = find(i), find(i + 1 + j)
            if a != b:
                parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def consolidate_knowledge(items: list[dict], *, threshold: float = 0.86) -> list[dict]:
    if not items:
        return []
    texts = [f"{it['question']}\n{it['answer']}" for it in items]
    emb = np.asarray(embed_texts(texts), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

    out: list[dict] = []
    for idx in _cluster(emb, threshold):
        members = [items[i] for i in idx]
        members.sort(key=lambda m: m.get("date") or "")
        rep = members[-1]  # most recent = representative
        dates = [m.get("date") or "" for m in members if m.get("date")]
        types = [m.get("type", "stable") for m in members]
        out.append(
            {
                "question": rep["question"],
                "answer": rep["answer"],
                "type": Counter(types).most_common(1)[0][0],
                "support_count": len(members),
                "date_start": min(dates) if dates else "",
                "date_end": max(dates) if dates else "",
                "sources": [
                    {"link": m["root_link"], "date": m.get("date", ""), "type": m.get("type", "stable")}
                    for m in members
                ],
                "chat_username": rep["chat_username"],
                "chat_title": rep["chat_title"],
            }
        )
    # most-confirmed first
    out.sort(key=lambda k: k["support_count"], reverse=True)
    return out


def main() -> None:
    for chat in config.CHATS:
        items = _load_knowledge(chat["username"])
        if not items:
            print(f"[consolidate] skip {chat['username']}: no knowledge file")
            continue
        consolidated = consolidate_knowledge(items)
        out_path = config.KNOWLEDGE_DIR / f"{chat['username']}.consolidated.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for k in consolidated:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        merged = sum(1 for k in consolidated if k["support_count"] > 1)
        print(
            f"[consolidate] {chat['username']}: {len(items)} -> {len(consolidated)} "
            f"units ({merged} merged from duplicates) -> {out_path}"
        )


if __name__ == "__main__":
    main()
