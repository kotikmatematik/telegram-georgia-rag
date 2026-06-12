"""Reconstruct conversation threads from a list of messages.

A "thread" is a connected component over two relations:
  1. reply links  — message.reply_to points to another message, and
  2. time-bursts   — consecutive messages within CHUNK_MAX_GAP_MINUTES.

Relation (1) joins formal replies (even days apart); relation (2) joins answers
people typed WITHOUT using reply. Together they group a whole discussion —
including the unrelated time-neighbours that may sneak in (the LLM distillation
step is expected to ignore those).

Run (stats):  uv run python -m src.threads
"""
from __future__ import annotations

import config
from src.preprocess import _load_raw, _time_bursts


class _UnionFind:
    def __init__(self, ids: list[int]):
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_threads(msgs: list[dict], *, gap_minutes: float = None) -> list[list[dict]]:
    """Group messages into threads. Each thread is sorted chronologically;
    threads are ordered by their root (lowest) message id."""
    gap_minutes = config.CHUNK_MAX_GAP_MINUTES if gap_minutes is None else gap_minutes
    by_id = {m["msg_id"]: m for m in msgs}
    uf = _UnionFind(list(by_id))

    # (1) reply links
    for m in msgs:
        parent = m.get("reply_to")
        if parent in by_id:
            uf.union(m["msg_id"], parent)

    # (2) time-bursts: union all members of each burst
    burst_map = _time_bursts(msgs, gap_minutes * 60)
    seen_bursts: set[int] = set()
    for burst in burst_map.values():
        key = id(burst)
        if key in seen_bursts:
            continue
        seen_bursts.add(key)
        for m in burst[1:]:
            uf.union(burst[0]["msg_id"], m["msg_id"])

    # collect components
    comps: dict[int, list[dict]] = {}
    for m in msgs:
        comps.setdefault(uf.find(m["msg_id"]), []).append(m)

    threads = [sorted(c, key=lambda x: x["msg_id"]) for c in comps.values()]
    threads.sort(key=lambda t: t[0]["msg_id"])
    return threads


def main() -> None:
    for chat in config.CHATS:
        path = config.RAW_DIR / f"{chat['username']}.jsonl"
        if not path.exists():
            print(f"[threads] skip {chat['username']}: {path} not found")
            continue
        msgs = _load_raw(path)
        threads = build_threads(msgs)
        sizes = [len(t) for t in threads]
        multi = [s for s in sizes if s > 1]
        print(
            f"[threads] {chat['username']}: {len(msgs)} messages -> "
            f"{len(threads)} threads "
            f"({len(multi)} multi-message, max {max(sizes) if sizes else 0} msgs)"
        )


if __name__ == "__main__":
    main()
