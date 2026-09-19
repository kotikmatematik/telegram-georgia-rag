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


def build_threads(
    msgs: list[dict], *, gap_minutes: float = None, max_burst_size: int = None
) -> list[list[dict]]:
    """Group messages into threads. Each thread is sorted chronologically;
    threads are ordered by their root (lowest) message id.

    max_burst_size caps a component two ways: (a) a single time-burst can only
    chain max_burst_size messages via relation (2) before starting a fresh
    chunk, and (b) as a final backstop, ANY resulting component still over
    max_burst_size (bursts bridged back together via reply links crossing
    chunk boundaries — measured on nogotochki: capping (a) alone barely
    helped, 58% of its messages have a reply_to, so replies kept re-stitching
    separately-capped burst chunks back into one blob) is split into
    chronological sub-groups of at most max_burst_size. Needed for very
    high-traffic chats: on nogotochki (a busy private chat, 42k+ messages),
    even a 30-SECOND gap still chained 28% of all messages into one component
    — the chat is close to continuously active, so shrinking the gap alone
    never isolates real discussions, it just delays where the chaining
    happens.
    """
    gap_minutes = config.CHUNK_MAX_GAP_MINUTES if gap_minutes is None else gap_minutes
    max_burst_size = config.THREAD_MAX_BURST_SIZE if max_burst_size is None else max_burst_size
    by_id = {m["msg_id"]: m for m in msgs}
    uf = _UnionFind(list(by_id))

    # (1) reply links
    for m in msgs:
        parent = m.get("reply_to")
        if parent in by_id:
            uf.union(m["msg_id"], parent)

    # (2) time-bursts: union all members of each burst, capped in chronological
    # chunks of at most max_burst_size (burst members are already time-ordered).
    burst_map = _time_bursts(msgs, gap_minutes * 60)
    seen_bursts: set[int] = set()
    for burst in burst_map.values():
        key = id(burst)
        if key in seen_bursts:
            continue
        seen_bursts.add(key)
        for i in range(0, len(burst), max_burst_size):
            chunk = burst[i : i + max_burst_size]
            for m in chunk[1:]:
                uf.union(chunk[0]["msg_id"], m["msg_id"])

    # collect components
    comps: dict[int, list[dict]] = {}
    for m in msgs:
        comps.setdefault(uf.find(m["msg_id"]), []).append(m)

    threads: list[list[dict]] = []
    for c in comps.values():
        c = sorted(c, key=lambda x: x["msg_id"])
        if len(c) <= max_burst_size:
            threads.append(c)
        else:
            # Final backstop: reply links can re-stitch separately-capped
            # burst chunks back together (see docstring). Split without
            # blindly severing a real reply conversation where avoidable.
            threads.extend(_split_oversized(c, max_burst_size))
    threads.sort(key=lambda t: t[0]["msg_id"])
    return threads


def _split_oversized(component: list[dict], max_size: int) -> list[list[dict]]:
    """Split an over-cap component (see build_threads) in a reply-aware way:
    a reply-connected sub-conversation (>=2 messages, purely via reply_to —
    time-bursts are ignored here) is treated as an ATOMIC block that's never
    torn across two output threads, as long as it fits under max_size on its
    own. Blocks (islands and lone, reply-isolated messages) are then packed
    chronologically into threads of at most max_size — so unrelated
    time-adjacent messages still get grouped the same as before, they just
    can't split a reply island in half. Only a reply island that's STILL over
    max_size by itself (rare — measured once, an 81-message chain, on
    nogotochki's 42k+ messages) gets chopped chronologically, and even then
    only within that island's own genuinely-connected messages, not mixed
    with unrelated neighbours.
    """
    ids_here = {m["msg_id"] for m in component}
    reply_uf = _UnionFind([m["msg_id"] for m in component])
    for m in component:
        parent = m.get("reply_to")
        if parent in ids_here:
            reply_uf.union(m["msg_id"], parent)

    groups: dict[int, list[dict]] = {}
    for m in component:
        groups.setdefault(reply_uf.find(m["msg_id"]), []).append(m)

    blocks = [sorted(g, key=lambda x: x["msg_id"]) for g in groups.values()]
    blocks.sort(key=lambda b: b[0]["msg_id"])

    out: list[list[dict]] = []
    bucket: list[dict] = []
    for block in blocks:
        if bucket and len(bucket) + len(block) > max_size:
            out.append(bucket)
            bucket = []
        if len(block) > max_size:
            if bucket:
                out.append(bucket)
                bucket = []
            for i in range(0, len(block), max_size):
                out.append(block[i : i + max_size])
        else:
            bucket.extend(block)
    if bucket:
        out.append(bucket)
    return out


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
