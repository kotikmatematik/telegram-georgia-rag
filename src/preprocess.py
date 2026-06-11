"""Clean messages and group them into "chunks".

Two mechanisms work together:

1. Time windows. Chat messages are short and often come as "question ->
   answers" bursts, so we group consecutive messages while the time gap stays
   below CHUNK_MAX_GAP_MINUTES and the size stays under CHUNK_MAX_CHARS.

2. Reply chains. A reply can arrive much later (even days), so time grouping
   alone would separate it from the message it answers. For any message whose
   parent (`reply_to`) is outside its chunk, we walk the FULL reply chain up to
   the root and prepend ancestors as quoted context (prefixed with "↪"). For
   each ancestor we pull not just that one message but its whole time-burst
   (the little surrounding conversation), since the real answer is often not a
   formal reply but a neighbouring message. This keeps question and answer
   together regardless of the time gap. No age limit; chain depth is capped by
   REPLY_CHAIN_MAX_MSGS and total quoted context by REPLY_CONTEXT_MAX_CHARS.

Run:  uv run python -m src.preprocess
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config
from src.spam import filter_spam


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


# A context_fn decides WHAT to quote around a chunk (the part worth researching).
# Signature: context_fn(buf, by_id, burst_map) -> list[dict] of context messages.
def ancestor_context(
    buf: list[dict],
    by_id: dict[int, dict],
    burst_map: dict[int, list[dict]],
    *,
    chain_max: int = None,
    context_max_chars: int = None,
    whole_burst: bool = True,
) -> list[dict]:
    """Default context strategy: walk reply_to chains upward for messages whose
    parent is outside the current chunk and collect the ancestors as context.

    Args:
        chain_max: max ancestors to walk per chain (default config.REPLY_CHAIN_MAX_MSGS).
        context_max_chars: total char cap for the quoted context (default config.REPLY_CONTEXT_MAX_CHARS).
        whole_burst: if True, for each ancestor pull its whole time-burst (the
            surrounding conversation); if False, only the single ancestor message.

    Returns the collected messages (oldest first), de-duplicated and size-capped.
    """
    chain_max = config.REPLY_CHAIN_MAX_MSGS if chain_max is None else chain_max
    context_max_chars = (
        config.REPLY_CONTEXT_MAX_CHARS if context_max_chars is None else context_max_chars
    )
    body_ids = {m["msg_id"] for m in buf}
    collected: dict[int, dict] = {}
    ctx_chars = 0

    def add(msg: dict) -> bool:
        """Add a message to the context. Returns False if the size cap is hit."""
        nonlocal ctx_chars
        sid = msg["msg_id"]
        if sid in body_ids or sid in collected:
            return True
        if ctx_chars + len(msg["text"]) > context_max_chars:
            return False
        collected[sid] = msg
        ctx_chars += len(msg["text"])
        return True

    for m in buf:
        parent_id = m.get("reply_to")
        steps = 0
        # msg ids strictly decrease up the chain, so this can't loop forever
        while parent_id and parent_id in by_id and steps < chain_max:
            if parent_id in body_ids:
                break  # parent already part of this chunk
            parent = by_id[parent_id]
            siblings = burst_map.get(parent_id, [parent]) if whole_burst else [parent]
            for sib in siblings:
                if not add(sib):
                    break  # size cap reached
            parent_id = parent.get("reply_to")
            steps += 1
    return [collected[i] for i in sorted(collected)]


def _flush(
    buf: list[dict],
    by_id: dict[int, dict],
    burst_map: dict[int, list[dict]],
    *,
    min_chars: int,
    context_fn,
) -> dict | None:
    """Combine the buffered messages (plus reply context) into one chunk."""
    if not buf:
        return None
    lines = []
    # Quoted context first (the thread this burst replies to), then the burst.
    for m in context_fn(buf, by_id, burst_map):
        sender = m.get("sender") or "Аноним"  # "Anonymous" shown in chunk text
        lines.append(f"↪ {sender}: {m['text']}")
    for m in buf:
        sender = m.get("sender") or "Аноним"
        lines.append(f"{sender}: {m['text']}")
    text = "\n".join(lines).strip()
    if len(text) < min_chars:
        return None
    first = buf[0]
    return {
        "id": f"{first['chat_username']}:{first['msg_id']}",
        "text": text,
        "chat_username": first["chat_username"],
        "chat_title": first["chat_title"],
        "first_msg_id": first["msg_id"],
        "last_msg_id": buf[-1]["msg_id"],
        "date": first.get("date") or "",
        "link": first["link"],
    }


def chunk_messages(
    msgs: list[dict],
    *,
    gap_minutes: float = None,
    max_chars: int = None,
    min_chars: int = None,
    context_max_chars: int = None,
    whole_burst: bool = True,
    chain_max: int = None,
    context_fn=None,
) -> list[dict]:
    """Group messages into chunks.

    All parameters are arguments (not globals) so you can sweep them from a
    notebook without editing config or fighting autoreload:

        gap_minutes:        start a new chunk when the gap exceeds this many minutes
        max_chars:          soft size cap for the message burst (the chunk body)
        min_chars:          drop chunks shorter than this
        context_max_chars:  total char cap for the quoted reply context
        whole_burst:        for each ancestor, pull its whole time-burst (True)
                            or only the single ancestor message (False)
        chain_max:          max ancestors to walk up a reply chain
        context_fn:         full override of the context strategy (see
                            ancestor_context). When given, the three context_*
                            / whole_burst / chain_max args above are ignored.
                            Use this to experiment with new logic from a notebook.
    """
    gap_minutes = config.CHUNK_MAX_GAP_MINUTES if gap_minutes is None else gap_minutes
    max_chars = config.CHUNK_MAX_CHARS if max_chars is None else max_chars
    min_chars = config.CHUNK_MIN_CHARS if min_chars is None else min_chars

    if context_fn is None:
        def context_fn(buf, by_id, burst_map):
            return ancestor_context(
                buf, by_id, burst_map,
                chain_max=chain_max,
                context_max_chars=context_max_chars,
                whole_burst=whole_burst,
            )

    # Lookups for reply-chain reconstruction (only messages we actually kept).
    by_id = {m["msg_id"]: m for m in msgs}
    gap = gap_minutes * 60
    burst_map = _time_bursts(msgs, gap)

    chunks: list[dict] = []
    buf: list[dict] = []
    buf_chars = 0
    prev_dt: datetime | None = None

    for m in msgs:
        dt = _parse_dt(m.get("date"))
        too_old = (
            prev_dt is not None
            and dt is not None
            and (dt - prev_dt).total_seconds() > gap
        )
        too_big = buf_chars + len(m["text"]) > max_chars
        if buf and (too_old or too_big):
            ch = _flush(buf, by_id, burst_map, min_chars=min_chars, context_fn=context_fn)
            if ch:
                chunks.append(ch)
            buf, buf_chars = [], 0
        buf.append(m)
        buf_chars += len(m["text"])
        prev_dt = dt

    ch = _flush(buf, by_id, burst_map, min_chars=min_chars, context_fn=context_fn)
    if ch:
        chunks.append(ch)
    return chunks


def process_chat(username: str) -> int:
    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        print(f"[preprocess] skip {username}: {raw_path} not found")
        return 0
    msgs = _load_raw(raw_path)
    removed: list[dict] = []
    if config.FILTER_SPAM:
        msgs, removed = filter_spam(msgs)
    chunks = chunk_messages(msgs)
    out_path = config.CHUNKS_DIR / f"{username}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    spam_note = f" (dropped {len(removed)} spam)" if removed else ""
    print(
        f"[preprocess] {username}: {len(msgs)} messages{spam_note} -> "
        f"{len(chunks)} chunks -> {out_path}"
    )
    return len(chunks)


def main() -> None:
    for chat in config.CHATS:
        process_chat(chat["username"])


if __name__ == "__main__":
    main()
