"""Incremental knowledge update: re-process only recently-active threads
instead of the whole chat from scratch every time.

Keeps a small per-chat cursor file (data/knowledge/<username>.state.json:
{"processed_until": "<ISO date>"}) recording how far a previous run got.
Each run:
  1. overlap_start = processed_until - config.REPROCESS_OVERLAP_DAYS.
  2. select_threads(since=overlap_start) -> only threads active since then.
     This naturally re-catches threads that were already fully processed but
     got a late straggler reply within the overlap window — see
     config.REPROCESS_OVERLAP_DAYS for why 3 days specifically. Threads
     outside the window are NOT touched at all.
  3. distill + eval + fix just that subset (not the whole chat — cheaper,
     and doesn't re-spend tokens re-judging already-settled old knowledge).
  4. Merge into data/knowledge/<username>.jsonl and .fixed.jsonl: units
     belonging to a re-processed thread REPLACE their old version (old ones
     for that root_msg_id dropped, fresh ones appended); everything else in
     the file (threads outside the window) passes through untouched.
  5. processed_until = now.

First run (no state file yet): processed_until defaults to
config.ingest_since_dt(), so the very first incremental run re-covers
basically the same threads a full distill_chat run would.

Run:  uv run python -m src.update_knowledge                 # all configured chats
      uv run python -m src.update_knowledge helpgeorgia      # just one
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

import config
from src.eval_knowledge import judge_units
from src.fix_knowledge import drop_thread_duplicates, fix_batch
from src.knowledge import _mask_contacts, distill_threads, select_threads


def _state_path(username: str):
    return config.KNOWLEDGE_DIR / f"{username}.state.json"


def _load_state(username: str) -> dict:
    path = _state_path(username)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(username: str, state: dict) -> None:
    _state_path(username).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _merge(existing: list[dict], touched_roots: set[int], fresh: list[dict]) -> list[dict]:
    """Units belonging to a re-processed thread are REPLACED (old ones for
    that root_msg_id dropped, fresh ones appended); everything else passes
    through unchanged."""
    kept = [u for u in existing if u["root_msg_id"] not in touched_roots]
    return kept + fresh


def bootstrap_state(username: str, *, processed_until: datetime | None = None) -> None:
    """Seed the cursor WITHOUT reprocessing anything — use this once after a
    full distill_chat/eval/fix run you already trust, so the next incremental
    run doesn't redo all of that from scratch just to establish a starting
    point. Defaults processed_until to now."""
    ts = (processed_until or datetime.now(timezone.utc)).isoformat()
    _save_state(username, {"processed_until": ts})
    print(f"[update] {username}: state bootstrapped, processed_until={ts}")


def update_knowledge(username: str) -> None:
    chat = next(c for c in config.CHATS if c["username"] == username)
    state = _load_state(username)

    processed_until = (
        datetime.fromisoformat(state["processed_until"])
        if state.get("processed_until")
        else config.ingest_since_dt()
    )
    overlap_start = (
        processed_until - timedelta(days=config.REPROCESS_OVERLAP_DAYS)
        if processed_until is not None
        else None
    )

    threads = select_threads(username, since=overlap_start)
    if not threads:
        print(f"[update] {username}: nothing to (re)process")
        _save_state(username, {"processed_until": datetime.now(timezone.utc).isoformat()})
        return

    touched_roots = {t[0]["msg_id"] for t in threads}
    threads_by_root = {t[0]["msg_id"]: t for t in threads}
    since_label = overlap_start.date() if overlap_start else "the beginning"
    print(f"[update] {username}: (re)processing {len(threads)} threads (active since {since_label})")

    fresh_units = distill_threads(threads, chat)
    judged = judge_units(fresh_units, threads_by_root, label=f"update:eval {username}")
    result = fix_batch(fresh_units, judged, threads_by_root)
    fresh_fixed = drop_thread_duplicates(result["kept"] + result["fixed"])
    for u in fresh_fixed:  # same masking as save_fixed — contacts never reach .fixed.jsonl
        u["answer"] = _mask_contacts(u["answer"])

    # Raw (pre-validation) file — mirrors distill_chat's output, kept in sync.
    raw_path = config.KNOWLEDGE_DIR / f"{username}.jsonl"
    _write_jsonl(raw_path, _merge(_load_jsonl(raw_path), touched_roots, fresh_units))

    fixed_path = config.KNOWLEDGE_DIR / f"{username}.fixed.jsonl"
    fixed_merged = _merge(_load_jsonl(fixed_path), touched_roots, fresh_fixed)
    fixed_merged = drop_thread_duplicates(fixed_merged)  # safety net across the merge boundary too
    _write_jsonl(fixed_path, fixed_merged)

    print(f"[update] {username}: {len(fresh_fixed)} fresh knowledge units -> "
          f"{len(fixed_merged)} total in {fixed_path}")

    _save_state(username, {"processed_until": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    usernames = sys.argv[1:] or [c["username"] for c in config.CHATS]
    for username in usernames:
        path = config.RAW_DIR / f"{username}.jsonl"
        if not path.exists():
            print(f"[update] skip {username}: {path} not found")
            continue
        update_knowledge(username)


if __name__ == "__main__":
    main()
