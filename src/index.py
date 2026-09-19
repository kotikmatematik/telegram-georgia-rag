"""Embed distilled knowledge units and load them into Chroma — incrementally.

Reads data/knowledge/<chat>.fixed.jsonl for every configured chat. Diffs
against what's already in the collection by a content hash (question+answer+
type+city) stored in each vector's metadata, and only:
  - embeds + upserts units that are NEW or whose content actually CHANGED;
  - deletes vectors whose unit no longer exists (dropped, or a thread that
    got re-processed and now has a different set of units);
  - leaves everything else untouched — no re-embedding, no API cost.

This matters once you're indexing several active chats (thousands of units):
re-embedding everything on every run (the old full-rebuild approach) would
mean paying for and re-computing vectors that never changed. At the current
~100-unit scale a full rebuild was cheap enough not to matter, but the diff
approach costs the same at small scale and scales correctly, so there's no
reason not to use it now.

Each knowledge unit's id is stable across runs: "<chat_username>:<root_msg_id>:<k>",
where k is its position among units sharing that root_msg_id (not a global
file position) — so unrelated threads never shift each other's ids, and a
thread re-processed by src.update_knowledge only touches its OWN ids.

Run:  uv run python -m src.index
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import config
from src.store import embed_texts, get_collection


def _load_knowledge(username: str) -> list[dict]:
    path = config.KNOWLEDGE_DIR / f"{username}.fixed.jsonl"
    if not path.exists():
        print(f"[index] skip {username}: {path} not found (run src.fix_knowledge first)")
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _content_hash(u: dict) -> str:
    link = u.get("source_link") or u["root_link"]
    payload = f"{u['question']}\n{u['answer']}\n{u['type']}\n{u.get('city') or ''}\n{link}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _assign_ids(units: list[dict]) -> dict[str, dict]:
    """id = '<chat>:<root_msg_id>:<k>', k = position within that root's group
    (stable regardless of other threads/units elsewhere in the file)."""
    by_root: dict[tuple, list[dict]] = defaultdict(list)
    for u in units:
        by_root[(u["chat_username"], u["root_msg_id"])].append(u)

    out: dict[str, dict] = {}
    for (username, root_msg_id), group in by_root.items():
        for k, u in enumerate(group):
            out[f"{username}:{root_msg_id}:{k}"] = u
    return out


def main() -> None:
    units: list[dict] = []
    for chat in config.CHATS:
        units.extend(_load_knowledge(chat["username"]))
    if not units:
        raise SystemExit("No knowledge found. Run src.knowledge / src.eval_knowledge / src.fix_knowledge first")

    desired = _assign_ids(units)
    desired_hash = {i: _content_hash(u) for i, u in desired.items()}

    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    existing_hash = {
        i: (m or {}).get("content_hash")
        for i, m in zip(existing["ids"], existing["metadatas"])
    }

    to_delete = [i for i in existing_hash if i not in desired_hash]
    to_upsert = [i for i, h in desired_hash.items() if existing_hash.get(i) != h]
    unchanged = len(desired_hash) - len(to_upsert)

    if to_delete:
        collection.delete(ids=to_delete)

    total = len(to_upsert)
    for i in range(0, total, config.EMBED_BATCH):
        batch_ids = to_upsert[i : i + config.EMBED_BATCH]
        batch = [desired[bid] for bid in batch_ids]
        embeddings = embed_texts([f"{u['question']}\n{u['answer']}" for u in batch])
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=[f"{u['question']}\n{u['answer']}" for u in batch],
            metadatas=[
                {
                    "question": u["question"],
                    "answer": u["answer"],
                    "type": u["type"],
                    "city": u.get("city") or "",
                    "date": u.get("date", ""),
                    "chat_title": u["chat_title"],
                    "chat_username": u["chat_username"],
                    # source_link (the specific message the answer came from)
                    # is more precise than root_link (the whole thread's
                    # first message, unrelated in a long thread) — fall back
                    # to root_link for knowledge distilled before this field
                    # existed.
                    "link": u.get("source_link") or u["root_link"],
                    "content_hash": desired_hash[bid],
                }
                for bid, u in zip(batch_ids, batch)
            ],
        )
        print(f"[index] upserted {min(i + config.EMBED_BATCH, total)}/{total}")

    print(
        f"[index] done. +/~{len(to_upsert)} upserted, {len(to_delete)} deleted, "
        f"{unchanged} unchanged. Records in collection: {collection.count()}"
    )


if __name__ == "__main__":
    main()
