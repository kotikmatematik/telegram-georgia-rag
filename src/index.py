"""Embed distilled knowledge units and load them into Chroma.

Reads data/knowledge/<chat>.fixed.jsonl (the output of src.fix_knowledge —
validated, corrected knowledge), NOT the raw chunks anymore: the old
chunk-based index (data/chunks/*) is retired now that distillation +
validation + fix produce a cleaner, atomic knowledge base directly.

Each knowledge unit becomes ONE vector, embedding "question\\nanswer" (matches
how src.consolidate used to compare units, before consolidation was dropped —
see project memory: no offline merge step, retrieval + rag.py handle
overlapping/near-duplicate sources at query time using type/date metadata).

Rebuilds the WHOLE collection from scratch on every run rather than upserting
incrementally: at this corpus size (~100s of units per chat) a full rebuild
is cheap and avoids orphaned vectors left behind when a fix run changes which
units exist for a thread (a unit that disappears between runs would otherwise
never get removed from an incremental upsert).

Run:  uv run python -m src.index
"""
from __future__ import annotations

import json

import config
from src.store import embed_texts, get_collection


def _load_knowledge(username: str) -> list[dict]:
    path = config.KNOWLEDGE_DIR / f"{username}.fixed.jsonl"
    if not path.exists():
        print(f"[index] skip {username}: {path} not found (run src.fix_knowledge first)")
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    units: list[dict] = []
    for chat in config.CHATS:
        units.extend(_load_knowledge(chat["username"]))
    if not units:
        raise SystemExit("No knowledge found. Run src.knowledge / src.eval_knowledge / src.fix_knowledge first")

    import chromadb  # local import: only needed here, to drop+recreate the collection

    _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        _client.delete_collection(name=config.COLLECTION_NAME)
    except Exception:
        pass  # collection doesn't exist yet (first run) — nothing to drop
    collection = get_collection()

    total = 0
    for i in range(0, len(units), config.EMBED_BATCH):
        batch = units[i : i + config.EMBED_BATCH]
        embeddings = embed_texts([f"{u['question']}\n{u['answer']}" for u in batch])
        collection.upsert(
            ids=[f"{u['chat_username']}:{u['root_msg_id']}:{i + j}" for j, u in enumerate(batch)],
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
                    "link": u["root_link"],
                }
                for u in batch
            ],
        )
        total += len(batch)
        print(f"[index] indexed {total}/{len(units)}")

    print(f"[index] done. Records in collection: {collection.count()}")


if __name__ == "__main__":
    main()
