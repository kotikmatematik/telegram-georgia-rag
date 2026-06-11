"""Embed chunks and load them into Chroma.

Idempotent: each chunk id is deterministic (chat:msg_id), so re-running
updates existing records instead of creating duplicates.

Run:  uv run python -m src.index
"""
from __future__ import annotations

import json

import config
from src.store import embed_texts, get_collection


def _load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for chat in config.CHATS:
        path = config.CHUNKS_DIR / f"{chat['username']}.jsonl"
        if not path.exists():
            print(f"[index] skip {chat['username']}: {path} not found")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def main() -> None:
    chunks = _load_chunks()
    if not chunks:
        raise SystemExit("No chunks found. Run src.preprocess first")

    collection = get_collection()
    total = 0
    for i in range(0, len(chunks), config.EMBED_BATCH):
        batch = chunks[i : i + config.EMBED_BATCH]
        embeddings = embed_texts([c["text"] for c in batch])
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "chat_title": c["chat_title"],
                    "chat_username": c["chat_username"],
                    "link": c["link"],
                    "date": c["date"],
                }
                for c in batch
            ],
        )
        total += len(batch)
        print(f"[index] indexed {total}/{len(chunks)}")

    print(f"[index] done. Records in collection: {collection.count()}")


if __name__ == "__main__":
    main()
