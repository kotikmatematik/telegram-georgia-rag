"""Эмбеддинг чанков и загрузка их в Chroma.

Идемпотентно: id чанка детерминирован (chat:msg_id), повторный запуск
обновляет записи, а не плодит дубли.

Запуск:  uv run python -m src.index
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
            print(f"[index] пропуск {chat['username']}: нет {path}")
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
        raise SystemExit("Нет чанков. Сначала запусти src.preprocess")

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
        print(f"[index] проиндексировано {total}/{len(chunks)}")

    print(f"[index] готово. Записей в коллекции: {collection.count()}")


if __name__ == "__main__":
    main()
