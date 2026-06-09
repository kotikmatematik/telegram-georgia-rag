"""Поиск релевантных чанков по запросу (для отладки и переиспользования).

Запуск:  uv run python -m src.retrieve "как открыть ип в грузии"
"""
from __future__ import annotations

import sys

import config
from src.store import embed_texts, get_collection


def search(query: str, k: int = config.TOP_K) -> list[dict]:
    collection = get_collection()
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    hits: list[dict] = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "meta": meta, "score": 1 - dist})
    return hits


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Использование: python -m src.retrieve "ваш вопрос"')
    query = " ".join(sys.argv[1:])
    for i, h in enumerate(search(query), 1):
        m = h["meta"]
        print(f"\n--- #{i}  score={h['score']:.3f}  {m['chat_title']}  {m['link']}")
        print(h["text"][:500])


if __name__ == "__main__":
    main()
