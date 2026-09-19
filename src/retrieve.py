"""Search for relevant knowledge units by query (for debugging and reuse).

Run:  uv run python -m src.retrieve "как открыть ип в грузии"
"""
from __future__ import annotations

import sys

import config
from src.store import embed_texts, get_collection


def search(query: str, k: int = config.TOP_K, *, min_score: float = config.RETRIEVAL_MIN_SCORE) -> list[dict]:
    """Top-k by cosine score, then drop anything below min_score — see
    config.RETRIEVAL_MIN_SCORE for why: below that, hits are noise, not
    real matches (measured empirically, not guessed). Pass min_score=0 to
    get the raw top-k back (e.g. for debugging what was filtered out)."""
    collection = get_collection()
    q_emb = embed_texts([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    hits: list[dict] = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        score = 1 - dist
        if score >= min_score:
            hits.append({"text": doc, "meta": meta, "score": score})
    return hits


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.retrieve "your question"')
    query = " ".join(sys.argv[1:])
    for i, h in enumerate(search(query), 1):
        m = h["meta"]
        print(f"\n--- #{i}  score={h['score']:.3f}  {m['chat_title']}  {m['link']}")
        print(h["text"][:500])


if __name__ == "__main__":
    main()
