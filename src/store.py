"""Shared helpers: OpenAI client, embeddings, access to the Chroma collection."""
from __future__ import annotations

import chromadb
from openai import OpenAI

import config

_openai: OpenAI | None = None
_chroma: chromadb.ClientAPI | None = None


def openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        if not config.OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY is not set in .env")
        _openai = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts (in batches)."""
    client = openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i : i + config.EMBED_BATCH]
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def get_collection():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
