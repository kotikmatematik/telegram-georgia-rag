"""Shared helpers: OpenAI client, embeddings, access to the Chroma collection."""
from __future__ import annotations

import json

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


def chat_json(
    model: str, system: str, user: str, *, temperature: float = 0, reasoning_effort: str | None = None
) -> dict:
    """One JSON-mode chat call, picking the right parameter shape for the model.

    Reasoning-family models (config.REASONING_MODELS: gpt-5.6-*, o1/o3-*) only
    accept the default `temperature` and use `reasoning_effort` instead;
    classic chat models (gpt-4o*, gpt-4.1*) are the other way round. Callers
    just pass both and this picks what's actually sent, so swapping a model in
    config.py doesn't require touching call sites.

    Returns the parsed JSON object, or {} if the model didn't return valid JSON.
    """
    client = openai_client()
    kwargs: dict = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model in config.REASONING_MODELS:
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        # no `temperature`: these models only support the API default (1).
    else:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


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
