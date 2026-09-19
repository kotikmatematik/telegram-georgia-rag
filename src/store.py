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
        if config.USE_AZURE_OPENAI:
            if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY):
                raise SystemExit(
                    "AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY are not set in .env "
                    "(config.USE_AZURE_OPENAI is True)"
                )
            # Azure's v1-compatible surface: plain OpenAI client, just a
            # different base_url — no AzureOpenAI class / api_version needed.
            _openai = OpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                base_url=config.AZURE_OPENAI_ENDPOINT,
            )
        else:
            if not config.OPENAI_API_KEY:
                raise SystemExit("OPENAI_API_KEY is not set in .env")
            _openai = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai


def chat_json(
    model: str, system: str, user: str, *, temperature: float = 0, reasoning_effort: str | None = None
) -> dict:
    """One JSON-mode chat call, picking the right parameter shape for the model.

    Reasoning-family models (config.REASONING_MODELS: gpt-5.6-*, gpt-5-mini,
    gpt-5.4-mini, o1/o3-*) reject a custom `temperature` outright (API error —
    only their default is allowed) and use `reasoning_effort` instead; classic
    chat models (gpt-4o*, gpt-4.1*) take `temperature`, no `reasoning_effort`.
    Callers just pass both and this picks what's actually sent, so swapping a
    model in config.py doesn't require touching call sites.

    `config.LLM_SEED` is always sent (every tested model accepts `seed`,
    including the reasoning ones that reject `temperature`) — it's the one
    lever for "give me the same answer again" that works everywhere, since
    temperature=0 isn't available on reasoning models at all.

    Returns the parsed JSON object, or {} if the model didn't return valid JSON.
    """
    client = openai_client()
    kwargs: dict = {
        "model": config.azure_deployment(model),
        "response_format": {"type": "json_object"},
        "seed": config.LLM_SEED,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model in config.REASONING_MODELS:
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        # no `temperature`: these models reject anything but their default.
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
        resp = client.embeddings.create(model=config.azure_deployment(config.EMBED_MODEL), input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def get_collection():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
