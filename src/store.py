"""Shared helpers: OpenAI client, embeddings, access to the Chroma collection."""
from __future__ import annotations

import io
import json
import threading

import chromadb
from openai import OpenAI

import config

_openai: OpenAI | None = None
_chroma: chromadb.ClientAPI | None = None
# Both clients are lazily created on first use and cached in the globals
# above. That's safe under sequential use (the only pattern this file saw
# until src/eval_retrieval.py and src/eval_rag.py started calling search()/
# chat_json() from a ThreadPoolExecutor) but not under concurrent first-use:
# multiple threads could all see the global as None and race to construct
# it — for chromadb.PersistentClient specifically, that race corrupts its
# tenant validation (observed directly: "Could not connect to tenant
# default_tenant" on the very first eval_retrieval run). This lock makes
# that lazy init atomic; it's only ever held for the cheap one-time
# construction, not for every call.
_client_lock = threading.Lock()


def openai_client() -> OpenAI:
    global _openai
    if _openai is None:
        with _client_lock:
            if _openai is None:  # re-check: another thread may have won the race
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


_groq: OpenAI | None = None


def _groq_client() -> OpenAI:
    """A plain OpenAI()-shaped client pointed at Groq's OpenAI-compatible
    endpoint, used only for voice transcription. Separate from
    openai_client() because neither of the other two paths work for this:
    the Azure resource in use has no Whisper/transcription deployment
    (verified — every whisper/gpt-*-transcribe name 404s with
    DeploymentNotFound there), and the OpenAI-direct account has no
    balance. Groq hosts open-weight Whisper (whisper-large-v3-turbo) with a
    free tier (no card required), and its API is OpenAI-compatible — same
    client class, just a different base_url/key."""
    global _groq
    if _groq is None:
        with _client_lock:
            if _groq is None:
                if not config.GROQ_API_KEY:
                    raise SystemExit(
                        "GROQ_API_KEY is not set in .env (needed for voice "
                        "transcription — free key at https://console.groq.com/keys)"
                    )
                _groq = OpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    return _groq


def transcribe_audio(file_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Speech-to-text via Groq's hosted Whisper (see _groq_client above).
    filename's extension is a format hint the SDK reads off file.name for
    the multipart upload — Telegram voice messages are .ogg (Opus), which
    Whisper accepts directly, no conversion needed."""
    client = _groq_client()
    buf = io.BytesIO(file_bytes)
    buf.name = filename
    resp = client.audio.transcriptions.create(model=config.TRANSCRIBE_MODEL, file=buf)
    return (resp.text or "").strip()


def get_collection():
    global _chroma
    if _chroma is None:
        with _client_lock:
            if _chroma is None:  # re-check: another thread may have won the race
                _chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
