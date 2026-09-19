"""Project settings: paths, models, chat list, pipeline parameters."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # raw messages: data/raw/<chat>.jsonl
CHUNKS_DIR = DATA_DIR / "chunks"        # chunks: data/chunks/<chat>.jsonl
KNOWLEDGE_DIR = DATA_DIR / "knowledge"  # distilled Q&A: data/knowledge/<chat>.jsonl
CHROMA_DIR = ROOT / "chroma_db"         # persistent vector DB

for _d in (RAW_DIR, CHUNKS_DIR, KNOWLEDGE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", "")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE", "")

# --- Azure OpenAI (temporary alt. billing path, e.g. borrowed startup credits) ---
# Same models, same Chat Completions API + Structured Outputs — just routed
# through Azure's endpoint instead of OpenAI's directly, so nothing else in
# the pipeline needs to change. Data (data/raw, data/knowledge, chroma_db)
# is always local regardless of which endpoint produced it — there is
# nothing to migrate back when this access goes away, just flip
# USE_AZURE_OPENAI back to False and everything resumes hitting OpenAI
# directly with OPENAI_API_KEY.
#
# Uses Azure's newer v1-compatible surface (endpoint ending in /openai/v1) —
# that means the plain OpenAI() client works with just base_url + api_key, no
# AzureOpenAI class / dated api_version needed (see src/store.py).
USE_AZURE_OPENAI = True
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")   # https://<resource>.openai.azure.com/openai/v1
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
# Azure calls a deployed model by whatever "deployment name" whoever set it up
# chose — not guaranteed to equal the model name. Only fill this in if your
# friend's deployment names differ from the model names below; empty = assume
# identical (ask them to name deployments after the model, it's simplest).
AZURE_DEPLOYMENT_MAP: dict[str, str] = {}


def azure_deployment(model: str) -> str:
    """Translate an OpenAI model name to its Azure deployment name — a no-op
    (returns `model` unchanged) unless USE_AZURE_OPENAI + AZURE_DEPLOYMENT_MAP
    say otherwise, so this is always safe to call regardless of mode."""
    if not USE_AZURE_OPENAI:
        return model
    return AZURE_DEPLOYMENT_MAP.get(model, model)

# --- OpenAI models ---
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # kept as the "old model" reference value for reverts below

# Final answer generation (src/rag.py) — the live, user-facing step. Reasoning
# effort kept low: this runs synchronously per user question (unlike the
# offline pipeline stages), so latency matters more here.
# Revert: GENERATION_MODEL = CHAT_MODEL, GENERATION_REASONING_EFFORT = None.
GENERATION_MODEL = "gpt-5.6-luna"
GENERATION_REASONING_EFFORT = "low"

# --- Distillation stage 1 (src/knowledge.py), split into two sequential calls ---
# 1a splits the raw thread into semantic branches (message ids only, so the
# link back to original messages is explicit); 1b extracts knowledge from
# those branches PLUS the raw thread (raw thread is always ground truth — 1a's
# split is a draft 1b may correct, never fed to anything else as truth).
# Independently configurable so any piece can be reverted/swapped alone:
#   - revert the whole stage to the old single-call pipeline: set both models to CHAT_MODEL
#   - use gpt-4.1-mini only for extraction: EXTRACT_MODEL = "gpt-4.1-mini" (leave SPLIT_MODEL)
SPLIT_MODEL = "gpt-5.6-luna"
SPLIT_REASONING_EFFORT = "low"
EXTRACT_MODEL = "gpt-5.6-luna"
EXTRACT_REASONING_EFFORT = "low"

# Stage 3: main validation (src/eval_knowledge.py) — one call per THREAD,
# judging every knowledge unit extracted from it together.
# Tried gpt-5-mini/high (2026-09-12), then this. Revert to the classic model:
# JUDGE_MODEL = "gpt-4.1-mini", JUDGE_REASONING_EFFORT = None.
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_REASONING_EFFORT = "medium"

# Applying "fix" verdicts (src/fix_knowledge.py: type correction + re-atomize)
# — cheap classic model, separate from CHAT_MODEL (the distiller) so the two
# can be tuned independently. Tried gpt-5.4-mini (same as JUDGE_MODEL) on
# 2026-09-16 — reverted: 2-3x pricier for no real benefit at this stage.
FIX_MODEL = "gpt-4.1-mini"
FIX_REASONING_EFFORT = None

# Stage 4 (src/fix_knowledge.py): independent recheck of knowledge stage 3
# marked invalid/for removal — a genuine third model, deliberately NOT shown
# stage 3's verdict/reasoning, so it judges the raw thread fresh instead of
# anchoring on the previous call. Its decision is final for what it reviews.
# Revert: REVERIFY_MODEL = "gpt-4.1", REVERIFY_REASONING_EFFORT = None.
# Candidate to try later instead: "gpt-5.6-sol".
REVERIFY_MODEL = "gpt-5.6-terra"
REVERIFY_REASONING_EFFORT = "medium"

# Reasoning-family models accept only the default `temperature` (no custom
# value) and support `reasoning_effort`; classic chat models are the reverse.
# src/store.chat_json() uses this to pick the right call shape per model.
REASONING_MODELS = {
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
    "gpt-5-mini", "gpt-5.4-mini",
    "o1", "o1-mini", "o3", "o3-mini",
}

# Fixed seed for every LLM call (src/store.chat_json) — the one determinism
# lever that works uniformly: temperature=0 is rejected outright by the
# reasoning models above, so `seed` is what "give the same answer again"
# actually relies on across the whole pipeline.
LLM_SEED = 0

# --- Vector DB ---
COLLECTION_NAME = "georgia_chats"

# --- Ingest ---
# Cutoff by DATE, not by count: a fixed message count pulls ~2 years of a quiet
# chat but only ~3 months of a busy one. INGEST_SINCE is the oldest message we
# trust as source content; messages up to INGEST_PARENT_LOOKBACK_DAYS before it
# are still fetched, but only to serve as reply-parents / context for threads
# that have activity after the cutoff (see src/knowledge.py). Empty string =
# no date cutoff (fall back to INGEST_LIMIT alone).
INGEST_SINCE = "2025-03-01"       # ISO date; revisit when type-based ranking lands
INGEST_PARENT_LOOKBACK_DAYS = 30  # extra history before the cutoff, parents only
INGEST_LIMIT = 50000              # safety cap on messages fetched per chat per run


def ingest_since_dt() -> datetime | None:
    """INGEST_SINCE as a UTC datetime — the real cutoff for trusted content."""
    if not INGEST_SINCE:
        return None
    return datetime.fromisoformat(INGEST_SINCE).replace(tzinfo=timezone.utc)


def ingest_fetch_floor_dt() -> datetime | None:
    """How far back ingest actually fetches: the cutoff minus the parent-lookback
    tail. None means 'no floor' (bounded only by INGEST_LIMIT)."""
    since = ingest_since_dt()
    if since is None:
        return None
    return since - timedelta(days=INGEST_PARENT_LOOKBACK_DAYS)


# --- Incremental knowledge updates (src/update_knowledge.py) ---
# Weekly-ish re-run: threads whose latest activity is older than
# (processed_until - REPROCESS_OVERLAP_DAYS) are left untouched; anything more
# recent gets fully re-distilled + re-validated and REPLACES its old knowledge
# (not appended). The overlap exists because a thread can go quiet and then
# get a late straggler reply — measured on helpgeorgia: of threads quiet for
# >=1 day, 20.3% still get another reply later; >=3 days, only 8.6%. 3 days
# was picked as the balance between catching most late replies and not
# re-processing (and re-spending tokens on) too much every run.
REPROCESS_OVERLAP_DAYS = 3


# --- Pipeline parameters ---
FILTER_SPAM = True           # drop spam (money / drugs / ads / pets) before chunking; questions are kept
CHUNK_MAX_GAP_MINUTES = 10   # if the gap between messages exceeds this, start a new chunk
# Hard cap on a time-burst thread (src/threads.py) — prevents very busy chats
# (near-continuous activity, no natural gaps) from chaining thousands of
# unrelated messages into one giant "thread". Does not affect reply-based
# links, which are never capped.
THREAD_MAX_BURST_SIZE = 50
CHUNK_MAX_CHARS = 1500       # max chunk size in characters
CHUNK_MIN_CHARS = 40         # drop chunks shorter than this (low signal)
REPLY_CHAIN_MAX_MSGS = 50    # follow full reply chains, but stop after this many ancestors (safety ceiling)
REPLY_CONTEXT_MAX_CHARS = 2000  # cap total quoted reply context per chunk (each ancestor pulls its whole time-burst, bounded here)
TOP_K = 8                    # how many chunks to feed into the LLM context
# Below this cosine score a hit is noise, not a real match — measured on
# helpgeorgia: a genuine match scores 0.7+ and drops sharply after; a query
# with NO real answer in the base still returns hits, but all clustered
# 0.38-0.50 with no clear top pick. 0.5 cuts that noise while keeping
# legitimate secondary sources (e.g. a real second source scored 0.57).
RETRIEVAL_MIN_SCORE = 0.5
EMBED_BATCH = 100            # batch size for embedding requests

# --- Chats ---
# `username` is used to build t.me/<username>/<msg_id> source links (and as
# the internal slug for data/raw/<username>.jsonl etc. — for a private chat
# it's just that slug, not a real Telegram @username); `chat_id` is used by
# Telethon to actually fetch the chat history. `private: True` marks a closed
# chat with no public @username — src/ingest.py then builds source links as
# t.me/c/<internal_id>/<msg_id> instead, which only opens for members of that
# chat (Telegram's restriction, not ours) — see src/ingest.py::_chat_link.
# Fetching a private chat requires the Telethon session's own account
# (config.TELEGRAM_PHONE) to already be a member — it can't be added "blind".
CHATS = [
    {"username": "helpgeorgia", "chat_id": -1001452236047, "title": "Взаимопомощь. Грузия"},
    {"username": "ipgeorgiachat", "chat_id": -1001670908431, "title": "ИП/Бизнес Грузия"},
    {"username": "nogotochki", "chat_id": -1001318697228, "title": "Ноготочки", "private": True},
    # Uncomment to add more chats when scaling up:
    # {"username": "paravaingeorgia", "chat_id": -1001512786455, "title": "Получение водительских прав в Грузии"},
    # {"username": "mygeorgia_chat", "chat_id": -1001486751358, "title": "ГРУЗИЯ ЧАТ"},
    # {"username": "tbilisi_girl", "chat_id": -1001549075106, "title": "Женский чат Тбилиси"},
    # {"username": "georgia_it", "chat_id": -1001688709586, "title": "Грузия IT чат"},
    # {"username": "gruzia_medicina", "chat_id": -1001781403833, "title": "Грузия медицина"},
    # {"username": "georgia_woman", "chat_id": -1001276829180, "title": "Тбилиси женский чат"},
]
