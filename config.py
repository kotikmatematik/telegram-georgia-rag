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

# --- OpenAI models ---
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
# Eval judge: deliberately a stronger, different model than CHAT_MODEL so it
# doesn't share the distiller's blind spots (src/eval_knowledge.py).
JUDGE_MODEL = "gpt-4o"
# Second opinion for faithful=false drops (src/fix_knowledge.py): a genuine
# third model, stronger than both CHAT_MODEL and JUDGE_MODEL, for a real
# cross-model check rather than distiller self-consistency.
REVERIFY_MODEL = "gpt-4.1"

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


# --- Pipeline parameters ---
FILTER_SPAM = True           # drop spam (money / drugs / ads / pets) before chunking; questions are kept
CHUNK_MAX_GAP_MINUTES = 10   # if the gap between messages exceeds this, start a new chunk
CHUNK_MAX_CHARS = 1500       # max chunk size in characters
CHUNK_MIN_CHARS = 40         # drop chunks shorter than this (low signal)
REPLY_CHAIN_MAX_MSGS = 50    # follow full reply chains, but stop after this many ancestors (safety ceiling)
REPLY_CONTEXT_MAX_CHARS = 2000  # cap total quoted reply context per chunk (each ancestor pulls its whole time-burst, bounded here)
TOP_K = 8                    # how many chunks to feed into the LLM context
EMBED_BATCH = 100            # batch size for embedding requests

# --- Chats ---
# `username` is used to build t.me/<username>/<msg_id> source links;
# `chat_id` is used by Telethon to actually fetch the chat history.
CHATS = [
    {"username": "helpgeorgia", "chat_id": -1001452236047, "title": "Взаимопомощь. Грузия"},
    {"username": "ipgeorgiachat", "chat_id": -1001670908431, "title": "ИП/Бизнес Грузия"},
    # Uncomment to add more chats when scaling up:
    # {"username": "paravaingeorgia", "chat_id": -1001512786455, "title": "Получение водительских прав в Грузии"},
    # {"username": "mygeorgia_chat", "chat_id": -1001486751358, "title": "ГРУЗИЯ ЧАТ"},
    # {"username": "tbilisi_girl", "chat_id": -1001549075106, "title": "Женский чат Тбилиси"},
    # {"username": "georgia_it", "chat_id": -1001688709586, "title": "Грузия IT чат"},
    # {"username": "gruzia_medicina", "chat_id": -1001781403833, "title": "Грузия медицина"},
    # {"username": "georgia_woman", "chat_id": -1001276829180, "title": "Тбилиси женский чат"},
]
