"""Общие настройки проекта: пути, модели, список чатов, параметры пайплайна."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Пути ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # сырые сообщения: data/raw/<chat>.jsonl
CHUNKS_DIR = DATA_DIR / "chunks"    # чанки: data/chunks/<chat>.jsonl
CHROMA_DIR = ROOT / "chroma_db"     # персистентная векторная БД

for _d in (RAW_DIR, CHUNKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Секреты ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", "")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE", "")

# --- Модели OpenAI ---
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# --- Векторная БД ---
COLLECTION_NAME = "georgia_chats"

# --- Параметры пайплайна ---
INGEST_LIMIT = 5000          # сколько последних сообщений выгружать на чат
CHUNK_TIME_GAP_MIN = 10      # разрыв (мин) между сообщениями => новый чанк
CHUNK_MAX_CHARS = 1500       # максимальный размер чанка в символах
CHUNK_MIN_CHARS = 40         # чанки короче — отбрасываем как малоинформативные
TOP_K = 8                    # сколько чанков подаём в контекст LLM
EMBED_BATCH = 100            # размер батча при эмбеддинге

# --- Чаты для прототипа ---
# username нужен для построения ссылок t.me/<username>/<msg_id>
CHATS = [
    {"username": "helpgeorgia", "chat_id": -1001452236047, "title": "Взаимопомощь. Грузия"},
    {"username": "ipgeorgiachat", "chat_id": -1001670908431, "title": "ИП/Бизнес Грузия"},
    # Раскомментируй для 2-го чата на этапе масштабирования:
    # {"username": "paravaingeorgia", "chat_id": -1001512786455, "title": "Получение водительских прав в Грузии"},
    # {"username": "mygeorgia_chat", "chat_id": -1001486751358, "title": "ГРУЗИЯ ЧАТ"},
    # {"username": "tbilisi_girl", "chat_id": -1001549075106, "title": "Женский чат Тбилиси"},
    # {"username": "georgia_it", "chat_id": -1001688709586, "title": "Грузия IT чат"},
    # {"username": "gruzia_medicina", "chat_id": -1001781403833, "title": "Грузия медицина"},
    # {"username": "georgia_woman", "chat_id": -1001276829180, "title": "Тбилиси женский чат"},
]
