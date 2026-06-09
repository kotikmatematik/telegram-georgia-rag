"""Очистка сообщений и сборка их в «чанки» по временным окнам.

Сообщения в чатах короткие и часто идут связками «вопрос -> ответы».
Группируем подряд идущие сообщения, пока разрыв во времени меньше
CHUNK_TIME_GAP_MIN и размер не превысил CHUNK_MAX_CHARS. Так в одном чанке
остаётся цельный мини-диалог, что повышает качество поиска.

Запуск:  uv run python -m src.preprocess
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config


def _load_raw(path: Path) -> list[dict]:
    msgs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                msgs.append(json.loads(line))
    # по возрастанию id => хронологический порядок
    msgs.sort(key=lambda m: m["msg_id"])
    return msgs


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _flush(buf: list[dict]) -> dict | None:
    """Собрать накопленные сообщения в один чанк."""
    if not buf:
        return None
    lines = []
    for m in buf:
        sender = m.get("sender") or "Аноним"
        lines.append(f"{sender}: {m['text']}")
    text = "\n".join(lines).strip()
    if len(text) < config.CHUNK_MIN_CHARS:
        return None
    first = buf[0]
    return {
        "id": f"{first['chat_username']}:{first['msg_id']}",
        "text": text,
        "chat_username": first["chat_username"],
        "chat_title": first["chat_title"],
        "first_msg_id": first["msg_id"],
        "last_msg_id": buf[-1]["msg_id"],
        "date": first.get("date") or "",
        "link": first["link"],
    }


def chunk_messages(msgs: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    buf: list[dict] = []
    buf_chars = 0
    prev_dt: datetime | None = None
    gap = config.CHUNK_TIME_GAP_MIN * 60

    for m in msgs:
        dt = _parse_dt(m.get("date"))
        too_old = (
            prev_dt is not None
            and dt is not None
            and (dt - prev_dt).total_seconds() > gap
        )
        too_big = buf_chars + len(m["text"]) > config.CHUNK_MAX_CHARS
        if buf and (too_old or too_big):
            ch = _flush(buf)
            if ch:
                chunks.append(ch)
            buf, buf_chars = [], 0
        buf.append(m)
        buf_chars += len(m["text"])
        prev_dt = dt

    ch = _flush(buf)
    if ch:
        chunks.append(ch)
    return chunks


def process_chat(username: str) -> int:
    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        print(f"[preprocess] пропуск {username}: нет {raw_path}")
        return 0
    msgs = _load_raw(raw_path)
    chunks = chunk_messages(msgs)
    out_path = config.CHUNKS_DIR / f"{username}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(
        f"[preprocess] {username}: {len(msgs)} сообщений -> "
        f"{len(chunks)} чанков -> {out_path}"
    )
    return len(chunks)


def main() -> None:
    for chat in config.CHATS:
        process_chat(chat["username"])


if __name__ == "__main__":
    main()
