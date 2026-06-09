"""Выгрузка истории Telegram-чатов в data/raw/<username>.jsonl через Telethon.

Запуск:  uv run python -m src.ingest
При первом запуске Telethon попросит код подтверждения из Telegram.
"""
from __future__ import annotations

import asyncio
import json

from telethon import TelegramClient

import config


def _msg_to_record(msg, chat) -> dict | None:
    """Преобразовать сообщение Telethon в плоскую запись. None — если пропускаем."""
    text = (msg.message or "").strip()
    if not text:
        return None  # пропускаем медиа без подписи, сервисные сообщения
    sender = None
    if msg.sender is not None:
        sender = getattr(msg.sender, "first_name", None) or getattr(
            msg.sender, "title", None
        )
    return {
        "msg_id": msg.id,
        "date": msg.date.isoformat() if msg.date else None,
        "sender": sender,
        "sender_id": msg.sender_id,
        "text": text,
        "reply_to": msg.reply_to_msg_id,
        "chat_username": chat["username"],
        "chat_title": chat["title"],
        "link": f"https://t.me/{chat['username']}/{msg.id}",
    }


async def ingest_chat(client: TelegramClient, chat: dict) -> int:
    out_path = config.RAW_DIR / f"{chat['username']}.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        async for msg in client.iter_messages(
            chat["chat_id"], limit=config.INGEST_LIMIT
        ):
            rec = _msg_to_record(msg, chat)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"[ingest] {chat['username']}: сохранено {count} сообщений -> {out_path}")
    return count


async def main() -> None:
    if not (config.TELEGRAM_API_ID and config.TELEGRAM_API_HASH):
        raise SystemExit(
            "Не заданы TELEGRAM_API_ID / TELEGRAM_API_HASH в .env "
            "(получить на https://my.telegram.org/apps)"
        )
    client = TelegramClient(
        "georgia_ingest", int(config.TELEGRAM_API_ID), config.TELEGRAM_API_HASH
    )
    await client.start(phone=config.TELEGRAM_PHONE or None)
    try:
        for chat in config.CHATS:
            await ingest_chat(client, chat)
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
