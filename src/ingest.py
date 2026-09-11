"""Fetch Telegram chat history into data/raw/<username>.jsonl via Telethon.

Incremental: each run appends only messages newer than the last one already on
disk (by id), so re-running is cheap and the ragged "oldest message has no
parent" edge is only crossed once — on the very first fetch.

First fetch goes back to config.ingest_fetch_floor_dt() (INGEST_SINCE minus the
parent-lookback tail), capped by config.INGEST_LIMIT.

Run:  uv run python -m src.ingest
On first run Telethon will ask for the confirmation code from Telegram.
"""
from __future__ import annotations

import asyncio
import json

from telethon import TelegramClient

import config


def _msg_to_record(msg, chat) -> dict | None:
    """Convert a Telethon message into a flat record. None means skip it."""
    text = (msg.message or "").strip()
    if not text:
        return None  # skip media without caption and service messages
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


def _last_msg_id(path) -> int:
    """Largest msg_id already saved for this chat, or 0 if the file is new."""
    if not path.exists():
        return 0
    last = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = max(last, json.loads(line)["msg_id"])
    return last


async def ingest_chat(client: TelegramClient, chat: dict) -> int:
    out_path = config.RAW_DIR / f"{chat['username']}.jsonl"
    since_id = _last_msg_id(out_path)
    floor_dt = config.ingest_fetch_floor_dt()

    # Incremental append if we already have history; fresh (write) otherwise.
    mode = "a" if since_id else "w"
    kwargs = {"limit": config.INGEST_LIMIT}
    if since_id:
        kwargs["min_id"] = since_id  # only messages strictly newer than this

    count = 0
    with out_path.open(mode, encoding="utf-8") as f:
        async for msg in client.iter_messages(chat["chat_id"], **kwargs):
            # iter_messages yields newest -> oldest; on the first fetch stop once
            # we pass the floor date (min_id already bounds the incremental case).
            if not since_id and floor_dt and msg.date and msg.date < floor_dt:
                break
            rec = _msg_to_record(msg, chat)
            if rec is None:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    how = f"appended {count} new" if since_id else f"fetched {count}"
    print(f"[ingest] {chat['username']}: {how} messages -> {out_path}")
    return count


async def main() -> None:
    if not (config.TELEGRAM_API_ID and config.TELEGRAM_API_HASH):
        raise SystemExit(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH are not set in .env "
            "(get them at https://my.telegram.org/apps)"
        )
    floor_dt = config.ingest_fetch_floor_dt()
    print(
        f"[ingest] cutoff INGEST_SINCE={config.INGEST_SINCE or '(none)'} "
        f"| fetch floor: {floor_dt.date() if floor_dt else '(none)'} "
        f"| cap: {config.INGEST_LIMIT}"
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
