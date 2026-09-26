"""Daily digest of bot usage, sent to config.BOT_UNLIMITED_USER_IDS via
Telegram — questions asked today, unique users, new supporters. Reads
data/bot_log.jsonl and data/bot_supporters.json directly (no bot process
involved, safe to run alongside the live bot).

Deployed as scripts/georgia-daily-stats.service + .timer (see those files),
same pattern as georgia-weekly.service.

Run (from repo root, so `import config` resolves):
  uv run python scripts/daily_stats.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone

import httpx

import config


def _load_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_digest() -> str:
    today = date.today().isoformat()
    rows = [
        r for r in _load_jsonl(config.DATA_DIR / "bot_log.jsonl")
        if r.get("ts", "").startswith(today)
    ]
    by_user = Counter(r["user_id"] for r in rows)
    usernames = {r["user_id"]: r.get("username") for r in rows}

    supporters = {}
    supporters_path = config.DATA_DIR / "bot_supporters.json"
    if supporters_path.exists():
        supporters = json.loads(supporters_path.read_text())
    new_supporters = [
        uid for uid, ts in supporters.items()
        if ts.startswith(today)
    ]

    lines = [f"📊 Статистика за {today}", ""]
    lines.append(f"Вопросов: {len(rows)}")
    lines.append(f"Уникальных пользователей: {len(by_user)}")
    if by_user:
        lines.append("")
        lines.append("Топ по активности:")
        for uid, count in by_user.most_common(10):
            uname = f"@{usernames[uid]}" if usernames.get(uid) else str(uid)
            lines.append(f"  {uname}: {count}")
    if new_supporters:
        lines.append("")
        lines.append(f"Новых supporter'ов сегодня: {len(new_supporters)} ({', '.join(new_supporters)})")
    lines.append("")
    lines.append(f"Всего supporter'ов: {len(supporters)}")
    return "\n".join(lines)


def main() -> None:
    text = build_digest()
    print(text, flush=True)
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set in .env")
    for owner_id in config.BOT_UNLIMITED_USER_IDS:
        resp = httpx.post(
            f"https://api.telegram.org/bot{config.BOT_TOKEN}/sendMessage",
            data={"chat_id": owner_id, "text": text},
            timeout=10,
        )
        resp.raise_for_status()


if __name__ == "__main__":
    main()
