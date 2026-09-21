"""Quick usage report from data/bot_log.jsonl (see src/bot.py::_log_interaction) —
who's asking the bot how much.

Run:  uv run python -m src.bot_stats
"""
from __future__ import annotations

import json
from collections import Counter

import config


def main() -> None:
    path = config.DATA_DIR / "bot_log.jsonl"
    if not path.exists():
        print(f"{path} not found — no interactions logged yet")
        return

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    by_user = Counter((r["user_id"], r.get("username")) for r in rows)

    print(f"=== {len(rows)} вопросов всего, {len(by_user)} пользователей ===\n")
    for (user_id, username), count in by_user.most_common():
        label = f"@{username}" if username else str(user_id)
        print(f"  {count:4}  {label}")


if __name__ == "__main__":
    main()
