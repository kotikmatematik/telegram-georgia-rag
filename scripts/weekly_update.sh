#!/bin/bash
# Recurring knowledge refresh: ingest new messages, incrementally (re)distill
# only threads active since the last run, re-embed what changed, sync the
# updated Chroma DB to the server, and restart the bot so it serves the new
# data. Scheduled via launchd (see the LaunchAgent plist installed alongside
# this script) — not meant to be run by hand except to test it.
set -euo pipefail
# launchd runs with a minimal PATH (no ~/.zshrc, no Homebrew) — spell it out
# explicitly so `uv`/rsync/ssh resolve the same way they do in a real shell.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."

LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/weekly_update_$(date +%Y%m%d_%H%M%S).log"

{
    echo "=== weekly update started $(date) ==="
    uv run python -m src.weekly_pipeline
    echo "=== syncing chroma_db to server ==="
    rsync -az chroma_db/ root@173.249.40.224:/opt/telegram-georgia-rag/chroma_db/
    echo "=== restarting bot ==="
    ssh root@173.249.40.224 "systemctl restart georgia-bot"
    echo "=== weekly update finished $(date) ==="
} >> "$LOG" 2>&1
