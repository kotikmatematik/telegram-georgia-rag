#!/bin/bash
# Called by fail2ban (see /etc/fail2ban/action.d/telegram-notify.conf) every
# time it bans an IP — one Telegram message per ban, so an actual attack
# (brute force, scanning) is visible without having to go check
# `fail2ban-client status` by hand.
set -euo pipefail

ENV_FILE="/opt/telegram-georgia-rag/.env"
BOT_TOKEN=$(grep -E '^BOT_TOKEN=' "$ENV_FILE" | cut -d= -f2-)
CHAT_ID=273465125  # Aleksandra (@elder_flower)

JAIL="$1"
IP="$2"
FAILURES="${3:-?}"

TEXT="🚫 fail2ban забанил $IP в jail '$JAIL' ($FAILURES неудачных попыток)"

curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d chat_id="$CHAT_ID" \
    --data-urlencode "text=$TEXT" > /dev/null
