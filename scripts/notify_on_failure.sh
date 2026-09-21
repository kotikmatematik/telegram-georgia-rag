#!/bin/bash
# Called by systemd (georgia-bot.service's ExecStopPost) on EVERY stop of the
# bot process — clean or not. Only notifies on a non-clean stop ($SERVICE_RESULT
# set by systemd, e.g. "exit-code"/"signal"/"timeout") — a normal
# `systemctl restart` during a redeploy has $SERVICE_RESULT=success and stays
# silent. Fires on each individual crash even though Restart=always brings
# the service straight back up — ExecStopPost runs on every stop regardless
# of the subsequent restart, unlike OnFailure= (which only fires once the
# unit gives up entirely after exhausting its restart-rate limit).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${SERVICE_RESULT:-success}" = "success" ]; then
    exit 0
fi

BOT_TOKEN=$(grep -E '^BOT_TOKEN=' .env | cut -d= -f2-)
CHAT_ID=273465125  # Aleksandra (@elder_flower) — same id as config.BOT_UNLIMITED_USER_IDS

TEXT="⚠️ georgia-bot упал (result=${SERVICE_RESULT:-?}, exit_code=${EXIT_CODE:-?}, exit_status=${EXIT_STATUS:-?}). systemd перезапускает автоматически."

curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d chat_id="$CHAT_ID" \
    --data-urlencode "text=$TEXT" > /dev/null
