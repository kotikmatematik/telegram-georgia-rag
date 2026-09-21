"""Telegram bot: ask a question -> the bot answers from the Georgia chats.

Run:  uv run python -m src.bot
Requires BOT_TOKEN in .env (get it from @BotFather).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date, datetime, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)

import config
from src.rag import answer, to_telegram_html
from src.store import transcribe_audio

logging.basicConfig(level=logging.INFO)
dp = Dispatcher()

# Simple per-user daily cap (config.BOT_MAX_REQUESTS_PER_DAY) so no single
# account can run up the Azure bill — persisted to a small JSON file (not
# just in memory) so a bot restart/redeploy doesn't quietly reset everyone's
# count. Keyed "user_id:date" and pruned to today on every write, so the
# file never grows past one day of activity.
_USAGE_PATH = config.DATA_DIR / "bot_usage.json"


def _check_and_count(user_id: int) -> bool:
    """True = allowed (and counted); False = today's cap already hit.
    config.BOT_UNLIMITED_USER_IDS bypasses the cap entirely (not even counted
    here — their usage still shows up in the interaction log below)."""
    if user_id in config.BOT_UNLIMITED_USER_IDS:
        return True
    today = date.today().isoformat()
    try:
        usage = json.loads(_USAGE_PATH.read_text()) if _USAGE_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        usage = {}
    key = f"{user_id}:{today}"
    if usage.get(key, 0) >= config.BOT_MAX_REQUESTS_PER_DAY:
        return False
    usage = {k: v for k, v in usage.items() if k.endswith(today)}  # drop old days
    usage[key] = usage.get(key, 0) + 1
    _USAGE_PATH.write_text(json.dumps(usage))
    return True


# One line per real interaction: who asked what, and what the bot actually
# said (already-substituted text — what the user saw, not the raw model
# output) — this is both "who's using it how much" analytics and, later, a
# ready-made pool of REAL questions (with real answers to judge) to grow
# eval/golden_queries.jsonl from, instead of only ones we thought up ourselves.
_LOG_PATH = config.DATA_DIR / "bot_log.jsonl"


def _log_interaction(
    user_id: int, username: str | None, question: str, answer_text: str,
    history: list[dict],
) -> None:
    """history is the conversation BEFORE this question (same list passed
    into src.rag.answer) — attached as-is (no extra API calls, it's already
    sitting on disk in bot_history.json) so a later reader can tell a
    context-dependent question like "тбилиси" apart from a standalone one
    without having to reconstruct the surrounding chat by hand."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "username": username,
        "history": history,
        "question": question,
        "answer": answer_text,
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Full per-chat conversation history, persisted to disk (like _USAGE_PATH
# above) so it survives a bot restart/redeploy — a real multi-question dialog
# can span more turns than any fixed cap, and losing it mid-conversation on
# every redeploy was worse than the disk I/O cost of keeping it. Only used
# to resolve a short reply to the bot's own clarifying question ("рабочая"
# answering "туристическая или рабочая?") into a standalone query before
# retrieval (see src.rag._rewrite_query, which actually reads this).
_HISTORY_PATH = config.DATA_DIR / "bot_history.json"
# The stored dialog itself is NOT capped — but only the last N messages of it
# are ever read back into a request (see _get_history below). Without a cap,
# _rewrite_query's prompt (and its cost/latency) would grow with the total
# length of the conversation instead of staying roughly constant; 20 messages
# (~10 user/assistant pairs) is generous room for a composite question built
# from several earlier answers while keeping that cost bounded.
_HISTORY_CONTEXT_MESSAGES = 20


def _load_histories() -> dict[str, list[dict]]:
    try:
        return json.loads(_HISTORY_PATH.read_text()) if _HISTORY_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_histories(histories: dict[str, list[dict]]) -> None:
    _HISTORY_PATH.write_text(json.dumps(histories, ensure_ascii=False))


def _append_history(chat_id: int, question: str, answer_text: str) -> None:
    histories = _load_histories()
    h = histories.setdefault(str(chat_id), [])
    h.append({"role": "user", "content": question})
    h.append({"role": "assistant", "content": answer_text})
    _save_histories(histories)


def _get_history(chat_id: int) -> list[dict]:
    """Last _HISTORY_CONTEXT_MESSAGES of the full persisted dialog — see the
    module comment above for why the stored dialog itself stays uncapped."""
    return _load_histories().get(str(chat_id), [])[-_HISTORY_CONTEXT_MESSAGES:]


def _clear_history(chat_id: int) -> None:
    histories = _load_histories()
    histories.pop(str(chat_id), None)
    _save_histories(histories)

WELCOME = (
    "Привет! Я отвечаю на вопросы о жизни в Грузии на основе переписок из "
    "тематических Telegram-чатов.\n\n"
    "Просто напиши вопрос, например:\n"
    "• Какие документы нужны для открытия ИП?\n"
    "• Как получить водительские права?\n"
    "• В каком районе Тбилиси лучше снять квартиру?"
)


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    _clear_history(message.chat.id)  # explicit fresh start clears any old context
    await message.answer(WELCOME)


_LIMIT_REACHED_TEXT = (
    f"На сегодня бесплатные вопросы закончились (лимит {config.BOT_MAX_REQUESTS_PER_DAY}/день) "
    "— каждый ответ сжигает токены, а токены стоят денег 💀\n\n"
    "Возвращайся завтра — лимит обновится. Или напиши @elder_flower, "
    "если хочешь повысить лимит за $1."
)


async def _answer_query(message: Message, query: str) -> None:
    """Shared tail of both on_question and on_voice, once we have a plain-text
    query (typed, or transcribed from voice) and already know the rate limit
    allows it."""
    await message.chat.do("typing")
    history = _get_history(message.chat.id)
    # answer() is synchronous (blocking OpenAI calls), so run it in a thread
    result = await asyncio.to_thread(answer, query, history=history)
    text = result["answer"] or "Не удалось сформировать ответ."
    _append_history(message.chat.id, query, text)
    _log_interaction(message.from_user.id, message.from_user.username, query, text, history)
    await message.answer(to_telegram_html(text), parse_mode="HTML", disable_web_page_preview=True)


@dp.message(F.voice)
async def on_voice(message: Message) -> None:
    if not _check_and_count(message.from_user.id):
        await message.answer(_LIMIT_REACHED_TEXT)
        return
    await message.chat.do("typing")
    try:
        buf = await message.bot.download(message.voice)
        query = await asyncio.to_thread(transcribe_audio, buf.read())
    except Exception:
        logging.exception("voice transcription failed")
        await message.answer(
            "Не получилось распознать голосовое сообщение — попробуй ещё раз или напиши текстом."
        )
        return
    if not query:
        await message.answer(
            "Не удалось разобрать текст в голосовом сообщении — попробуй ещё раз или напиши текстом."
        )
        return
    # Echo the transcript back first — speech recognition isn't perfect, and
    # the user should be able to tell "неправильно услышал" from "не нашла
    # ответа" without having to guess which one happened.
    await message.answer(f"🎤 {query}")
    await _answer_query(message, query)


# Inline mode: typing "@georgia_insider_bot вопрос" in ANY chat (no need to
# add the bot there) shows one result with the actual answer already in it.
# See https://core.telegram.org/bots/inline.
#
# Originally tried the "insert a placeholder, fill it in later" pattern
# (answer instantly, do the real RAG call in on_chosen_inline_result once the
# user picks the result, then edit the message via inline_message_id) — the
# standard approach for slow inline bots. Verified end-to-end that Telegram
# never actually delivered chosen_inline_result to this bot at all (tested
# directly at the MTProto level, bypassing any client-side quirk, with
# /setinlinefeedback = Enabled on @BotFather) — a known reliability problem
# with that update in practice, not something fixable on the bot's end.
#
# So instead: no placeholder, no edit. The RAG call happens directly in
# on_inline_query, with a short debounce so a fast typer's earlier
# keystrokes don't each trigger a full generation — only the LAST query for
# a given user (the one no longer superseded after the debounce delay)
# actually calls answer(). Costs slightly more than the placeholder pattern
# would have (a debounce-survived call per pause in typing, not just per
# final selection) but that pattern didn't work at all.
_INLINE_DEBOUNCE_SECONDS = 1.2
_inline_seq: dict[int, int] = {}  # user_id -> sequence number of its latest query


@dp.inline_query()
async def on_inline_query(query: InlineQuery) -> None:
    text = query.query.strip()
    if not text:
        await query.answer([], cache_time=1, is_personal=True)
        return
    user_id = query.from_user.id
    seq = _inline_seq.get(user_id, 0) + 1
    _inline_seq[user_id] = seq

    await asyncio.sleep(_INLINE_DEBOUNCE_SECONDS)
    if _inline_seq.get(user_id) != seq:
        return  # a newer keystroke for this user already superseded this query

    if not _check_and_count(user_id):
        result = InlineQueryResultArticle(
            id="limit",
            title="Лимит вопросов на сегодня исчерпан",
            description=_LIMIT_REACHED_TEXT,
            input_message_content=InputTextMessageContent(message_text=_LIMIT_REACHED_TEXT),
        )
        await query.answer([result], cache_time=0, is_personal=True)
        return

    result_data = await asyncio.to_thread(answer, text)
    # No conversation history: unlike a normal chat, an inline insertion has
    # no stable "this chat" to persist a thread against (the same query text
    # can land in a different chat every time) — each inline question is
    # answered stateless, same as a first message with no history.
    answer_text = result_data["answer"] or "Не удалось сформировать ответ."
    _log_interaction(user_id, query.from_user.username, text, answer_text, [])
    html_text = to_telegram_html(answer_text)
    result = InlineQueryResultArticle(
        id=hashlib.sha1(text.encode("utf-8")).hexdigest(),
        title=f"Спросить: {text[:60]}",
        description=answer_text[:120],
        input_message_content=InputTextMessageContent(
            message_text=html_text, parse_mode="HTML", disable_web_page_preview=True,
        ),
    )
    await query.answer([result], cache_time=0, is_personal=True)


@dp.message()
async def on_question(message: Message) -> None:
    query = (message.text or "").strip()
    if not query:
        await message.answer("Напиши, пожалуйста, текстовый вопрос.")
        return
    if not _check_and_count(message.from_user.id):
        await message.answer(_LIMIT_REACHED_TEXT)
        return
    await _answer_query(message, query)


async def main() -> None:
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set in .env (get it from @BotFather)")
    bot = Bot(config.BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
