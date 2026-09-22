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
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
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


# Per-user default city (src.rag.answer's user_city) — so a question that
# would otherwise need "в каком городе?" can just be answered directly,
# instead of the user having to type their city into every single question.
# Keyed by user_id (a preference of the PERSON, unlike history/rate-limit
# which are keyed by chat_id) — "" means "asked, explicitly skipped" (don't
# nag again on /start); key absent entirely means "never asked yet".
_CITY_PATH = config.DATA_DIR / "bot_user_city.json"

# Which cities to offer is computed from the ACTUAL knowledge base, not a
# hardcoded list — as more chats/threads get collected, the real spread of
# well-covered cities shifts, and a hardcoded list would silently go stale.
# A city needs at least this many tagged knowledge units to be offered at
# all — below that, "answer for this city by default" wouldn't have enough
# to actually work with most of the time.
_CITY_MIN_COUNT = 5
# The `city` field is LLM-tagged per knowledge unit and occasionally names a
# place mentioned only as a travel destination/origin from a Georgia-based
# thread ("дорога до Еревана из Тбилиси"), not a real city the BOT'S USER
# could be based in — excluded even if it clears the count threshold above.
_NON_GEORGIAN_CITY_NOISE = {"Ереван", "Москва", "Владикавказ", "Санкт-Петербург", "Стамбул"}
# How many cities the /city keyboard shows at once — _CITY_OPTIONS itself
# (below) can be longer; the rest are still valid /city <name> targets,
# just not worth a dedicated button.
_CITY_KEYBOARD_SIZE = 8


def _compute_city_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in config.KNOWLEDGE_DIR.glob("*.fixed.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                city = json.loads(line).get("city")
                if city:
                    counts[city] = counts.get(city, 0) + 1
    return counts


# Computed once at import (process start) — data/knowledge only actually
# changes via the weekly refresh job, which restarts this bot process
# anyway (see scripts/georgia-weekly.service), so a fresh count on every
# request would just re-read the same ~20k lines for no benefit.
_CITY_COUNTS = _compute_city_counts()
_CITY_OPTIONS = [
    c for c, n in sorted(_CITY_COUNTS.items(), key=lambda kv: -kv[1])
    if n >= _CITY_MIN_COUNT and c not in _NON_GEORGIAN_CITY_NOISE
]


def _load_cities() -> dict[str, str]:
    try:
        return json.loads(_CITY_PATH.read_text()) if _CITY_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cities(cities: dict[str, str]) -> None:
    _CITY_PATH.write_text(json.dumps(cities, ensure_ascii=False))


def _get_city(user_id: int) -> str | None:
    """Saved default city, or None if never set / explicitly skipped."""
    return _load_cities().get(str(user_id)) or None


def _has_chosen_city(user_id: int) -> bool:
    """Whether the city prompt already ran once (picked a city OR skipped) —
    distinct from _get_city being None, which is also true for "skipped"."""
    return str(user_id) in _load_cities()


def _set_city(user_id: int, city: str) -> None:
    cities = _load_cities()
    cities[str(user_id)] = city  # "" for an explicit skip
    _save_cities(cities)


def _city_keyboard() -> InlineKeyboardMarkup:
    shown = _CITY_OPTIONS[:_CITY_KEYBOARD_SIZE]  # already sorted by count, most-covered first
    rows = [
        [InlineKeyboardButton(text=c, callback_data=f"city:{c}") for c in shown[i : i + 2]]
        for i in range(0, len(shown), 2)
    ]
    rows.append([InlineKeyboardButton(text="Не важно / пропустить", callback_data="city:__skip__")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


GREETING = (
    "Привет! Я отвечаю на вопросы о жизни в Грузии на основе переписок из "
    "тематических Telegram-чатов."
)

CITY_PROMPT = (
    "К какому городу вы ближе? Тогда на вопросы вроде «где найти врача» "
    "смогу сразу отвечать по нему, не переспрашивая каждый раз.\n\n"
    "Своего города нет в списке — напишите «/city Название». Сменить "
    "выбор потом всегда можно той же командой /city."
)

EXAMPLES = (
    "Задай вопрос, например:\n"
    "• Какие документы нужны для открытия ИП?\n"
    "• Как получить водительские права?\n"
    "• Где найти мастера по ремонту в Тбилиси?"
)

WELCOME = f"{GREETING}\n\n{EXAMPLES}"  # returning user, no city question involved


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    _clear_history(message.chat.id)  # explicit fresh start clears any old context
    # A brand-new user: greeting + city question together first (greeting
    # someone before asking them something, not after) — the "here's how to
    # ask" examples (EXAMPLES) only follow once they've actually answered
    # the city question (see on_city_callback's is_onboarding branch). A
    # returning user (already has a city, even "skipped") just gets the
    # full WELCOME (greeting + examples) — no city question to repeat.
    if not _has_chosen_city(message.from_user.id):
        await message.answer(f"{GREETING}\n\n{CITY_PROMPT}", reply_markup=_city_keyboard())
    else:
        await message.answer(WELCOME)


@dp.message(Command("city"))
async def on_city_command(message: Message) -> None:
    arg = (message.text or "").split(maxsplit=1)
    custom = arg[1].strip() if len(arg) > 1 else ""
    if custom:
        # Same as on_city_callback: only true if /city is literally this
        # user's first-ever interaction (typed a custom city before ever
        # touching /start's picker) — WELCOME follows in that case too.
        is_onboarding = not _has_chosen_city(message.from_user.id)
        count = _CITY_COUNTS.get(custom, 0)
        alternatives = ", ".join(_CITY_OPTIONS[:5])
        if count == 0:
            # No knowledge at all is tagged with this city — setting it as
            # the default would never actually change anything (the [CITY]
            # mechanism has nothing to answer from), so it's functionally
            # the same as skipping. Say so plainly instead of pretending it
            # did something.
            _set_city(message.from_user.id, "")
            await message.answer(
                f"По городу «{custom}» знаний в базе пока нет — по умолчанию его не "
                f"поставить, буду переспрашивать город, если понадобится. Более полные "
                f"варианты: {alternatives} — выбрать: /city"
            )
        elif count < _CITY_MIN_COUNT:
            # Some knowledge exists, just not much — respect the choice
            # (still set it, it's their real city), but be honest that it
            # likely won't help most of the time.
            _set_city(message.from_user.id, custom)
            await message.answer(
                f"Поставила «{custom}» по умолчанию, но знаний про этот город в базе мало "
                f"({count}) — сработает не всегда. Более полные варианты: {alternatives} "
                f"— выбрать: /city"
            )
        else:
            _set_city(message.from_user.id, custom)
            await message.answer(f"Готово — город по умолчанию: {custom}. Сменить снова: /city")
        if is_onboarding:
            await message.answer(EXAMPLES)  # greeting already sent alongside CITY_PROMPT
        return
    current = _get_city(message.from_user.id)
    suffix = f"\n\nСейчас выбрано: {current}." if current else ""
    await message.answer(CITY_PROMPT + suffix, reply_markup=_city_keyboard())


@dp.callback_query(F.data.startswith("city:"))
async def on_city_callback(callback: CallbackQuery) -> None:
    # Checked BEFORE _set_city below: true only the very first time this
    # user ever answers the city question (via /start) — a later /city
    # change (they already have an entry either way) shouldn't re-send
    # WELCOME, only the initial onboarding should.
    is_onboarding = not _has_chosen_city(callback.from_user.id)
    value = callback.data.split(":", 1)[1]
    if value == "__skip__":
        _set_city(callback.from_user.id, "")
        text = "Хорошо, буду переспрашивать город, когда понадобится. Задать его позже: /city"
    else:
        _set_city(callback.from_user.id, value)
        text = f"Готово — город по умолчанию: {value}. Сменить: /city"
    await callback.message.edit_text(text)
    await callback.answer()
    if is_onboarding:
        await callback.message.answer(EXAMPLES)  # greeting already sent alongside CITY_PROMPT


_LIMIT_REACHED_TEXT = (
    f"На сегодня бесплатные вопросы закончились (лимит {config.BOT_MAX_REQUESTS_PER_DAY}/день) "
    "— каждый ответ сжигает токены, а токены стоят денег 💀\n\n"
    "Возвращайся завтра — лимит обновится. Или напиши @elder_flower, "
    "если хочешь повысить лимит."
)


async def _answer_query(message: Message, query: str) -> None:
    """Shared tail of both on_question and on_voice, once we have a plain-text
    query (typed, or transcribed from voice) and already know the rate limit
    allows it."""
    await message.chat.do("typing")
    history = _get_history(message.chat.id)
    user_city = _get_city(message.from_user.id)
    # answer() is synchronous (blocking OpenAI calls), so run it in a thread
    result = await asyncio.to_thread(answer, query, history=history, user_city=user_city)
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

    result_data = await asyncio.to_thread(answer, text, user_city=_get_city(user_id))
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
