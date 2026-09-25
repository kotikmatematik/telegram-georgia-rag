"""Telegram bot: ask a question -> the bot answers from the Georgia chats.

Run:  uv run python -m src.bot
Requires BOT_TOKEN in .env (get it from @BotFather).
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from datetime import date, datetime, timezone

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    LabeledPrice,
    LinkPreviewOptions,
    Message,
    PreCheckoutQuery,
)

import config
from src.rag import _CITY_COUNTS, _CITY_MIN_COUNT, _CITY_OPTIONS, answer, classify_guest, to_telegram_html
from src.store import transcribe_audio

logging.basicConfig(level=logging.INFO)
dp = Dispatcher()

# Simple per-user daily cap (config.BOT_MAX_REQUESTS_PER_DAY) so no single
# account can run up the Azure bill — persisted to a small JSON file (not
# just in memory) so a bot restart/redeploy doesn't quietly reset everyone's
# count. Keyed "user_id:date" and pruned to today on every write, so the
# file never grows past one day of activity.
_USAGE_PATH = config.DATA_DIR / "bot_usage.json"

# Who has supported the bot (see /support, /grant below) — {user_id (str):
# granted_at ISO timestamp}. A dict keyed by the grant date rather than a
# plain set/list, even though nothing reads the date yet: whether supporter
# status should EXPIRE (e.g. after 30 days) is still an open question, and
# storing the date now means answering that later is just a new condition in
# _is_supporter, not a data-file migration.
_SUPPORTERS_PATH = config.DATA_DIR / "bot_supporters.json"


def _load_supporters() -> dict[str, str]:
    try:
        return json.loads(_SUPPORTERS_PATH.read_text()) if _SUPPORTERS_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _is_supporter(user_id: int) -> bool:
    return str(user_id) in _load_supporters()


async def _grant_supporter(bot: Bot, user_id: int, *, source: str = "") -> None:
    """The one place that actually makes a user_id a supporter — called from
    BOTH payment paths (successful_payment for Stars, /grant for a manually
    verified bank transfer) so they can never drift apart. `source` is a
    short human-readable note for the owner notification below (e.g. "50⭐
    через Stars", "вручную через /grant") — purely cosmetic, doesn't affect
    the grant itself."""
    supporters = _load_supporters()
    supporters[str(user_id)] = datetime.now(timezone.utc).isoformat()
    _SUPPORTERS_PATH.write_text(json.dumps(supporters, ensure_ascii=False))
    try:
        await bot.send_message(
            user_id,
            "Спасибо за поддержку! 🙏 Теперь доступны голосовые сообщения и "
            f"лимит до {config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY} вопросов в день.",
        )
    except Exception:
        # Best-effort — a failed notification shouldn't undo the grant itself
        # (already written to disk above).
        logging.exception("failed to notify %s about supporter status", user_id)
    # Owner notification — the Stars path is otherwise silent to her (only
    # the payer gets told); the bank-transfer path she already knows about
    # (she's the one running /grant), but notifying both paths the same way
    # keeps this one place the single source of truth for "who paid, when".
    note = f" ({source})" if source else ""
    for owner_id in config.BOT_UNLIMITED_USER_IDS:
        try:
            await bot.send_message(owner_id, f"💰 Новый supporter: {user_id}{note}")
        except Exception:
            logging.exception("failed to notify owner %s about new supporter %s", owner_id, user_id)


def _limit_reached(user_id: int) -> bool:
    """Read-only peek at the same cap _check_and_count enforces — for guest
    mode, which has to know BEFORE spending a classifier call."""
    if user_id in config.BOT_UNLIMITED_USER_IDS:
        return False
    limit = config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY if _is_supporter(user_id) else config.BOT_MAX_REQUESTS_PER_DAY
    try:
        usage = json.loads(_USAGE_PATH.read_text()) if _USAGE_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        usage = {}
    return usage.get(f"{user_id}:{date.today().isoformat()}", 0) >= limit


def _check_and_count(user_id: int) -> bool:
    """True = allowed (and counted); False = today's cap already hit.
    Three tiers: config.BOT_UNLIMITED_USER_IDS bypasses the cap entirely (not
    even counted here — their usage still shows up in the interaction log
    below); a supporter (see _is_supporter) gets
    config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY; everyone else gets
    config.BOT_MAX_REQUESTS_PER_DAY."""
    if user_id in config.BOT_UNLIMITED_USER_IDS:
        return True
    limit = config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY if _is_supporter(user_id) else config.BOT_MAX_REQUESTS_PER_DAY
    today = date.today().isoformat()
    try:
        usage = json.loads(_USAGE_PATH.read_text()) if _USAGE_PATH.exists() else {}
    except (json.JSONDecodeError, OSError):
        usage = {}
    key = f"{user_id}:{today}"
    if usage.get(key, 0) >= limit:
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
    history: list[dict], extra: dict | None = None,
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
        **(extra or {}),
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

# Which cities to offer — _CITY_OPTIONS/_CITY_COUNTS/_CITY_MIN_COUNT now
# live in src/rag.py (shared with answer()'s own city injection into the
# embedding query — see there for why). Just the keyboard-only knob here:
# how many cities the /city button grid shows at once — _CITY_OPTIONS itself
# can be longer; the rest are still valid /city <name> targets, just not
# worth a dedicated button.
_CITY_KEYBOARD_SIZE = 8


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
    "• Где найти мастера по ремонту в Тбилиси?\n\n"
    f"Бесплатно: {config.BOT_MAX_REQUESTS_PER_DAY} вопросов в день. Голосовые "
    "и больший лимит — для тех, кто поддержал бота, см. /support.\n\n"
    "Есть предложение или что-то не так с ответом — пиши /feedback."
)

WELCOME = f"{GREETING}\n\n{EXAMPLES}"  # returning user, no city question involved


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    if (message.text or "").split(maxsplit=1)[1:] == ["support"]:
        # "Поддержать бота" button under a guest-mode limit notice — go
        # straight to /support, and don't wipe the history of someone who
        # may be mid-dialog in the DM already.
        await on_support_command(message)
        return
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


@dp.message(Command("id"))
async def on_id_command(message: Message) -> None:
    # Plain, no formatting — meant to be copy-pasted along with a bank
    # transfer receipt so Aleksandra knows which user_id to /grant.
    await message.answer(str(message.from_user.id))


@dp.message(Command("feedback"))
async def on_feedback_command(message: Message) -> None:
    arg = (message.text or "").split(maxsplit=1)
    text = arg[1].strip() if len(arg) > 1 else ""
    if not text:
        await message.answer(
            "Использование: /feedback <текст> — напиши прямо в этом же "
            "сообщении, что понравилось, не понравилось, или чего не хватает."
        )
        return
    user = message.from_user
    handle = f"@{user.username}" if user.username else f"id {user.id}"
    where = "" if message.chat.type == "private" else f" (чат «{message.chat.title}»)"
    for owner_id in config.BOT_UNLIMITED_USER_IDS:
        try:
            await message.bot.send_message(owner_id, f"📝 Фидбэк от {handle}{where}:\n\n{text}")
        except Exception:
            logging.exception("failed to relay feedback from %s to owner %s", user.id, owner_id)
    await message.answer("Спасибо, передала! 🙏")


_SUPPORT_TEXT = (
    "Бот бесплатный для пользователей, но не бесплатный для меня — каждый "
    "ответ и периодический пересбор знаний из чатов стоят денег в API. Всё, "
    "что присылают сюда, идёт только на то, чтобы бот продолжал жить, не на "
    "заработок.\n\n"
    f"Бесплатно: {config.BOT_MAX_REQUESTS_PER_DAY} вопросов в день. "
    f"Поддержавшим: голосовые + до {config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY} "
    "вопросов в день.\n\n"
    "Сумма символическая (от ~$1) — два способа на выбор:\n\n"
    "⭐ <b>Звёздами Telegram</b> — кнопки ниже, зачисляется сразу автоматически.\n\n"
    "🏦 <b>Переводом на счёт</b>:\n"
    f"{config.SUPPORT_BANK_INFO}\n"
    "После перевода:\n"
    "1. Напиши мне лично (@elder_flower) чек об оплате.\n"
    "2. Отправь боту команду /id — он ответит числом (твой id в Telegram).\n"
    "3. Перешли это число тоже мне — по нему я отмечу тебя в боте вручную."
)


def _support_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"Поддержать — {n}⭐", callback_data=f"support:{n}")]
            for n in config.STARS_SUPPORT_PRICES
        ]
    )


@dp.message(Command("support"))
async def on_support_command(message: Message) -> None:
    # HTML so the IBAN in config.SUPPORT_BANK_INFO renders as monospace
    # (<code>) — that's what makes it tap-to-copy in Telegram clients.
    await message.answer(_SUPPORT_TEXT, reply_markup=_support_keyboard(), parse_mode="HTML")


@dp.callback_query(F.data.startswith("support:"))
async def on_support_callback(callback: CallbackQuery) -> None:
    stars = int(callback.data.split(":", 1)[1])
    await callback.answer()
    await callback.bot.send_invoice(
        chat_id=callback.from_user.id,
        title="Поддержать бота",
        description="Символическая поддержка — покрывает API и периодический пересбор знаний.",
        payload="support",
        currency="XTR",
        prices=[LabeledPrice(label="Поддержка", amount=stars)],
    )


@dp.pre_checkout_query()
async def on_pre_checkout_query(query: PreCheckoutQuery) -> None:
    # Must be answered within 10s or Telegram cancels the payment — nothing
    # to validate here (payload is always "support"), just confirm.
    await query.answer(ok=True)


@dp.message(F.successful_payment)
async def on_successful_payment(message: Message) -> None:
    stars = message.successful_payment.total_amount
    await _grant_supporter(message.bot, message.from_user.id, source=f"{stars}⭐ через Stars")


@dp.message(Command("grant"))
async def on_grant_command(message: Message) -> None:
    # The manual side of /support — for a bank transfer Aleksandra verified
    # herself. Only she can call this (same exemption set as the rate limit).
    if message.from_user.id not in config.BOT_UNLIMITED_USER_IDS:
        return
    arg = (message.text or "").split(maxsplit=1)
    target = arg[1].strip() if len(arg) > 1 else ""
    if not target:
        await message.answer("Использование: /grant <user_id или @username>")
        return
    if target.lstrip("@").isdigit():
        user_id = int(target.lstrip("@"))
    else:
        # @username path — Bot API can resolve it directly, no need to make
        # the payer run /id themselves. Not 100% reliable (depends on that
        # person's privacy settings), so fall back to asking for /id instead
        # of failing silently.
        try:
            chat = await message.bot.get_chat(target if target.startswith("@") else f"@{target}")
            user_id = chat.id
        except Exception:
            await message.answer(
                f"Не смогла найти {target} по юзернейму (могло не получиться из-за его "
                "настроек приватности) — попроси прислать /id и вызови /grant с числом."
            )
            return
    await _grant_supporter(message.bot, user_id, source="вручную через /grant")
    await message.answer(f"Готово — {user_id} теперь supporter.")


_LIMIT_REACHED_TEXT = (
    f"На сегодня бесплатные вопросы закончились (лимит {config.BOT_MAX_REQUESTS_PER_DAY}/день) "
    "— каждый ответ сжигает токены, а токены стоят денег 💀\n\n"
    "Возвращайся завтра — лимит обновится. Или поддержи бота (/support) — "
    f"получишь лимит {config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY}/день и голосовые."
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


@dp.message(F.voice, F.chat.type == "private")
async def on_voice(message: Message) -> None:
    user_id = message.from_user.id
    if not (_is_supporter(user_id) or user_id in config.BOT_UNLIMITED_USER_IDS):
        # Gated before transcription — no point spending Groq quota on a
        # voice message we're not going to answer anyway.
        await message.answer(
            "🎤 Голосовые — для тех, кто поддержал бота (см. /support). "
            f"Текстом вопросы по-прежнему бесплатны ({config.BOT_MAX_REQUESTS_PER_DAY}/день)."
        )
        return
    if not _check_and_count(user_id):
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


# Guest mode (https://core.telegram.org/bots/features#guest-bots): someone
# mentions @georgia_insider_bot in ANY chat, or replies to one of its
# messages there, and the bot gets a guest_message update and may answer
# ONCE via answerGuestQuery — without being a member and without seeing the
# chat's history. Replaced inline mode: inline inserts content the USER
# sends, guest mode is the bot answering as itself, which is what a Q&A
# assistant actually is.
#
# Not every guest update deserves an answer — a reply to our answer is just
# as likely "спасибо" or people arguing among themselves under it, and a
# mention can be someone recommending the bot. classify_guest decides
# (request / thanks / chatter / mention) and only a request gets an answer
# and counts against the caller's daily limit. Answering isn't required.
#
# Flow for a request: placeholder first ("🔎 Ищу ответ…" — the guest query
# deadline is undocumented and RAG can take a while), then edit it in place
# via the inline_message_id answerGuestQuery returns. (Inline mode's version
# of this pattern failed because chosen_inline_result never arrived; here the
# id comes back synchronously from our own call.)
_GUEST_PLACEHOLDER = "🔎 Ищу ответ…"
_GUEST_HINT = (
    "Я отвечаю на практические вопросы о жизни в Грузии по опыту людей из "
    "чатов. Напиши вопрос после упоминания бота — или ответь упоминанием на "
    "сообщение с вопросом."
)
_GUEST_FAILED = "Не получилось ответить — попробуй ещё раз чуть позже."
# Telegram's limit is 4096 chars of text AFTER HTML parsing (and in UTF-16
# units); comparing the raw HTML length against a lower cap is conservative
# on both counts.
_GUEST_MAX_HTML = 3900
_NO_WORDS_RX = re.compile(r"^\W*$")  # emoji / punctuation only


def _guest_keyboard(bot_username: str) -> InlineKeyboardMarkup:
    # Funnel into the DM (history, /city, voice) — and an inline message with
    # a keyboard is the safe bet for being editable later. Plain link, NOT a
    # ?start= deep link: for someone who already uses the bot, clients send
    # /start by themselves on a deep link, and on_start wipes their DM
    # history mid-dialog and re-sends the welcome. A new user still gets
    # Telegram's own Start button either way.
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(
        text="Спросить в боте", url=f"https://t.me/{bot_username}",
    )]])


def _guest_article(text: str, keyboard: InlineKeyboardMarkup, *, parse_mode: str | None = None) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id="0",
        title="Ответ",
        input_message_content=InputTextMessageContent(
            message_text=text, parse_mode=parse_mode,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        ),
        reply_markup=keyboard,
    )


def _is_own_message(message: Message, bot_id: int) -> bool:
    # Which of these a guest-sent message carries isn't documented — check both.
    return bool(
        (message.from_user and message.from_user.id == bot_id)
        or (message.via_bot and message.via_bot.id == bot_id)
    )


def _strip_mention(text: str, bot_username: str) -> str:
    return re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE).strip()


async def _guest_voice_text(message: Message, caller_id: int) -> str:
    """Transcript of message's voice — same supporter gate as the DM, but a
    silent "" instead of a public nag in someone else's chat."""
    if not message.voice or not (_is_supporter(caller_id) or caller_id in config.BOT_UNLIMITED_USER_IDS):
        return ""
    try:
        buf = await message.bot.download(message.voice)
        return (await asyncio.to_thread(transcribe_audio, buf.read())) or ""
    except Exception:
        logging.exception("guest voice transcription failed")
        return ""


def _guest_answer_html(question: str, answer_text: str) -> str:
    """The question goes on top: the group sees what's being answered, and a
    follow-up reply to this message hands classify_guest both the question
    and the answer (the only "history" guest mode has)."""
    header = f"❓ <i>{html.escape(question)}</i>\n\n"
    body = to_telegram_html(answer_text)
    while len(header) + len(body) > _GUEST_MAX_HTML and len(answer_text) > 200:
        answer_text = answer_text[: int(len(answer_text) * 0.8)].rsplit("\n", 1)[0]
        body = to_telegram_html(answer_text) + "\n\n…продолжение — спроси в боте"
    return header + body


def _guest_limit_keyboard(bot_username: str) -> InlineKeyboardMarkup:
    # Deep link on purpose here (unlike _guest_keyboard): on_start routes
    # "support" straight to the /support screen without touching history.
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(
        text="Поддержать бота", url=f"https://t.me/{bot_username}?start=support",
    )]])


def _guest_limit_html(message: Message) -> str:
    """Public, in someone else's chat — so it names WHOSE limit ran out (not
    the chat's, not the bot's) and stays friendly: no token-cost lecture, no
    /commands that don't work there."""
    if message.sender_chat:
        who = html.escape(message.sender_chat.title or "этот чат")
    else:
        u = message.from_user
        who = f"@{u.username}" if u.username else f'<a href="tg://user?id={u.id}">{html.escape(u.first_name)}</a>'
    if message.from_user and _is_supporter(message.from_user.id):
        return f"{who}, на сегодня твои вопросы закончились — завтра лимит обновится 🙌"
    return (
        f"{who}, на сегодня твои бесплатные вопросы закончились — завтра лимит "
        "обновится 🙌\n\n"
        "Если бот тебе пригодился, его можно поддержать: так он продолжит "
        f"жить, а у тебя будет до {config.BOT_SUPPORTER_MAX_REQUESTS_PER_DAY} "
        "вопросов в день и голосовые."
    )


@dp.guest_message()
async def on_guest_message(message: Message) -> None:
    # Raw dump while guest-mode behavior is still being verified live (what
    # a reply to our own guest message looks like, etc.) — remove after.
    logging.info("guest_message: %s", message.model_dump_json(exclude_none=True))
    bot = message.bot
    me = await bot.me()
    if message.sender_chat is None and message.from_user and message.from_user.is_bot:
        return  # another bot — no bot-to-bot ping-pong on our API bill
    # Anonymous admin / posting as a channel: from_user is GroupAnonymousBot,
    # the real identity is sender_chat.
    caller_id = message.sender_chat.id if message.sender_chat else message.from_user.id
    caller_username = message.from_user.username if message.from_user else None
    if _limit_reached(caller_id):
        # Before anything that costs money (transcription, classifier) — and
        # answered whatever the message was, even a "спасибо".
        await bot.answer_guest_query(
            guest_query_id=message.guest_query_id,
            result=_guest_article(_guest_limit_html(message), _guest_limit_keyboard(me.username), parse_mode="HTML"),
        )
        return

    reply = message.reply_to_message
    replied_is_bot = reply is not None and _is_own_message(reply, me.id)
    text = _strip_mention(message.text or message.caption or "", me.username)
    if not text:
        text = await _guest_voice_text(message, caller_id)
    if _NO_WORDS_RX.match(text):
        text = ""  # 👍 / "!!!" — nothing to act on by itself
    replied_text = ""
    if reply is not None:
        replied_text = reply.text or reply.caption or ""
        if not replied_text and not replied_is_bot:
            replied_text = await _guest_voice_text(reply, caller_id)

    if not text and not replied_text:
        if reply is None:
            # Bare "@bot": they clearly want the bot, just didn't say what.
            await bot.answer_guest_query(
                guest_query_id=message.guest_query_id,
                result=_guest_article(_GUEST_HINT, _guest_keyboard(me.username)),
            )
        return  # sticker / emoji / unsupported voice reply — stay quiet
    if replied_is_bot and not text:
        return  # emoji-only / sticker reply to our own answer

    verdict = await asyncio.to_thread(classify_guest, text, replied_text or None, replied_is_bot)
    extra = {"source": "guest", "chat_type": message.chat.type, "label": verdict["label"], "raw_text": text}
    if verdict["label"] != "request":
        _log_interaction(caller_id, caller_username, text, "", [], extra)
        return

    query = verdict["query"]
    keyboard = _guest_keyboard(me.username)
    if not _check_and_count(caller_id):  # hit the cap in a parallel request since the peek
        await bot.answer_guest_query(
            guest_query_id=message.guest_query_id,
            result=_guest_article(_guest_limit_html(message), _guest_limit_keyboard(me.username), parse_mode="HTML"),
        )
        return

    sent = await bot.answer_guest_query(
        guest_query_id=message.guest_query_id, result=_guest_article(_GUEST_PLACEHOLDER, keyboard),
    )
    try:
        result = await asyncio.to_thread(answer, query, user_city=_get_city(caller_id))
        answer_text = result["answer"] or "Не удалось сформировать ответ."
        final_html = _guest_answer_html(query, answer_text)
    except Exception:
        logging.exception("guest answer failed")
        answer_text, final_html = "", _GUEST_FAILED
    _log_interaction(caller_id, caller_username, query, answer_text, [], extra)
    try:
        await bot.edit_message_text(
            text=final_html, inline_message_id=sent.inline_message_id, parse_mode="HTML",
            link_preview_options=LinkPreviewOptions(is_disabled=True), reply_markup=keyboard,
        )
    except Exception:
        logging.exception("failed to edit guest placeholder %s", sent.inline_message_id)


# Private only: groups are guest mode's job (joining groups is disabled in
# BotFather, but a bot added earlier would still get mentions/replies here as
# plain messages — mention not stripped, one history shared by the group).
@dp.message(F.chat.type == "private")
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
