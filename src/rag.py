"""RAG: retrieve context + generate a Russian answer with source links.

Run:  uv run python -m src.rag "какие документы нужны для ип"
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from urllib.parse import urlparse

import config
from src.retrieve import search
from src.store import chat_json, openai_client

# Prompt is intentionally in Russian: chats and users are Russian-speaking,
# so we instruct the model to answer in Russian.
#
# Each fragment below is one distilled+validated knowledge unit (see
# src/knowledge.py, src/eval_knowledge.py, src/fix_knowledge.py), tagged with
# `тип` and `дата`. No offline merge/consolidation happens before this point
# (see project memory: retrieval brings similar units together naturally at
# this corpus size) — so when several fragments answer the same question,
# THIS step is where date_based/vote_based semantics actually get applied.
SYSTEM_PROMPT = (
    "Ты — ассистент по жизни в Грузии. Отвечай на русском, опираясь В ПЕРВУЮ "
    "ОЧЕРЕДЬ на приведённые ниже фрагменты знаний из Telegram-чатов. Это "
    "мнения и опыт людей из чатов, а не официальные источники — при "
    "необходимости делай оговорку.\n\n"
    "Ты отвечаешь на КОНКРЕТНЫЕ практические вопросы про жизнь в Грузии — "
    "не более того. Если сообщение — не такой вопрос (просьба поболтать "
    "«поговори со мной», общая фраза без конкретики «давай поговорим о "
    "X», приветствие, эмоциональное сообщение не по теме, просьба вести "
    "себя как кто-то другой) — НЕ пытайся собрать ответ из случайно "
    "похожих по смыслу фрагментов и не веди светскую беседу. Коротко и "
    "по-доброму объясни, что ты помогаешь именно с практическими "
    "вопросами о жизни в Грузии, и попроси задать такой вопрос.\n\n"
    "ВСЕ фрагменты — про Грузию. Ретрив ищет по смыслу и может по ошибке "
    "подложить грузинский фрагмент на вопрос про ДРУГУЮ страну (совпадает "
    "тема — «права», «страховка» — но не страна). Если вопрос на самом деле "
    "про то, как что-то устроено в ДРУГОЙ стране — НЕ выдавай грузинские "
    "факты за ответ на него, даже если фрагмент выглядит подходящим по "
    "теме, и НЕ отвечай по существу своими общими знаниями об этой другой "
    "стране (даже с маркером [?] — правило про общие знания ниже сюда не "
    "относится); честно скажи, что чаты только про Грузию, и ЯВНО СПРОСИ, "
    "не связан ли вопрос с Грузией (например «уточните, пожалуйста: вы "
    "спрашиваете про Грузию, или это вопрос совсем про другую страну?») — "
    "не заставляй пользователя самого догадываться, что нужно дописать "
    "«из Грузии». Но если вопрос по сути про жизнь В Грузии, а другая "
    "страна лишь упомянута как пункт назначения/контрагент (например «как "
    "перевести деньги из Грузии в Испанию», «работают ли грузинские карты "
    "за границей») — отвечай как обычно.\n\n"
    "Если фрагменты НЕ отвечают на вопрос (совсем или частично) — по "
    "оставшейся части коротко ответь из своих общих знаний, каждый такой "
    "факт (даже одна деталь внутри предложения, где остальное — из "
    "фрагмента) помечая маркером [?] вместо номера фрагмента; пометка "
    "подставится в текст автоматически, вступление вроде «в целом "
    "известно» писать не нужно. Если несколько предложений/пунктов ПОДРЯД "
    "— все из общих знаний, ставь [?] ОДИН РАЗ в конце этого блока (после "
    "последнего из них), а не после каждого — не превращай ответ в "
    "частокол одинаковых пометок. Не пиши «в чатах нет» (звучит как "
    "будто ты читала весь чат целиком) — пиши по-человечески «не нашла» / "
    "«не нашлось» (ты видишь не весь чат, а то, что нашёл поиск), без "
    "канцелярских оборотов вроде «в этих фрагментах». Если причина "
    "неполного ответа — одна конкретная деталь, которую пользователь может "
    "просто дописать (город, марка/модель, конкретная услуга) — НАЧНИ ответ "
    "с уточняющего вопроса по этой детали (например «уточните, пожалуйста, "
    "город — Тбилиси, Батуми или другой?»), а «не нашла...» в этом случае "
    "вообще не пиши — не пугай пользователя ложным отказом, если дело "
    "просто в недостающей детали; пиши «не нашла» только если знаний "
    "действительно не хватает и уточнение не поможет. ИСКЛЮЧЕНИЕ: если "
    "недостающая деталь — именно город, а ниже дан «город пользователя по "
    "умолчанию» — НЕ переспрашивай город, сразу отвечай для этого города и "
    "поставь маркер [CITY] один раз в конце ответа (подставится "
    "автоматически); не ставь [CITY], если в самом вопросе уже назван "
    "другой город, или вопрос вообще не про конкретное место.\n\n"
    "ИСКЛЮЧЕНИЕ — медицина/здоровье: используй ТОЛЬКО факты из "
    "фрагментов, ничего от себя не добавляй (даже с маркером [?]) — ни "
    "протоколов первой помощи, ни советов про срочность/врача/анализы. "
    "Если фрагментов не хватает для полного ответа — просто добавь одну "
    "фразу: «Рекомендуем обратиться к врачу.» — без деталей. Если "
    "сказать нечего вообще — честно признай, что не нашла ответа.\n\n"
    "У каждого фрагмента указан тип:\n"
    "- date_based — со временем меняется (цены, официальные требования). "
    "Если несколько фрагментов дают разные значения — доверяй более "
    "свежему по дате и упомяни, что мнения/значения расходятся (дату "
    "писать не нужно — она подставится автоматически рядом со ссылкой).\n"
    "- vote_based — рекомендация/мнение/способ. Если фрагменты называют "
    "РАЗНЫЕ варианты (места, контакты, способы) — перечисли ВСЕ различающиеся "
    "варианты, не выбирай один за пользователя.\n\n"
    "Если у фрагмента указан город — это знание касается именно этого "
    "города, не всей Грузии.\n\n"
    "ЦИТИРОВАНИЕ: сразу после каждого факта/утверждения, которое реально "
    "взято из фрагмента, поставь номер этого фрагмента в квадратных скобках "
    "— например: «через приложение Metro Georgia [2]». В скобках — ТОЛЬКО "
    "число. Несколько фрагментов подряд — [2][4]. Ссылку и дату сам не "
    "пиши — подставятся автоматически по номеру. Факт из общих знаний "
    "(не из фрагментов) получает [?] вместо номера — см. выше.\n\n"
    "Если упоминаешь ссылку/чат/сайт как часть содержания ответа (не как "
    "цитирование фрагмента) — пиши её обычным текстом как есть, НИКОГДА не "
    "в виде markdown-ссылки [текст](url)."
)

# Model marks each claim with its fragment number(s) inline — [2] or [2][4] —
# instead of a trailing "used fragments" line; this substitutes each such
# marker (or run of adjacent markers) with the real source link(s) right
# there in the text, so the reader sees what backs each specific claim
# instead of one undifferentiated list at the end.
#
# The number-only regex only matches a run of clean "[N]" markers; the digit
# regex used inside repl() is deliberately looser (matches "[N" even with
# junk before the "]", e.g. a stray "[1, сентябрь 2026]") so a model that
# doesn't follow the number-only instruction still gets linked instead of
# leaving a raw, unprocessed bracket in the answer.
_CITE_RX = re.compile(r"(?:\[\d+[^\[\]]*\])+")
_CITE_NUM_RX = re.compile(r"\[(\d+)")
# General-knowledge marker: the model tags a fact that ISN'T from a fragment
# with [?] instead of composing its own "in general it's known that..."
# transition every time — same reasoning as _format_date: delegating the
# marking itself to code is more reliable than trusting free-text compliance
# (see chat history: the model reliably marks whole paragraphs this way but
# was inconsistent about marking a single unsupported detail inside an
# otherwise fragment-grounded sentence).
_GK_RX = re.compile(r"\[\?\]")
_GK_MARKER = " (не из чата)"

# Same "delegate the marking to code, not free text" pattern, for the
# per-user default-city feature (src/bot.py's /city) — the model signals
# "I answered for the default city instead of asking" with a fixed [CITY]
# token; the actual city name/footnote is filled in here since the model
# never even sees the mechanism, only the city name itself (see answer()).
_CITY_MARKER_RX = re.compile(r"\[CITY\]")

# Which cities are well-covered enough to be worth defaulting to — computed
# from the ACTUAL knowledge base (not hardcoded), shared by src.bot's /city
# picker and answer()'s own search-query city injection below. Lives here
# (not in src/bot.py) so both can import it without a circular import —
# src/bot.py already imports from src.rag, not the other way around.
_CITY_MIN_COUNT = 5
_NON_GEORGIAN_CITY_NOISE = {"Ереван", "Москва", "Владикавказ", "Санкт-Петербург", "Стамбул"}


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


# Computed once at import — see src.bot's identical comment on why a fresh
# count per request isn't worth it (data/knowledge only changes via the
# weekly refresh job, which restarts this process anyway).
_CITY_COUNTS = _compute_city_counts()
_CITY_OPTIONS = [
    c for c, n in sorted(_CITY_COUNTS.items(), key=lambda kv: -kv[1])
    if n >= _CITY_MIN_COUNT and c not in _NON_GEORGIAN_CITY_NOISE
]

_GEORGIA_WIDE_RX = re.compile(r"[Гг]рузи[а-я]*")


def _mentions_place(query: str) -> bool:
    """True if the query already names a specific city (so injecting the
    user's default would silently override an explicit different city) OR
    asks about Georgia-wide/unspecified scope ("по всей Грузии", "в
    Грузии") — a country-wide question shouldn't get narrowed to one city
    either. Cheap substring/regex check, same style as the other
    deterministic markers in this file — no extra LLM call."""
    if _GEORGIA_WIDE_RX.search(query):
        return True
    return any(city in query for city in _CITY_OPTIONS)


_MONTHS_RU = [
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
]


def _is_private(chat_username: str) -> bool:
    chat = next((c for c in config.CHATS if c["username"] == chat_username), None)
    return bool(chat and chat.get("private"))


def _format_date(date_str: str | None) -> str:
    """'2026-09-19T...' -> 'сентябрь 2026' — computed here, not written by the
    model, so it can't hallucinate or forget a date (see chat history: it did
    both when asked to write dates itself)."""
    if not date_str:
        return ""
    try:
        dt = datetime.fromisoformat(date_str)
    except ValueError:
        return ""
    return f"{_MONTHS_RU[dt.month - 1]} {dt.year}"


def _inline_citations(text: str, hits: list[dict]) -> tuple[str, list[dict]]:
    used_indices: set[int] = set()
    seen_citations: set[tuple] = set()  # (lock, link, date) already rendered
    # inline, tracked across the WHOLE answer, not just one "[N][M]" run —
    # retrieval can return two different hits (different indices in `hits`)
    # that are near-duplicate knowledge units citing the same source
    # message, and the model may then cite that source after every sentence
    # in a block that's really about one thing. A source already shown once
    # doesn't need re-showing on every following claim it also backs — the
    # first mark is enough, later ones would just be visual noise; it's
    # still listed once in the footer regardless of how many claims used it.

    def repl(m: re.Match) -> str:
        parts = []
        for n in _CITE_NUM_RX.findall(m.group(0)):
            i = int(n)
            if 1 <= i <= len(hits):
                m_ = hits[i - 1]["meta"]
                used_indices.add(i)
                lock = "🔒" if _is_private(m_["chat_username"]) else ""
                date = _format_date(m_.get("date"))
                key = (lock, m_["link"], date)
                if key in seen_citations:
                    continue
                seen_citations.add(key)
                parts.append(f"({lock}{m_['link']}{f', {date}' if date else ''})")
        return f" {' '.join(parts)}" if parts else ""

    text = _GK_RX.sub(_GK_MARKER, text)
    text = _CITE_RX.sub(repl, text)
    text = re.sub(r"[ \t]{2,}", " ", text)  # citation markers leave a double space behind

    sources = []
    seen: set[str] = set()
    for i in sorted(used_indices):
        m_ = hits[i - 1]["meta"]
        if m_["link"] not in seen:
            sources.append({
                "title": m_["chat_title"], "link": m_["link"],
                "private": _is_private(m_["chat_username"]),
            })
            seen.add(m_["link"])
    return text, sources


# Contact masking — applied HERE, live, per-fragment, in _build_context
# below (config.MASK_PHONE_IN_PRIVATE_CHATS/MASK_USERNAME_IN_PRIVATE_CHATS),
# never baked into data/knowledge/*.fixed.jsonl itself. That file always
# stores the full, real text for every chat, private or public — so
# changing the masking policy later (mask usernames but not phones, or vice
# versa; add/remove a chat from the private list) is a config edit only,
# never a re-collection or re-index. Only a fragment from a chat marked
# "private": True in config.CHATS (see _is_private) is ever masked; public
# chats' contacts pass through untouched.
_PHONE_RX = re.compile(r"\+?\(?\d[\d\-\s\(\)]{5,}\d")
_TG_HANDLE_RX = re.compile(r"@\w{4,}")
_HANDLE_PLACEHOLDER = "@username"


def _mask_contacts(text: str) -> str:
    if config.MASK_PHONE_IN_PRIVATE_CHATS:
        def _phone_repl(m: re.Match) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            # >=9 digits: matches Georgian/Russian mobile numbers, not dates
            # or short incidental numbers.
            return "X" * len(digits) if len(digits) >= 9 else m.group(0)
        text = _PHONE_RX.sub(_phone_repl, text)
    if config.MASK_USERNAME_IN_PRIVATE_CHATS:
        text = _TG_HANDLE_RX.sub(_HANDLE_PLACEHOLDER, text)
    return text


# A bare "@username" or a run of X's standing in for a real phone number
# reads as broken/suspicious if the reader doesn't know it's deliberate
# privacy masking, not a redaction of something wrong. One short footnote,
# only when an answer actually contains one — same "annotate only when it
# fires" pattern as _GK_MARKER above.
_MASKED_CONTACT_RX = re.compile(r"@username\b|X{9,}")
_MASKED_CONTACT_NOTE = (
    "\n\n🙈 Контакты (телефон/юзернейм) иногда скрыты — так безопаснее "
    "для тех, кто их оставлял в чате."
)

_URL_RX = re.compile(r"https?://\S+")
_BOLD_RX = re.compile(r"\*\*(.+?)\*\*")
_URL_TRAILING_PUNCT = ".,!?;:)]"  # sentence punctuation right after a bare URL isn't part of it


def _link_text_for_url(url: str) -> str:
    """What to show as the clickable word for a URL, instead of a generic
    "ссылка" that gives the reader no idea what's behind it. A t.me link
    always identifies its own chat by construction (see src/ingest.py::
    _chat_link — public: /<username>/, private: /c/<internal_id>/), so this
    needs no metadata beyond config.CHATS, and works for BOTH the citation
    links _inline_citations adds AND any chat link the model wrote itself as
    part of the answer's content (e.g. "чат https://t.me/paravaingeorgia").
    A link to anything else (an external guide, a website) shows its domain
    instead — still concrete, never the content-free "ссылка"."""
    for chat in config.CHATS:
        if chat.get("private"):
            marker = f"t.me/c/{str(abs(chat['chat_id']))[3:]}/"
        else:
            marker = f"t.me/{chat['username']}"
        if marker in url:
            return chat["title"]
    return urlparse(url).netloc or "ссылка"


def _linkify(m: re.Match) -> str:
    url = m.group(0)
    trail = ""
    while url and url[-1] in _URL_TRAILING_PUNCT:
        trail = url[-1] + trail
        url = url[:-1]
    return f'<a href="{url}">{_link_text_for_url(url)}</a>{trail}'


# Matches exactly the citation parenthetical _inline_citations produces —
# "(🔒?URL, month year)" or "(🔒?URL)" — so it can be told apart from a bare
# content link elsewhere in the text (e.g. a chat the model recommends by
# name/URL as part of the actual answer). Citations get the plain word
# "source" instead of a full link/name: the reader wants those out of the
# way, not competing for attention with an actionable "go join this chat"
# link — but the date must stay visible (dropped once already by mistake).
_CITE_PAREN_RX = re.compile(r"\((🔒)?(https?://[^()\s]+)(?:,\s*([^()]*))?\)")


def to_telegram_html(text: str) -> str:
    """Render an answer (raw links, **bold**) as Telegram HTML parse-mode
    markup. Two different kinds of link get two different treatments:
      - citation parentheticals (provenance, not something to act on) become
        "(🔒month year)" / "(month year)" — the DATE ITSELF is the link, no
        extra word/icon/number: it's real information (not a repeated filler
        word), reads differently every time, and Telegram's blue underline
        already makes it obviously clickable;
      - any other bare URL (a chat/site the model recommends as part of the
        answer's actual content) becomes a real clickable chat name/domain
        via _link_text_for_url, since that IS actionable information.
    **bold** becomes <b>. Escape first (before adding our own tags), per
    Telegram's HTML rules — only &, <, > need escaping in plain text:
    https://core.telegram.org/bots/api#html-style"""
    escaped = html.escape(text, quote=False)

    # Citations are pulled out to placeholders first so the later bare-URL
    # pass (content links) can't re-match a URL that's already sitting
    # inside a citation's href — without this, that URL gets linkified a
    # second time, nested inside the first anchor tag.
    placeholders: list[str] = []

    def cite_repl(m: re.Match) -> str:
        lock, url, date = m.group(1) or "", m.group(2), m.group(3)
        link_text = date or "источник"  # dateless is rare (old knowledge) — a link needs some visible text
        placeholders.append(f'({lock}<a href="{url}">{link_text}</a>)')
        return f"\x00{len(placeholders) - 1}\x00"

    escaped = _CITE_PAREN_RX.sub(cite_repl, escaped)
    escaped = _URL_RX.sub(_linkify, escaped)
    for i, html_snippet in enumerate(placeholders):
        escaped = escaped.replace(f"\x00{i}\x00", html_snippet)
    escaped = _BOLD_RX.sub(r"<b>\1</b>", escaped)
    if _MASKED_CONTACT_RX.search(escaped):
        escaped += _MASKED_CONTACT_NOTE
    return escaped


def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        city = f", город: {m['city']}" if m.get("city") else ""
        # Mask ONLY if this fragment's own source chat is private — the
        # knowledge store itself is never masked (see _mask_contacts above),
        # so a public-chat fragment's real contacts pass straight through.
        answer = _mask_contacts(m["answer"]) if _is_private(m["chat_username"]) else m["answer"]
        blocks.append(
            f"[Фрагмент {i}] тип: {m['type']}{city}, дата: {m.get('date', '')}\n"
            f"ссылка: {m['link']}\n"
            f"Вопрос: {m['question']}\nОтвет: {answer}"
        )
    return "\n\n".join(blocks)


REWRITE_SYSTEM = (
    "Перепиши ПОСЛЕДНЕЕ сообщение пользователя в самостоятельный вопрос, "
    "понятный без истории переписки. НО: используй историю ТОЛЬКО если "
    "сообщение само по себе неполное и без истории вообще непонятно, о чём "
    "речь — одно слово/фраза-ответ на предыдущий вопрос ассистента "
    "(«рабочая» в ответ на «туристическая или рабочая виза?»), местоимение "
    "без антецедента («а там?», «и это тоже?»). Если сообщение — НОВЫЙ "
    "самостоятельный вопрос со своей темой (даже если тема рядом с "
    "предыдущей или использует то же слово) — верни его КАК ЕСТЬ, не "
    "притягивай к нему предыдущую тему. Пример ОШИБКИ: после разговора про "
    "визу в Испанию пользователь спрашивает «где найти врача» — это НОВЫЙ "
    "вопрос про врача вообще, а не «врач для медосмотра на испанскую визу» "
    "— так дописывать нельзя. Ответь строго JSON: {\"query\": \"...\"}"
)


def _rewrite_query(message: str, history: list[dict]) -> str:
    """Fold short history-dependent replies ("рабочая") into one standalone
    question BEFORE retrieval — retrieval only ever sees the current message,
    so without this, a one-word follow-up embeds to something unrelated and
    finds nothing useful, even though generation might have the history."""
    if not history:
        return message
    convo = "\n".join(
        f"{'Пользователь' if h['role'] == 'user' else 'Ассистент'}: {h['content']}"
        for h in history
    )
    data = chat_json(
        config.GENERATION_MODEL, REWRITE_SYSTEM,
        f"{convo}\nПоследнее сообщение пользователя: {message}",
        reasoning_effort=config.GENERATION_REASONING_EFFORT,
    )
    return (data.get("query") or "").strip() or message


GUEST_CLASSIFY_SYSTEM = (
    "Ты — фильтр для Telegram-бота, который отвечает на практические вопросы "
    "о жизни в Грузии. Бота позвали в чужой чат (упомянули его или ответили "
    "на его сообщение). Реши, просят ли бота о чём-то, и если да — "
    "сформулируй самостоятельный вопрос.\n\n"
    "Метки:\n"
    "- request — у бота просят информацию или помощь. В ЛЮБОЙ форме: вопрос, "
    "просьба («подскажи…»), или просто тема без глагола («рекомендации "
    "лор-врачей», «виза», «банк для ИП») — голая тема после вызова бота = "
    "просьба рассказать о ней. Если бота позвали ответом на чужое "
    "сообщение (даже без своего текста) — считай, что просят помочь с темой "
    "того сообщения. Ответ на предыдущее сообщение бота с уточнением, "
    "продолжением или возражением («рабочая», «а в Батуми?», «а сколько "
    "стоит?», «это устарело?») — тоже request.\n"
    "- thanks — благодарность, согласие, реакция без новой просьбы "
    "(«спасибо», «ок», «понял», «круто»).\n"
    "- chatter — ответили на сообщение бота, но обращаются к другим людям "
    "или просто комментируют/спорят, ничего не спрашивая у бота.\n"
    "- mention — бота упомянули мимоходом (советуют кому-то, говорят о "
    "боте), ни о чём его не прося.\n"
    "Если сомневаешься между request и другой меткой — выбирай request.\n"
    "Тема не про Грузию — всё равно request (это разберут дальше).\n\n"
    "query — только для request: вопрос по-русски, понятный без контекста "
    "переписки (подставь тему из сообщения, на которое ответили, или из "
    "предыдущего ответа бота, если без них непонятно, о чём речь). Для "
    "остальных меток — пустая строка.\n"
    "Ответь строго JSON: {\"label\": \"...\", \"query\": \"...\"}"
)


def classify_guest(
    text: str, replied_text: str | None = None, replied_is_bot: bool = False,
) -> dict:
    """Guest-mode gate (src.bot.on_guest_message): {"label": request|thanks|
    chatter|mention, "query": standalone question or ""}. Doubles as the
    follow-up rewrite for guest mode — there's no persisted history there
    (guest chat ids can collide with other chats), only the one message the
    caller replied to, so that message IS the history."""
    parts = []
    if replied_text:
        who = "предыдущий ответ бота" if replied_is_bot else "сообщение другого человека"
        parts.append(f"Сообщение, на которое ответили ({who}):\n{replied_text}")
    parts.append(f"Сообщение, которым позвали бота:\n{text or '(пусто — только упоминание бота)'}")
    data = chat_json(
        config.GUEST_CLASSIFY_MODEL, GUEST_CLASSIFY_SYSTEM, "\n\n".join(parts),
        reasoning_effort=config.GUEST_CLASSIFY_REASONING_EFFORT,
    )
    label = data.get("label") if data.get("label") in {"request", "thanks", "chatter", "mention"} else "request"
    query = (data.get("query") or "").strip()
    if label == "request" and not query:
        query = text or replied_text or ""
    return {"label": label, "query": query}


def answer(
    query: str, k: int = config.TOP_K, *,
    history: list[dict] | None = None, user_city: str | None = None,
) -> dict:
    """history, if given, is the recent conversation as
    [{"role": "user"|"assistant", "content": "..."}, ...] (oldest first) —
    used only to resolve a short follow-up into a standalone question before
    retrieval; the answer itself is still generated fresh from fragments,
    not from the conversation.

    user_city, if given (src.bot's /city preference), lets the model skip
    the "which city?" clarifying question and answer for that city instead
    — see the [CITY] marker handling below and SYSTEM_PROMPT's ИСКЛЮЧЕНИЕ
    clause. Also folded into the embedding query (see _mentions_place, and
    embed_query below) when the question doesn't already name a city or ask
    Georgia-wide — a bare
    "где купить фанеру?" used to search blind across every city's fragments,
    competing against Tbilisi-specific ones instead of being helped by
    knowing the user is in Tbilisi; a question that already says "в Батуми"
    or "по всей Грузии" is left alone so the default never overrides it."""
    search_query = _rewrite_query(query, history) if history else query
    # Separate from search_query on purpose: search_query is also what goes
    # into the generation prompt's "Вопрос:" line below (via city_line,
    # already stating the default city there) — appending it twice would
    # just be redundant, not wrong, but embed_query keeps that line clean.
    embed_query = f"{search_query} {user_city}" if user_city and not _mentions_place(search_query) else search_query
    hits = search(embed_query, k=k)
    # No hits now usually means "nothing relevant enough" (min_score filtered
    # everything out), not necessarily an empty index. Still call the model —
    # with no fragments it falls straight to the general-knowledge branch of
    # SYSTEM_PROMPT instead of a hardcoded "not found".
    context = _build_context(hits) if hits else "(пусто — по этому вопросу в чатах ничего релевантного не нашлось)"
    city_line = f"\nГород пользователя по умолчанию: {user_city}." if user_city else ""
    user_prompt = (
        f"Вопрос: {search_query}{city_line}\n\n"
        f"Фрагменты переписок:\n{context}\n\n"
        f"Дай ответ по существу, помечая каждый факт номером фрагмента в "
        f"квадратных скобках сразу после него (см. ЦИТИРОВАНИЕ выше)."
    )
    client = openai_client()
    kwargs: dict = {
        "model": config.azure_deployment(config.GENERATION_MODEL),
        "seed": config.LLM_SEED,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if config.GENERATION_MODEL in config.REASONING_MODELS:
        if config.GENERATION_REASONING_EFFORT:
            kwargs["reasoning_effort"] = config.GENERATION_REASONING_EFFORT
    else:
        kwargs["temperature"] = 0.2
    resp = client.chat.completions.create(**kwargs)
    raw_text = resp.choices[0].message.content or ""

    text, sources = _inline_citations(raw_text, hits)
    if user_city and _CITY_MARKER_RX.search(text):
        text = _CITY_MARKER_RX.sub("", text).rstrip()
        text += f"\n\n📍 Ответ для города по умолчанию — {user_city}. Сменить: /city"
    # raw_text (model's own [N]-marker output, before link substitution) is
    # for eval (src/eval_rag.py) to inspect citation-format compliance —
    # bot.py and the notebook only ever read "answer"/"sources".
    return {"answer": text, "sources": sources, "raw_text": raw_text}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.rag "your question"')
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print(result["answer"])


if __name__ == "__main__":
    main()
