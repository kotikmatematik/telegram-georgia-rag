"""RAG: retrieve context + generate a Russian answer with source links.

Run:  uv run python -m src.rag "какие документы нужны для ип"
"""
from __future__ import annotations

import html
import re
import sys
from datetime import datetime

import config
from src.retrieve import search
from src.store import openai_client

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
    "ОЧЕРЕДЬ на приведённые ниже фрагменты знаний, извлечённые и проверенные "
    "из Telegram-чатов. Это мнения и опыт людей из чатов, а не официальные "
    "источники — при необходимости делай оговорку.\n\n"
    "ВСЕ фрагменты — про Грузию. Ретрив ищет по смыслу и может по ошибке "
    "подложить грузинский фрагмент на вопрос про ДРУГУЮ страну (совпадает "
    "тема — «права», «страховка» — но не страна). Если вопрос на самом деле "
    "про то, как что-то устроено в ДРУГОЙ стране — НЕ выдавай грузинские "
    "факты за ответ на него, даже если фрагмент выглядит подходящим по "
    "теме; честно скажи, что чаты только про Грузию и этого вопроса они не "
    "касаются. Но если вопрос по сути про жизнь В Грузии, а другая страна "
    "лишь упомянута как пункт назначения/контрагент (например «как "
    "перевести деньги из Грузии в Испанию», «работают ли грузинские карты "
    "за границей») — отвечай как обычно.\n\n"
    "Если фрагменты НЕ отвечают на вопрос (совсем или частично) — по "
    "оставшейся части коротко ответь из своих общих знаний, БЕЗ выдумывания "
    "фактов, которых не знаешь. Такой ответ явно отдели фразой вроде "
    "«В чатах такого не обсуждали, но в целом известно, что...» — не выдавай "
    "общие знания за опыт чата. Если и общих знаний нет — честно скажи, что "
    "не нашла ответа.\n\n"
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
    "пиши — подставятся автоматически по номеру. Общие знания (не из "
    "фрагментов) номера не получают.\n\n"
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

    def repl(m: re.Match) -> str:
        parts = []
        for n in _CITE_NUM_RX.findall(m.group(0)):
            i = int(n)
            if 1 <= i <= len(hits):
                used_indices.add(i)
                m_ = hits[i - 1]["meta"]
                lock = "🔒" if _is_private(m_["chat_username"]) else ""
                date = _format_date(m_.get("date"))
                parts.append(f"({lock}{m_['link']}{f', {date}' if date else ''})")
        return f" {' '.join(parts)}" if parts else ""

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


_URL_RX = re.compile(r"https?://\S+")
_BOLD_RX = re.compile(r"\*\*(.+?)\*\*")
_URL_TRAILING_PUNCT = ".,!?;:)]"  # sentence punctuation right after a bare URL isn't part of it


def _linkify(m: re.Match) -> str:
    url = m.group(0)
    trail = ""
    while url and url[-1] in _URL_TRAILING_PUNCT:
        trail = url[-1] + trail
        url = url[:-1]
    return f'<a href="{url}">ссылка</a>{trail}'


def to_telegram_html(text: str) -> str:
    """Render an answer (raw links, **bold**) as Telegram HTML parse-mode
    markup: bare URLs become a clickable "ссылка" (a 🔒 in front, if present,
    stays outside the link as a plain marker) and **bold** becomes <b>. Escape
    first (before adding our own tags), per Telegram's HTML rules — only
    &, <, > need escaping in plain text: https://core.telegram.org/bots/api#html-style"""
    escaped = html.escape(text, quote=False)
    escaped = _URL_RX.sub(_linkify, escaped)
    escaped = _BOLD_RX.sub(r"<b>\1</b>", escaped)
    return escaped


def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        city = f", город: {m['city']}" if m.get("city") else ""
        blocks.append(
            f"[Фрагмент {i}] тип: {m['type']}{city}, дата: {m.get('date', '')}\n"
            f"ссылка: {m['link']}\n"
            f"Вопрос: {m['question']}\nОтвет: {m['answer']}"
        )
    return "\n\n".join(blocks)


def answer(query: str, k: int = config.TOP_K) -> dict:
    hits = search(query, k=k)
    # No hits now usually means "nothing relevant enough" (min_score filtered
    # everything out), not necessarily an empty index. Still call the model —
    # with no fragments it falls straight to the general-knowledge branch of
    # SYSTEM_PROMPT instead of a hardcoded "not found".
    context = _build_context(hits) if hits else "(пусто — по этому вопросу в чатах ничего релевантного не нашлось)"
    user_prompt = (
        f"Вопрос: {query}\n\n"
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
    text = resp.choices[0].message.content or ""

    text, sources = _inline_citations(text, hits)
    return {"answer": text, "sources": sources}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.rag "your question"')
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print(result["answer"])


if __name__ == "__main__":
    main()
