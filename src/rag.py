"""RAG: retrieve context + generate a Russian answer with source links.

Run:  uv run python -m src.rag "какие документы нужны для ип"
"""
from __future__ import annotations

import re
import sys

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
    "Если фрагменты НЕ отвечают на вопрос (совсем или частично) — по "
    "оставшейся части коротко ответь из своих общих знаний, БЕЗ выдумывания "
    "фактов, которых не знаешь. Такой ответ явно отдели фразой вроде "
    "«В чатах такого не обсуждали, но в целом известно, что...» — не выдавай "
    "общие знания за опыт чата. Если и общих знаний нет — честно скажи, что "
    "не нашла ответа.\n\n"
    "У каждого фрагмента указан тип:\n"
    "- date_based — со временем меняется (цены, официальные требования). "
    "ВСЕГДА указывай в ответе, на какую дату эти сведения (месяц и год из "
    "поля «дата» фрагмента) — не только когда фрагменты расходятся, а "
    "каждый раз. Если несколько фрагментов дают разные значения — доверяй "
    "более свежим по дате, но упомяни расхождение и обе даты.\n"
    "- vote_based — рекомендация/мнение/способ. Если фрагменты называют "
    "РАЗНЫЕ варианты (места, контакты, способы) — перечисли ВСЕ различающиеся "
    "варианты, не выбирай один за пользователя.\n\n"
    "Если у фрагмента указан город — это знание касается именно этого "
    "города, не всей Грузии.\n\n"
    "В САМОМ ответе никаких ссылок не пиши — они добавятся отдельно. "
    "Вместо этого последней строкой, отдельно, укажи номера фрагментов, "
    "факты из которых реально вошли в ответ (не все, что тебе просто "
    "показали), в формате: ИСПОЛЬЗОВАНО: 1, 3\n"
    "Если ничего не использовал — напиши ИСПОЛЬЗОВАНО: (пусто)."
)

_USED_LINE_RX = re.compile(r"\n?ИСПОЛЬЗОВАНО:\s*(.*)\s*$", re.IGNORECASE)


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
        f"Дай ответ по существу, без ссылок в тексте; номера использованных фрагментов — отдельной строкой в конце."
    )
    client = openai_client()
    kwargs: dict = {
        "model": config.GENERATION_MODEL,
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

    # The model reports which fragment NUMBERS it used on a trailing
    # "ИСПОЛЬЗОВАНО: 1, 3" line (see SYSTEM_PROMPT) instead of writing links
    # in the answer itself — one source of truth (`sources`, built from
    # `hits` by index below), not links duplicated in both the prose and a
    # separate field.
    m = _USED_LINE_RX.search(text)
    used_indices: set[int] = set()
    if m:
        used_indices = {int(n) for n in re.findall(r"\d+", m.group(1))}
        text = text[: m.start()].rstrip()  # strip the marker line from the shown answer

    sources = []
    seen_links: set[str] = set()
    for i in sorted(used_indices):
        if 1 <= i <= len(hits):
            h = hits[i - 1]
            link = h["meta"]["link"]
            if link not in seen_links:
                sources.append({"title": h["meta"]["chat_title"], "link": link})
                seen_links.add(link)
    return {"answer": text, "sources": sources}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.rag "your question"')
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print(result["answer"])


if __name__ == "__main__":
    main()
