"""RAG: retrieve context + generate a Russian answer with source links.

Run:  uv run python -m src.rag "какие документы нужны для ип"
"""
from __future__ import annotations

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
    "Ты — ассистент по жизни в Грузии. Отвечай на русском, опираясь ТОЛЬКО на "
    "приведённые ниже фрагменты знаний, извлечённые и проверенные из "
    "Telegram-чатов. Если в контексте нет ответа — честно скажи, что не нашёл "
    "информации. Не выдумывай. Это мнения и опыт людей из чатов, а не "
    "официальные источники — при необходимости делай оговорку.\n\n"
    "У каждого фрагмента указан тип:\n"
    "- date_based — со временем меняется (цены, официальные требования). "
    "Если несколько фрагментов дают разные значения — доверяй более свежим по "
    "дате, но упомяни расхождение и не скрывай, что цифра могла устареть.\n"
    "- vote_based — рекомендация/мнение/способ. Если фрагменты называют "
    "РАЗНЫЕ варианты (места, контакты, способы) — перечисли ВСЕ различающиеся "
    "варианты, не выбирай один за пользователя.\n\n"
    "Если у фрагмента указан город — это знание касается именно этого "
    "города, не всей Грузии.\n\n"
    "В конце ответа приведи ссылки на источники, которые реально "
    "использовал в ответе."
)


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
    if not hits:
        # User-facing message (Russian on purpose)
        return {"answer": "В базе пока нет данных. Запусти индексацию.", "sources": []}

    context = _build_context(hits)
    user_prompt = (
        f"Вопрос: {query}\n\n"
        f"Фрагменты переписок:\n{context}\n\n"
        f"Дай ответ по существу и список ссылок-источников."
    )
    client = openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content
    sources = [{"title": h["meta"]["chat_title"], "link": h["meta"]["link"]} for h in hits]
    return {"answer": text, "sources": sources}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.rag "your question"')
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print(result["answer"])


if __name__ == "__main__":
    main()
