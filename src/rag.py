"""RAG: поиск контекста + генерация ответа на русском со ссылками на источники.

Запуск:  uv run python -m src.rag "какие документы нужны для ип"
"""
from __future__ import annotations

import sys

import config
from src.retrieve import search
from src.store import openai_client

SYSTEM_PROMPT = (
    "Ты — ассистент по жизни в Грузии. Отвечай на русском, опираясь ТОЛЬКО на "
    "приведённые фрагменты переписок из Telegram-чатов. Если в контексте нет "
    "ответа — честно скажи, что не нашёл информации. Не выдумывай. "
    "Учитывай, что это мнения людей из чатов, а не официальные источники — "
    "при необходимости делай оговорку. В конце ответа приведи ссылки на "
    "сообщения-источники, которые использовал."
)


def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(
            f"[Фрагмент {i}] чат «{m['chat_title']}», {m.get('date', '')}\n"
            f"ссылка: {m['link']}\n{h['text']}"
        )
    return "\n\n".join(blocks)


def answer(query: str, k: int = config.TOP_K) -> dict:
    hits = search(query, k=k)
    if not hits:
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
        raise SystemExit('Использование: python -m src.rag "ваш вопрос"')
    query = " ".join(sys.argv[1:])
    result = answer(query)
    print(result["answer"])


if __name__ == "__main__":
    main()
