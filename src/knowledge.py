"""Distill conversation threads into reusable Q&A knowledge via an LLM.

Pipeline: raw messages -> spam filter -> threads (src.threads) -> for each
thread the LLM extracts 0..N {question, answer} knowledge units, each tagged
with a link to the thread's ROOT message. Threads with no reusable knowledge
(chit-chat, greetings, ads, off-topic) yield nothing.

Output: data/knowledge/<username>.jsonl

⚠️ This calls the OpenAI chat model once per thread — it costs tokens. For a
prototype run on a subset, use `distill_chat(username, limit=...)` from the
notebook; eyeball the result before scaling up.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from src.preprocess import _load_raw
from src.spam import filter_spam
from src.store import openai_client
from src.threads import build_threads

# Russian on purpose: source chats and the target assistant are Russian-speaking.
SYSTEM_PROMPT = (
    "Ты извлекаешь полезные ДОЛГОВЕЧНЫЕ знания о жизни в Грузии из переписок "
    "Telegram-чатов для справочного ассистента. На вход — один тред (обсуждение). "
    "Сформируй список пар «вопрос-ответ» в формате JSON.\n\n"
    "Правила:\n"
    "1. Опирайся ТОЛЬКО на то, что реально сказано в треде. Ничего не выдумывай. "
    "Если уверенности нет — не включай.\n"
    "2. АТОМАРНОСТЬ — главное правило. Каждая пара = РОВНО ОДНА самостоятельная "
    "тема/вопрос. Если в треде обсуждают несколько разных тем — раздели их на "
    "НЕСКОЛЬКО отдельных пар. НИКОГДА не смешивай разные темы в одном ответе "
    "(например, «как получить права» и «где стоматолог» — это две разные пары). "
    "Несколько фактов или нюансов в рамках ОДНОЙ темы — объединяй в один ответ.\n"
    "3. ЕСТЬ ОТВЕТ. Создавай пару, только если в треде есть реальный, полезный "
    "ответ. Если на вопрос никто не ответил, или ответ по сути «информации нет» / "
    "пересказ самого вопроса — НЕ включай такую пару.\n"
    "4. ТОЛЬКО ДОЛГОВЕЧНОЕ ЗНАНИЕ. Нас интересует справочная информация, полезная "
    "многим и надолго: процедуры, документы, как что устроено, общие советы «где/"
    "как сделать X». НЕ извлекай разовые объявления: продажа/покупка конкретных "
    "вещей, билеты, пристройство животных, разовые просьбы («кто едет», «ищу "
    "попутчика», «перевезти Х числа»). Пример: «где вообще продать книги» → можно "
    "(общий совет); «продаю конкретную арматуру / котёнка / билет» → пропустить.\n"
    "5. `question` — общий вопрос по одной теме, как его задал бы новый человек "
    "(без имён и лишнего контекста треда).\n"
    "6. `answer` — связный практичный ответ по этой одной теме. Если мнения "
    "расходятся или есть нюансы/условия — отрази это. Это мнения участников чата, "
    "а не официальный источник.\n"
    "7. ТИП. Для каждой пары укажи `type`:\n"
    "   - \"volatile\" — меняется со временем: законы, правила, налоги, цены, "
    "визовые/таможенные требования, требования к документам, расписания;\n"
    "   - \"stable\" — рекомендации (врач, мастер, магазин, кафе, специалист), "
    "практические советы «где/как», адреса и контакты;\n"
    "   - \"evergreen\" — вневременное: история, география, культура, язык, "
    "факты, которые почти не меняются.\n"
    "8. Один тред может дать несколько пар или ни одной. Если долговечного знания "
    "нет — верни {\"knowledge\": []}.\n"
    "9. Язык ответа — русский.\n\n"
    "Формат ответа строго: "
    "{\"knowledge\": [{\"question\": \"...\", \"answer\": \"...\", \"type\": \"volatile|stable|evergreen\"}]}"
)


def _thread_text(thread: list[dict]) -> str:
    lines = []
    for m in thread:
        sender = m.get("sender") or "Аноним"
        lines.append(f"{sender}: {m['text']}")
    return "\n".join(lines)


def distill_thread(thread: list[dict], chat: dict) -> list[dict]:
    """Return a list of knowledge units for one thread (possibly empty)."""
    client = openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _thread_text(thread)},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        items = data.get("knowledge", [])
    except (json.JSONDecodeError, AttributeError):
        return []

    root = thread[0]
    # latest activity in the thread — used later for recency weighting
    thread_date = thread[-1].get("date") or root.get("date") or ""
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not q or not a:
            continue
        ktype = (it.get("type") or "stable").strip().lower()
        if ktype not in {"volatile", "stable", "evergreen"}:
            ktype = "stable"
        out.append(
            {
                "question": q,
                "answer": a,
                "type": ktype,
                "date": thread_date,
                "chat_username": chat["username"],
                "chat_title": chat["title"],
                "root_msg_id": root["msg_id"],
                "root_link": root["link"],
            }
        )
    return out


def distill_chat(
    username: str,
    *,
    limit: int | None = None,
    min_thread_size: int = 2,
    write: bool = True,
) -> list[dict]:
    """Distill threads of one chat into knowledge units.

    Args:
        limit: process at most this many threads (for cheap prototype runs).
        min_thread_size: skip threads with fewer messages (default 2 = only
            discussions; set 1 to also distill standalone informative messages).
        write: also write data/knowledge/<username>.jsonl.
    """
    chat = next((c for c in config.CHATS if c["username"] == username), None)
    if chat is None:
        raise SystemExit(f"{username} is not in config.CHATS")

    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found — run src.ingest first")

    msgs, _ = filter_spam(_load_raw(raw_path)) if config.FILTER_SPAM else (_load_raw(raw_path), [])
    threads = [t for t in build_threads(msgs) if len(t) >= min_thread_size]
    if limit is not None:
        threads = threads[:limit]

    knowledge: list[dict] = []
    for i, thread in enumerate(threads, 1):
        knowledge.extend(distill_thread(thread, chat))
        if i % 25 == 0 or i == len(threads):
            print(f"[knowledge] {username}: {i}/{len(threads)} threads -> {len(knowledge)} items")

    if write:
        out_path = config.KNOWLEDGE_DIR / f"{username}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for k in knowledge:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        print(f"[knowledge] saved {len(knowledge)} items -> {out_path}")
    return knowledge


def main() -> None:
    # Full run over all configured chats (costs tokens — see module docstring).
    for chat in config.CHATS:
        distill_chat(chat["username"])


if __name__ == "__main__":
    main()
