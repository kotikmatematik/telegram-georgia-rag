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
import re
from pathlib import Path

import config
from src.preprocess import _load_raw, _parse_dt
from src.spam import filter_spam
from src.store import chat_json
from src.threads import build_threads


def _thread_latest_dt(thread: list[dict]):
    dts = [d for d in (_parse_dt(m.get("date")) for m in thread) if d]
    return max(dts) if dts else None


# Backstop for rule #3 of SYSTEM_PROMPT: even when told not to, the model
# sometimes still writes a pair whose `answer` just states that there's no
# answer (e.g. "информация ... в треде не указана"). Catch that mechanically
# rather than trust prompt-following alone — same approach as src/spam.py.
_NON_ANSWER_PATTERNS = [
    # "информации .{0,60} не указана / нет / отсутствует" — allow a gap, since
    # the actual subject usually sits between the trigger and the verdict
    # ("информация О ПОКУПКЕ ОРБИТРЕКА ... в треде не указана").
    r"информаци\w*.{0,120}?(не\s+(указан|предоставлен|найден|дан|сообщ)\w*|отсутствует|\bнет\b)",
    r"не\s+(указан|предоставлен|найден|сообщ)\w*\s+(конкретн\w*\s+)?информаци",
    r"нет\s+(точной|конкретной|)\s*информаци",
    r"данн\w*\s+(отсутств\w*|не\s+(указан|предоставлен)\w*)",
    r"\bнеизвестн\w*",
]
_NON_ANSWER_RX = re.compile("|".join(_NON_ANSWER_PATTERNS), re.IGNORECASE)


def _is_non_answer(answer: str) -> bool:
    return bool(_NON_ANSWER_RX.search(answer))

# Russian on purpose: source chats and the target assistant are Russian-speaking.
SYSTEM_PROMPT = (
    "Ты извлекаешь полезные ДОЛГОВЕЧНЫЕ знания о жизни в Грузии из переписок "
    "Telegram-чатов для справочного ассистента. На вход — один тред (обсуждение). "
    "Сформируй список пар «вопрос-ответ» в формате JSON.\n\n"
    "Правила:\n"
    "1. Опирайся ТОЛЬКО на то, что реально сказано в треде. Ничего не выдумывай.\n"
    "2. АТОМАРНОСТЬ. Каждая пара = РОВНО ОДНА тема. Тред с несколькими темами — "
    "раздели на несколько пар.\n"
    "3. ЕСТЬ ОТВЕТ. Создавай пару, только если на вопрос реально ответили. Если "
    "на какую-то тему в треде ответа нет — пропусти эту тему; никогда не пиши в "
    "`answer` «информация не указана/нет в треде» — это повод не создавать пару, "
    "а не ответ.\n"
    "4. ТОЛЬКО ДОЛГОВЕЧНОЕ. Процедуры, документы, устройство, общие советы "
    "«где/как». НЕ включай разовые объявления (продажа вещи, билет, пристройство "
    "животного, «кто едет Х числа»).\n"
    "5. `question` — общий вопрос по теме, как задал бы новый человек. Если "
    "исходное сообщение содержит НЕСКОЛЬКО вопросов («где X и сколько это "
    "стоит?»), а в треде ответили не на все — формулируй `question` под ТУ "
    "часть, на которую реально ответили, а не копируй исходную формулировку "
    "целиком. Пример: спросили «где сделать гравировку и сколько стоит», "
    "ответили только «100 лари» — вопрос должен быть «сколько стоит "
    "гравировка?», а НЕ «где сделать гравировку?» (на это не ответили).\n"
    "6. `answer` — связный практичный ответ; если мнения расходятся — отрази "
    "это. Это мнения чата, не официальный источник.\n"
    "7. `type` — как эту пару потом использовать при ответе:\n"
    "   - \"date_based\" — официально устанавливаемое (законы, налоги, визовые/"
    "таможенные требования, документы, тарифы) И любая конкретная ЦЕНА/СУММА "
    "— со временем меняется, важна самая свежая версия;\n"
    "   - \"vote_based\" — всё остальное долговечное: рекомендация ГДЕ/У КОГО/"
    "КАКИМ СПОСОБОМ без суммы в ответе, а также вневременные факты (история, "
    "география, культура) — даже если место закроется или мнение спорно, "
    "справка остаётся полезной; важно не свежесть, а сколько источников "
    "подтверждают. Например: «где обменять валюту» — vote_based (способ, без "
    "суммы); «сколько стоит обменять валюту» — date_based (это уже сумма).\n"
    "8. `city` — конкретный город (Тбилиси/Батуми/Кутаиси/...), ЕСЛИ вопрос "
    "или ответ реально привязаны к одному городу. Указывай город ТОЛЬКО если "
    "он явно назван в треде — никогда не угадывай по намёкам (улицы, районы "
    "без названия города погоды не делают). Если знание общегрузинское "
    "(визы, налоги, ИП, общий совет не про конкретное место) — city: null.\n"
    "9. Тред может дать несколько пар или ни одной. Если долговечного знания "
    "нет — верни {\"knowledge\": []}.\n"
    "10. Язык ответа — русский.\n\n"
    "Формат ответа строго: "
    "{\"knowledge\": [{\"question\": \"...\", \"answer\": \"...\", "
    "\"type\": \"date_based|vote_based\", \"city\": \"Тбилиси|null\"}]}"
)


def _thread_text(thread: list[dict]) -> str:
    lines = []
    for m in thread:
        sender = m.get("sender") or "Аноним"
        lines.append(f"{sender}: {m['text']}")
    return "\n".join(lines)


def _thread_text_with_ids(thread: list[dict]) -> str:
    """Like _thread_text, but each line is tagged [msg_id] so the branch split
    (stage 1a) can reference original messages by id, and stage 1b can check
    the split against the raw thread it's also given."""
    lines = []
    for m in thread:
        sender = m.get("sender") or "Аноним"
        lines.append(f"[{m['msg_id']}] {sender}: {m['text']}")
    return "\n".join(lines)


# Stage 1a (new): split the raw thread into semantic branches before
# extraction. This is a DRAFT for stage 1b, not ground truth by itself — see
# distill_thread, which always keeps the raw thread as the authoritative
# source and passes the split only as a hint stage 1b may correct.
SPLIT_SYSTEM = (
    "Раздели тред Telegram-чата (каждое сообщение помечено [id]) на отдельные "
    "смысловые ветки — независимые темы/вопросы, которые в нём обсуждаются "
    "(тред мог собраться по времени и реплаям, а не по теме, поэтому внутри "
    "могут быть несвязанные разговоры). Не меняй и не пересказывай текст "
    "сообщений — верни только id сообщений, входящих в каждую ветку, в "
    "исходном порядке.\n"
    "Ответь строго JSON: {\"branches\": [{\"topic\": \"кратко тема\", "
    "\"message_ids\": [id, id, ...]}]}"
)


def _split_branches(thread: list[dict]) -> list[dict]:
    """Stage 1a call: ask config.SPLIT_MODEL to group the thread's messages
    into semantic branches by id. Returns the raw branch list (possibly
    empty on failure — distill_thread then falls back to the raw thread alone)."""
    data = chat_json(
        config.SPLIT_MODEL, SPLIT_SYSTEM, _thread_text_with_ids(thread),
        reasoning_effort=config.SPLIT_REASONING_EFFORT,
    )
    return data.get("branches") or []


def _branches_text(thread: list[dict], branches: list[dict]) -> str:
    """Render id-based branches back to the same [id] Sender: text lines as
    the raw thread, so stage 1b can cross-check the split against ground
    truth it's given alongside (see distill_thread)."""
    by_id = {m["msg_id"]: m for m in thread}
    lines: list[str] = []
    for b in branches:
        topic = (b.get("topic") or "").strip()
        lines.append(f"### {topic}" if topic else "### (без темы)")
        for mid in b.get("message_ids") or []:
            m = by_id.get(mid)
            if m:
                sender = m.get("sender") or "Аноним"
                lines.append(f"[{mid}] {sender}: {m['text']}")
    return "\n".join(lines)


def _items_to_units(items: list[dict], thread: list[dict], chat: dict) -> list[dict]:
    """Shared post-processing: raw {"question","answer","type"} dicts from
    either extraction path -> validated knowledge units. Identical for
    distill_thread and distill_thread_one_stage so an A/B comparison isolates
    the extraction call itself, not this filtering."""
    root = thread[0]
    # latest activity in the thread — used later for recency weighting
    thread_date = thread[-1].get("date") or root.get("date") or ""
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not q or not a:
            continue
        if _is_non_answer(a):
            continue  # model stated "no info" instead of omitting the pair (rule #3)
        ktype = (it.get("type") or "vote_based").strip().lower()
        if ktype not in {"date_based", "vote_based"}:
            ktype = "vote_based"
        city = (it.get("city") or "").strip() or None
        if city and city.lower() in {"null", "none", "-"}:
            city = None  # model sometimes writes the literal word instead of JSON null
        out.append(
            {
                "question": q,
                "answer": a,
                "type": ktype,
                "city": city,
                "date": thread_date,
                "chat_username": chat["username"],
                "chat_title": chat["title"],
                "root_msg_id": root["msg_id"],
                "root_link": root["link"],
            }
        )
    return out


def distill_thread(thread: list[dict], chat: dict) -> list[dict]:
    """Return a list of knowledge units for one thread (possibly empty).

    Stage 1 is two sequential calls: config.SPLIT_MODEL first groups the
    thread into semantic branches by message id (_split_branches);
    config.EXTRACT_MODEL then extracts knowledge using SYSTEM_PROMPT,
    unchanged. The raw thread is always passed as ground truth alongside the
    branch split, which is only a draft the extraction step may correct — it
    is never used as truth on its own (a thread can mix unrelated topics; the
    split can be wrong).
    """
    raw_text = _thread_text_with_ids(thread)
    branches = _split_branches(thread)
    branches_text = _branches_text(thread, branches) if branches else ""
    user_content = (
        (
            f"ИСХОДНЫЙ ТРЕД (источник истины — опирайся на него):\n{raw_text}\n\n"
            f"ЧЕРНОВОЕ РАЗДЕЛЕНИЕ НА ВЕТКИ (может содержать ошибки, это только "
            f"подсказка, не факт):\n{branches_text}"
        )
        if branches_text
        else raw_text  # split call failed/empty — fall back to raw-only, as before
    )

    data = chat_json(
        config.EXTRACT_MODEL, SYSTEM_PROMPT, user_content,
        temperature=0, reasoning_effort=config.EXTRACT_REASONING_EFFORT,
    )
    return _items_to_units(data.get("knowledge", []), thread, chat)


def distill_thread_one_stage(thread: list[dict], chat: dict) -> list[dict]:
    """A/B comparison variant: config.EXTRACT_MODEL extracts DIRECTLY from the
    raw thread, no branch-split call first (the split stage skipped entirely,
    not just ignored). Same SYSTEM_PROMPT, same post-processing
    (_items_to_units) as distill_thread — the only difference under test is
    whether the split call helps. Not used by distill_chat/the main pipeline;
    call it directly to compare against distill_thread on the same threads.
    """
    data = chat_json(
        config.EXTRACT_MODEL, SYSTEM_PROMPT, _thread_text_with_ids(thread),
        temperature=0, reasoning_effort=config.EXTRACT_REASONING_EFFORT,
    )
    return _items_to_units(data.get("knowledge", []), thread, chat)


def select_threads(
    username: str, *, limit: int | None = None, min_thread_size: int = 2
) -> list[list[dict]]:
    """The exact thread selection distill_chat will process, exposed so you can
    inspect *which* threads "the first N" actually refers to before spending
    tokens on them.

    Threads are in ascending root-msg_id order (oldest root first). Threads
    whose latest message is older than config.INGEST_SINCE are dropped first:
    those live in the parent-lookback tail and exist only to give reply context
    to threads that are still active in the trusted window. `limit` is applied
    AFTER that drop, so "first `limit` threads" means first among the survivors,
    not first overall.
    """
    chat = next((c for c in config.CHATS if c["username"] == username), None)
    if chat is None:
        raise SystemExit(f"{username} is not in config.CHATS")

    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found — run src.ingest first")

    msgs, _ = filter_spam(_load_raw(raw_path)) if config.FILTER_SPAM else (_load_raw(raw_path), [])
    since = config.ingest_since_dt()
    threads = [t for t in build_threads(msgs) if len(t) >= min_thread_size]
    if since is not None:
        before = len(threads)
        threads = [
            t for t in threads
            if (d := _thread_latest_dt(t)) is None or d >= since
        ]
        dropped = before - len(threads)
        if dropped:
            print(f"[knowledge] {username}: skipped {dropped} threads older than {since.date()}")
    if limit is not None:
        threads = threads[:limit]
    return threads


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
            See select_threads() for exactly which threads that means.
        min_thread_size: skip threads with fewer messages (default 2 = only
            discussions; set 1 to also distill standalone informative messages).
        write: also write data/knowledge/<username>.jsonl.
    """
    chat = next((c for c in config.CHATS if c["username"] == username), None)
    threads = select_threads(username, limit=limit, min_thread_size=min_thread_size)

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
