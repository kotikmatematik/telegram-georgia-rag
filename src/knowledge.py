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
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from src.ingest import _chat_link
from src.preprocess import _load_raw, _parse_dt
from src.spam import filter_spam
from src.store import chat_json
from src.threads import build_threads

# Same concurrency level as eval_knowledge.JUDGE_WORKERS — safe because
# src.store's openai_client()/get_collection() are lock-protected
# (threading.Lock, added after the eval pipeline's first concurrent access
# corrupted the Chroma tenant). Distillation is 2 sequential LLM calls per
# thread (split + extract) and was previously the only fully-sequential step
# of the pipeline — the dominant cost of a full run.
DISTILL_WORKERS = 8


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


# Contact masking used to happen here (baked into stored knowledge at
# collection time) — moved to src.rag._build_context, applied live per-
# fragment at retrieval time instead, based on config.CHATS' private flag.
# .fixed.jsonl now always stores the full, real, unmasked text for every
# chat — see src.fix_knowledge.save_fixed's docstring for why.

# Russian on purpose: source chats and the target assistant are Russian-speaking.
SYSTEM_PROMPT = (
    "Ты извлекаешь полезные ДОЛГОВЕЧНЫЕ знания о жизни в Грузии из переписок "
    "Telegram-чатов для справочного ассистента. На вход — один тред (обсуждение). "
    "Сформируй список пар «вопрос-ответ» в формате JSON.\n\n"
    "Правила:\n"
    "1. Опирайся ТОЛЬКО на то, что реально сказано в треде. Ничего не выдумывай — "
    "включая советы, предупреждения и оговорки «на всякий случай» (например "
    "«обратитесь к врачу», «не доверяйте советам из интернета»), которых в "
    "треде не было. Это касается и медицинских тем: если участники обсуждали "
    "лечение/симптомы, пиши только то, что реально написали, без добавления "
    "дисклеймеров от себя.\n"
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
    "подтверждают. Например: «где обменять валюту» и «где оформить документ» "
    "— vote_based (это вопрос МЕСТА/СПОСОБА, не суммы и не самого требования); "
    "«сколько стоит обменять валюту», «какие документы нужны для X» — "
    "date_based. Не выбирай date_based только потому что что-то теоретически "
    "может измениться — это верно почти для всего.\n"
    "8. `city` — конкретный город (Тбилиси/Батуми/Кутаиси/...), ЕСЛИ вопрос "
    "или ответ реально привязаны к одному городу. Указывай город ТОЛЬКО если "
    "он явно назван в треде — никогда не угадывай по намёкам (улицы, районы "
    "без названия города погоды не делают). Если знание общегрузинское "
    "(визы, налоги, ИП, общий совет не про конкретное место) — city: null.\n"
    "9. `source_msg_id` — id (число) ОДНОГО сообщения из треда (у каждой "
    "строки в треде есть свой [id] в начале), где был задан САМ ВОПРОС (не "
    "ответ на него) — так по ссылке видно вопрос и можно пролистать вниз все "
    "ответы на него. В треде до 50 сообщений — это НЕ обязательно первое "
    "сообщение треда, найди именно то, где спросили.\n"
    "10. Тред может дать несколько пар или ни одной. Если долговечного знания "
    "нет — верни {\"knowledge\": []}.\n"
    "11. Язык ответа — русский.\n\n"
    "Формат ответа строго: "
    "{\"knowledge\": [{\"question\": \"...\", \"answer\": \"...\", "
    "\"type\": \"date_based|vote_based\", \"city\": \"Тбилиси|null\", "
    "\"source_msg_id\": 12345}]}"
)


_GAP_MARKER_MINUTES = 30  # show a pause marker above this — 3x the burst gap,
# so it only fires for genuinely notable silences, not normal back-and-forth.


def _format_gap(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} мин"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f} ч"
    return f"{minutes / (60 * 24):.1f} дн"


def _thread_text(thread: list[dict]) -> str:
    """[id] sender: text, one line per message — tagged with "↩<id>" when the
    message is a reply, and preceded by a "— пауза N —" marker after a
    notable silence. Neither is visible otherwise (to the LLM or to a human
    skimming a mixed-topic thread), which then has to guess purely from
    wording which reply belongs to which message and where one topic ends
    and another begins."""
    lines = []
    prev_dt = None
    for m in thread:
        dt = _parse_dt(m.get("date"))
        if prev_dt and dt:
            gap_min = (dt - prev_dt).total_seconds() / 60
            if gap_min >= _GAP_MARKER_MINUTES:
                lines.append(f"— пауза {_format_gap(gap_min)} —")
        sender = m.get("sender") or "Аноним"
        reply = f" ↩{m['reply_to']}" if m.get("reply_to") else ""
        lines.append(f"[{m['msg_id']}]{reply} {sender}: {m['text']}")
        prev_dt = dt or prev_dt
    return "\n".join(lines)


# Stage 1a (new): split the raw thread into semantic branches before
# extraction. This is a DRAFT for stage 1b, not ground truth by itself — see
# distill_thread, which always keeps the raw thread as the authoritative
# source and passes the split only as a hint stage 1b may correct.
SPLIT_SYSTEM = (
    "Раздели тред Telegram-чата (каждое сообщение помечено [id], реплай — "
    "«↩id», пауза перед сообщением — «— пауза N —») на отдельные смысловые "
    "ветки — независимые темы/вопросы, которые в нём обсуждаются (тред мог "
    "собраться по времени и реплаям, а не по теме, поэтому внутри могут быть "
    "несвязанные разговоры — используй реплаи и паузы как подсказку о "
    "границах тем). Не меняй и не пересказывай текст сообщений — верни "
    "только id сообщений, входящих в каждую ветку, в исходном порядке.\n"
    "Ответь строго JSON: {\"branches\": [{\"topic\": \"кратко тема\", "
    "\"message_ids\": [id, id, ...]}]}"
)


def _split_branches(thread: list[dict]) -> list[dict]:
    """Stage 1a call: ask config.SPLIT_MODEL to group the thread's messages
    into semantic branches by id. Returns the raw branch list (possibly
    empty on failure — distill_thread then falls back to the raw thread alone)."""
    data = chat_json(
        config.SPLIT_MODEL, SPLIT_SYSTEM, _thread_text(thread),
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
                reply = f" ↩{m['reply_to']}" if m.get("reply_to") else ""
                lines.append(f"[{mid}]{reply} {sender}: {m['text']}")
    return "\n".join(lines)


def _items_to_units(items: list[dict], thread: list[dict], chat: dict) -> list[dict]:
    """Raw {"question","answer","type",...} dicts from the extraction call ->
    validated knowledge units."""
    root = thread[0]
    # latest activity in the thread — used later for recency weighting
    thread_date = thread[-1].get("date") or root.get("date") or ""
    thread_ids = {m["msg_id"] for m in thread}
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if not q or not a:
            continue
        if _is_non_answer(a):
            continue  # model stated "no info" instead of omitting the pair (rule #3)
        # Contacts are masked later (src.fix_knowledge.save_fixed), not here —
        # see _mask_contacts for why.
        ktype = (it.get("type") or "vote_based").strip().lower()
        if ktype not in {"date_based", "vote_based"}:
            ktype = "vote_based"
        city = (it.get("city") or "").strip() or None
        if city and city.lower() in {"null", "none", "-"}:
            city = None  # model sometimes writes the literal word instead of JSON null

        # rule #9: which message the answer actually came from — NOT the same
        # as root_msg_id in a long (up to 50-msg) thread, where the root can
        # be unrelated to a given pair. root_msg_id/root_link stay pointing
        # at the thread itself (used everywhere for grouping/judging); this
        # is a separate, more precise citation. Falls back to the thread root
        # if the model omits it or names a message outside this thread.
        try:
            source_msg_id = int(it.get("source_msg_id"))
        except (TypeError, ValueError):
            source_msg_id = None
        if source_msg_id not in thread_ids:
            source_msg_id = root["msg_id"]

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
                "source_msg_id": source_msg_id,
                "source_link": _chat_link(chat, source_msg_id),
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
    raw_text = _thread_text(thread)
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


def select_threads(
    username: str,
    *,
    limit: int | None = None,
    min_thread_size: int = 2,
    since=None,
) -> list[list[dict]]:
    """The exact thread selection distill_chat will process, exposed so you can
    inspect *which* threads "the first N" actually refers to before spending
    tokens on them.

    Threads are in ascending root-msg_id order (oldest root first). Threads
    whose latest message is older than `since` (default: config.INGEST_SINCE)
    are dropped first: those live in the parent-lookback tail and exist only
    to give reply context to threads that are still active in the trusted
    window. Pass `since` explicitly to select a different (e.g. more recent,
    overlap-adjusted) cutoff — see src/update_knowledge.py, which uses this to
    pick only threads worth re-checking on an incremental run. `limit` is
    applied AFTER the date drop, so "first `limit` threads" means first among
    the survivors, not first overall.
    """
    chat = next((c for c in config.CHATS if c["username"] == username), None)
    if chat is None:
        raise SystemExit(f"{username} is not in config.CHATS")

    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found — run src.ingest first")

    msgs, _ = filter_spam(_load_raw(raw_path)) if config.FILTER_SPAM else (_load_raw(raw_path), [])
    if since is None:
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


def _distill_thread_safe(thread: list[dict], chat: dict) -> tuple[list[dict], str | None]:
    """distill_thread wrapped so a single bad thread (e.g. Azure's content
    filter tripping on one message) can be skipped without killing the whole
    ThreadPoolExecutor — returns (units, error_note); error_note is None on
    success. Exceptions must not escape into as_completed()/fut.result(),
    or one bad thread would still kill the whole run, just via a different
    mechanism than the old sequential loop's try/except."""
    try:
        return distill_thread(thread, chat), None
    except Exception as e:
        root_id = thread[0]["msg_id"]
        return [], f"SKIPPED thread root={root_id} ({type(e).__name__}: {e})"


def distill_threads(
    threads: list[list[dict]], chat: dict, *, checkpoint_cb=None, checkpoint_every: int = 100
) -> list[dict]:
    """Distill an already-selected list of threads (shared by distill_chat and
    the incremental update path in src/update_knowledge.py, which only wants
    to (re)distill a subset, not the whole chat).

    Runs DISTILL_WORKERS threads concurrently — order of completion is not
    the input order, but that's fine: each knowledge unit already carries its
    own root_msg_id/source_msg_id, nothing downstream (checkpointing, resume,
    indexing) depends on output order. All accumulation and every
    checkpoint_cb call happen in this one (main) thread, driven by
    as_completed() — never inside a worker — so no locking is needed here.

    checkpoint_cb, if given, is called with the knowledge collected SO FAR
    every `checkpoint_every` COMPLETED threads (and once more at the very
    end) — see distill_chat, which uses this to persist partial progress on
    a long run instead of only writing once at the very end."""
    knowledge: list[dict] = []
    total = len(threads)
    done = 0
    with ThreadPoolExecutor(max_workers=DISTILL_WORKERS) as ex:
        futures = {ex.submit(_distill_thread_safe, t, chat): t for t in threads}
        for fut in as_completed(futures):
            units, error_note = fut.result()
            if error_note:
                print(f"[knowledge] {chat['username']}: {error_note}")
            knowledge.extend(units)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[knowledge] {chat['username']}: {done}/{total} threads -> {len(knowledge)} items")
            if checkpoint_cb and (done % checkpoint_every == 0 or done == total):
                checkpoint_cb(knowledge)
    return knowledge


def distill_chat(
    username: str,
    *,
    limit: int | None = None,
    min_thread_size: int = 2,
    write: bool = True,
    resume: bool = False,
    checkpoint_every: int = 100,
) -> list[dict]:
    """Distill ALL selected threads of one chat into knowledge units (full,
    from-scratch run — overwrites data/knowledge/<username>.jsonl entirely).
    For incremental (re-distill only recently-active threads, keep the rest
    untouched) use src.update_knowledge instead.

    Args:
        limit: process at most this many threads (for cheap prototype runs).
            See select_threads() for exactly which threads that means.
        min_thread_size: skip threads with fewer messages (default 2 = only
            discussions; set 1 to also distill standalone informative messages).
        write: also write data/knowledge/<username>.jsonl.
        resume: if data/knowledge/<username>.jsonl already exists, skip any
            thread whose root_msg_id already appears in it (from a previous,
            interrupted run) instead of redoing it. A thread that previously
            ran but produced ZERO items has no trace in that file, so it gets
            redone on resume — wasted tokens, not a correctness problem.
        checkpoint_every: with write=True, re-save the file every this many
            threads (not just once at the very end) so a crash mid-run loses
            at most this many threads of work — see distill_threads.
    """
    chat = next((c for c in config.CHATS if c["username"] == username), None)
    threads = select_threads(username, limit=limit, min_thread_size=min_thread_size)
    out_path = config.KNOWLEDGE_DIR / f"{username}.jsonl"

    prior: list[dict] = []
    if resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            prior = [json.loads(line) for line in f if line.strip()]
        done_roots = {k["root_msg_id"] for k in prior}
        threads = [t for t in threads if t[0]["msg_id"] not in done_roots]
        print(f"[knowledge] {username}: resuming — {len(done_roots)} threads already "
              f"checkpointed, {len(threads)} remaining")

    def save(partial: list[dict]) -> None:
        with out_path.open("w", encoding="utf-8") as f:
            for k in prior + partial:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")

    knowledge = distill_threads(
        threads, chat, checkpoint_cb=save if write else None, checkpoint_every=checkpoint_every,
    )
    all_knowledge = prior + knowledge

    if write:
        save(knowledge)
        print(f"[knowledge] saved {len(all_knowledge)} items -> {out_path}")
    return all_knowledge


def main() -> None:
    # Full run over all configured chats (costs tokens — see module docstring).
    for chat in config.CHATS:
        distill_chat(chat["username"])


if __name__ == "__main__":
    main()
