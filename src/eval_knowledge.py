"""Validate the knowledge distillation step (src/knowledge.py) at scale.

Reading every distilled Q&A by hand doesn't scale, so an LLM (config.JUDGE_MODEL,
separate from the distiller) checks each unit against its source thread and
flags the bad ones. Two passes:

  precision — ONE call per thread (judge_thread), judging every knowledge unit
              extracted from it together: is each answer supported by the
              thread (no invention), is it one atomic topic, is it durable
              reference knowledge, is the `type` label right? -> verdict
              keep/fix/drop per unit.

  recall    — sample threads that produced NOTHING and ask whether reusable
              knowledge was actually there and got missed.

Outputs (next to data/knowledge/<chat>.jsonl):
  <chat>.eval.jsonl    — one judgement per knowledge unit
  <chat>.recall.jsonl  — one judgement per sampled empty thread
plus a summary to stdout and a dump of everything not marked "keep".

Run:  uv run python -m src.eval_knowledge helpgeorgia
      uv run python -m src.eval_knowledge helpgeorgia --limit 40 --recall-sample 30

⚠️ Spends OpenAI tokens: one JUDGE_MODEL call per thread (not per unit) + one
per sampled empty thread for recall.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Judge calls are independent and I/O-bound; run a handful in parallel so a full
# pass is minutes, not tens of minutes.
JUDGE_WORKERS = 8

import config
from src.knowledge import _thread_latest_dt, _thread_text
from src.preprocess import _load_raw
from src.spam import filter_spam
from src.store import chat_json
from src.threads import build_threads

# --- Judge prompts (Russian: the content and the labels are Russian) ----------

PRECISION_SYSTEM = (
    "Ты — строгий редактор справочной базы знаний о жизни в Грузии. На вход: "
    "исходный тред Telegram-чата и СПИСОК пар «вопрос-ответ», извлечённых из "
    "него автоматически. Оцени КАЖДУЮ пару по отдельности, ТОЛЬКО по этому "
    "треду.\n\n"
    "⚠️ Тред может содержать НЕСКОЛЬКО параллельных вопросов от разных людей "
    "вперемешку (собран по времени и реплаям, а не по теме) — определи, какие "
    "реплики отвечают именно на оцениваемый вопрос. Если тред явно не называет "
    "деталь (город, дату) — не изобретай её сам по фоновым знаниям о Грузии, "
    "пиши, что деталь не подтверждена.\n\n"
    "Критерии:\n"
    "1. faithful — каждое утверждение в ответе подтверждается тредом. "
    "Целевая рекомендация в ответ на запрос с критериями сама подтверждает "
    "соответствие этим критериям — не требуй дословного повтора каждого "
    "критерия отдельно. Додумывание — это добавление конкретных фактов, "
    "которых в треде вообще нет, а не разумный вывод из контекста вопроса.\n"
    "2. atomic — ответ про ОДНУ тему.\n"
    "3. useful — долговечная справка (процедуры, документы, «где/как», "
    "рекомендации, мнения, конкретные цены на момент ответа). НЕ useful: "
    "разовое объявление, событие на конкретную дату («открыт ли каньон "
    "сегодня»), «информации нет», пересказ вопроса. Устареет ли факт со "
    "временем — НЕ повод для useful=false, для этого есть type=date_based. "
    "То, что ответ дал ОДИН человек / это его личное мнение / это не "
    "профессиональный врач / это совет непроверенной квалификации — тоже НЕ "
    "повод для useful=false: справка от одного источника всё равно полезна, "
    "консенсус и надёжность источника — это отдельный вопрос ранжирования при "
    "поиске, а не критерий, входит ли пара в базу знаний вообще. Пример: «Я "
    "тренер по фитнесу, орбитрек лучше для суставов, чем дорожка» — useful=true "
    "(это мнение практика по теме вопроса), даже если это не врач и мнение "
    "пока единственное.\n"
    "4. type_suggested — какой `type` ПРАВИЛЬНЫЙ для этой пары:\n"
    "   - date_based — официально устанавливаемое (законы, налоги, визовые/"
    "таможенные требования, документы, тарифы) И любая конкретная ЦЕНА/СУММА;\n"
    "   - vote_based — всё остальное долговечное: рекомендация ГДЕ/У КОГО/"
    "КАКИМ СПОСОБОМ без суммы в ответе, а также вневременные факты. Например: "
    "«где обменять валюту» и «где оформить документ» — vote_based (это вопрос "
    "МЕСТА, не суммы и не самого требования); «сколько стоит обменять валюту», "
    "«какие документы нужны для X» — date_based.\n"
    "   Не выбирай date_based только потому что что-то теоретически может "
    "измениться — это верно почти для всего.\n"
    "5. city_suggested — какой ГОРОД правильный для этой пары: конкретный город, "
    "ЕСЛИ вопрос/ответ реально о конкретном месте (и тред явно его называет), "
    "иначе null (знание общегрузинское, не привязано к одному месту). НЕ "
    "исправляй/угадывай city по своим фоновым знаниям о географии Грузии — "
    "только по тому, что явно написано в треде.\n\n"
    "Не указывай type_ok/city_ok/verdict — это вычисляется отдельно, сравнением "
    "твоих type_suggested/city_suggested с тем, что было дано в паре. Твоя "
    "задача — только назвать правильные значения и объяснить в note, если "
    "заданные отличаются от них.\n\n"
    "Для КАЖДОЙ пары сначала напиши `note` (что не так, если что-то не так), и "
    "только потом остальные поля — они обязаны совпадать с note.\n"
    "Ответь строго JSON: {\"judgments\": [{\"index\": 0, \"note\": \"...\", "
    "\"faithful\": bool, \"atomic\": bool, \"useful\": bool, "
    "\"type_suggested\": \"date_based|vote_based\", "
    "\"city_suggested\": \"Тбилиси|null\"}, "
    "...]} — ровно один объект на каждую входную пару, `index` = номер пары "
    "во входном списке (с нуля), в том же порядке."
)

RECALL_SYSTEM = (
    "Ты проверяешь ПОЛНОТУ извлечения знаний. На вход — тред Telegram-чата о "
    "жизни в Грузии, из которого автоматический экстрактор НЕ извлёк ни одной "
    "пары «вопрос-ответ». Проверь, правильно ли это.\n\n"
    "Долговечное знание — это справочная информация, полезная многим и надолго: "
    "процедуры, документы, как что устроено, общие советы «где/как сделать X», "
    "рекомендации специалистов/мест с контактами. НЕ считается: болтовня, "
    "приветствия, споры без вывода, разовые объявления (продажа вещи, билет, "
    "пристройство животного, «кто едет»), вопросы без ответа.\n\n"
    "Ответь строго JSON: {\"has_knowledge\": bool, \"missed\": \"какую пару "
    "вопрос-ответ можно было извлечь (или пусто)\"}"
)


def _judge(system: str, user: str) -> dict:
    return chat_json(
        config.JUDGE_MODEL, system, user,
        temperature=0, reasoning_effort=config.JUDGE_REASONING_EFFORT,
    )


# --- Rebuild the source threads, keyed by root message id ---------------------

def _threads_by_root(username: str, *, min_thread_size: int = 2) -> dict[int, list[dict]]:
    """Rebuild threads exactly like src.knowledge.distill_chat does, so each
    knowledge unit's root_msg_id maps back to the thread it came from."""
    raw_path = config.RAW_DIR / f"{username}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found — run src.ingest first")
    msgs = _load_raw(raw_path)
    if config.FILTER_SPAM:
        msgs, _ = filter_spam(msgs)
    since = config.ingest_since_dt()
    threads = [t for t in build_threads(msgs) if len(t) >= min_thread_size]
    if since is not None:
        threads = [
            t for t in threads
            if (d := _thread_latest_dt(t)) is None or d >= since
        ]
    return {t[0]["msg_id"]: t for t in threads}


def _load_knowledge(username: str) -> list[dict]:
    path = config.KNOWLEDGE_DIR / f"{username}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run src.knowledge first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Pass 1: precision ------------------------------------------------------

def _pairs_block(units: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] Вопрос: {u['question']}\nОтвет: {u['answer']}\nТип: {u['type']}\n"
        f"Город: {u.get('city') or 'null'}"
        for i, u in enumerate(units)
    )


def _norm_city(c) -> str | None:
    c = (c or "").strip()
    if not c or c.lower() in {"null", "none", "-"}:
        return None
    return c


def judge_thread(thread: list[dict], units: list[dict]) -> list[dict]:
    """Judge ALL knowledge units extracted from ONE thread in a SINGLE call
    (config.JUDGE_MODEL) — cheaper than one call per unit and avoids resending
    the same thread text once per unit. Returns judgments in the same order
    as `units`.

    The model only reports the raw signals (faithful/atomic/useful and what
    it thinks type/city SHOULD be) — type_ok, city_ok and verdict are computed
    HERE, deterministically, rather than asked of the model. Earlier versions
    had the model self-report all of these and it repeatedly produced
    self-contradictory JSON (e.g. type_ok=true with a type_suggested that
    didn't match the given type) despite explicit prompt instructions not to.
    Computing the derived fields in code makes that class of bug structurally
    impossible instead of relying on the model to keep them consistent.
    """
    if not units:
        return []
    user = f"ТРЕД:\n{_thread_text(thread)}\n\nПАРЫ:\n{_pairs_block(units)}"
    data = _judge(PRECISION_SYSTEM, user)
    by_index = {j.get("index"): j for j in (data.get("judgments") or []) if isinstance(j, dict)}
    out = []
    for i, u in enumerate(units):
        j = by_index.get(i, {})
        faithful, atomic, useful = j.get("faithful"), j.get("atomic"), j.get("useful")
        type_suggested = j.get("type_suggested")
        city_suggested = _norm_city(j.get("city_suggested"))
        type_ok = (type_suggested == u["type"]) if type_suggested else None
        city_ok = city_suggested == _norm_city(u.get("city"))

        if faithful is False or useful is False:
            verdict = "drop"
        elif faithful is True and useful is True and atomic is True and type_ok is True and city_ok is True:
            verdict = "keep"
        elif faithful is True and useful is True:
            verdict = "fix"
        else:
            verdict = "error"  # judge didn't return faithful/useful for this pair

        out.append({
            **_unit_ref(u),
            "faithful": faithful,
            "atomic": atomic,
            "useful": useful,
            "type_ok": type_ok,
            "type_suggested": type_suggested,
            "city_ok": city_ok,
            "city_suggested": city_suggested,
            "verdict": verdict,
            "note": (j.get("note") or "").strip(),
        })
    return out


def judge_units(
    units: list[dict], threads_by_root: dict[int, list[dict]], *, label: str = "eval:precision"
) -> list[dict]:
    """Judge a list of knowledge units against their source threads, batched
    ONE call PER THREAD (judge_thread) rather than one call per unit. This is
    the reusable core of eval_precision — call it directly from a notebook to
    judge an in-memory batch (e.g. a cheap distill_chat(limit=..., write=False)
    prototype run) without touching data/knowledge/*.jsonl at all.

    Output preserves the input `units` order (not grouped by thread), so it
    still zips 1:1 with `units` the way src.fix_knowledge.fix_batch expects.
    """
    groups: dict[int, list[int]] = {}
    for idx, u in enumerate(units):
        groups.setdefault(u["root_msg_id"], []).append(idx)

    def judge_group(root_msg_id: int) -> list[tuple[int, dict]]:
        idxs = groups[root_msg_id]
        group_units = [units[i] for i in idxs]
        thread = threads_by_root.get(root_msg_id)
        if thread is None:
            # No matching source thread (e.g. code or spam patterns changed
            # since, or you're judging a batch built from different threads).
            rows = [
                {**_unit_ref(units[i]), "verdict": "skip", "note": "исходный тред не найден"}
                for i in idxs
            ]
        else:
            rows = judge_thread(thread, group_units)
        return list(zip(idxs, rows))

    out: list[dict | None] = [None] * len(units)
    for pairs in _run_parallel(judge_group, list(groups.keys()), label=label):
        for idx, row in pairs:
            out[idx] = row
    return out


def eval_precision(
    username: str, *, limit: int | None = None, write: bool = True
) -> list[dict]:
    units = _load_knowledge(username)
    if limit is not None:
        units = random.sample(units, min(limit, len(units)))
    threads = _threads_by_root(username)

    out = judge_units(units, threads, label=f"eval:precision {username}")

    if write:
        _dump(config.KNOWLEDGE_DIR / f"{username}.eval.jsonl", out)
    _report_precision(out)
    return out


def _unit_ref(u: dict) -> dict:
    return {
        "root_msg_id": u["root_msg_id"],
        "root_link": u["root_link"],
        "question": u["question"],
        "answer": u["answer"],
        "type": u["type"],
        "city": u.get("city"),
    }


def _report_precision(rows: list[dict]) -> None:
    judged = [r for r in rows if r["verdict"] in {"keep", "fix", "drop"}]
    n = len(judged) or 1
    print(f"\n=== precision: {len(judged)} judged "
          f"({len(rows) - len(judged)} skipped/errored) ===")
    for key in ("faithful", "atomic", "useful", "type_ok", "city_ok"):
        ok = sum(1 for r in judged if r.get(key) is True)
        print(f"  {key:9}: {ok:4}/{n}  ({100 * ok // n}%)")
    print("  verdict   :", dict(Counter(r["verdict"] for r in judged)))

    flagged = [r for r in rows if r["verdict"] in {"fix", "drop", "skip", "error"}]
    print(f"\n--- {len(flagged)} units to review (verdict != keep) ---")
    for r in flagged:
        print(f"\n[{r['verdict']}] {r['root_link']}   note: {r.get('note', '')}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:200]}")


# --- Pass 2: recall -------------------------------------------------------

def eval_recall(
    username: str, *, sample: int = 30, write: bool = True
) -> list[dict]:
    units = _load_knowledge(username)
    produced = {u["root_msg_id"] for u in units}
    # Distillation processes threads in ascending root-id order and stops at a
    # limit, so anything past the largest processed root was never seen. Only
    # judge empty threads inside the processed range.
    cutoff = max(produced)
    threads = _threads_by_root(username)
    empty = [
        t for rid, t in threads.items()
        if rid <= cutoff and rid not in produced
    ]
    random.shuffle(empty)
    empty = empty[:sample]

    def judge_one(thread: list[dict]) -> dict:
        j = _judge(RECALL_SYSTEM, f"ТРЕД:\n{_thread_text(thread)}")
        root = thread[0]
        return {
            "root_msg_id": root["msg_id"],
            "root_link": root["link"],
            "thread_size": len(thread),
            "has_knowledge": j.get("has_knowledge"),
            "missed": (j.get("missed") or "").strip(),
        }

    out = _run_parallel(judge_one, empty, label=f"eval:recall {username}")

    if write:
        _dump(config.KNOWLEDGE_DIR / f"{username}.recall.jsonl", out)
    _report_recall(out, total_empty=len([
        rid for rid in threads if rid <= cutoff and rid not in produced
    ]))
    return out


def _report_recall(rows: list[dict], *, total_empty: int) -> None:
    missed = [r for r in rows if r.get("has_knowledge") is True]
    n = len(rows) or 1
    print(f"\n=== recall: {len(rows)} empty threads sampled "
          f"(of {total_empty} in range) ===")
    print(f"  had extractable knowledge that was MISSED: {len(missed)}/{n} "
          f"({100 * len(missed) // n}%)")
    for r in missed:
        print(f"\n  {r['root_link']}  ({r['thread_size']} msgs)")
        print(f"    missed: {r['missed']}")


# --- Calibration: trust the judge only after checking it against yourself ----

def sample_for_calibration(
    username: str, *, n: int = 12, seed: int = 0
) -> list[dict]:
    """Pick a spread of units (by judge verdict) with their source thread text,
    for you to hand-label keep/fix/drop in the notebook. Requires that
    eval_precision has been run (reads <chat>.eval.jsonl)."""
    eval_path = config.KNOWLEDGE_DIR / f"{username}.eval.jsonl"
    if not eval_path.exists():
        raise SystemExit(f"{eval_path} not found — run eval_precision first")
    with eval_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    threads = _threads_by_root(username)

    rnd = random.Random(seed)
    by_verdict: dict[str, list[dict]] = {}
    for r in rows:
        by_verdict.setdefault(r["verdict"], []).append(r)
    picked: list[dict] = []
    per = max(1, n // max(1, len(by_verdict)))
    for group in by_verdict.values():
        picked.extend(rnd.sample(group, min(per, len(group))))
    picked = picked[:n]

    for r in picked:
        t = threads.get(r["root_msg_id"])
        r["thread_text"] = _thread_text(t) if t else ""
    return picked


def calibration_report(picked: list[dict], manual: dict[int, str]) -> None:
    """Compare your labels to the judge's. `manual` maps root_msg_id -> your
    verdict ('keep'/'fix'/'drop')."""
    pairs = [(r, manual[r["root_msg_id"]]) for r in picked if r["root_msg_id"] in manual]
    if not pairs:
        print("no overlap between picked units and manual labels")
        return
    agree = sum(1 for r, m in pairs if r["verdict"] == m)
    # keep vs (fix|drop) — the decision that actually matters
    binary = sum(
        1 for r, m in pairs
        if (r["verdict"] == "keep") == (m == "keep")
    )
    print(f"exact agreement:      {agree}/{len(pairs)} ({100 * agree // len(pairs)}%)")
    print(f"keep/not-keep agree:  {binary}/{len(pairs)} ({100 * binary // len(pairs)}%)")
    for r, m in pairs:
        mark = "OK " if r["verdict"] == m else "!! "
        print(f"  {mark} judge={r['verdict']:5} you={m:5}  {r['root_link']}")


# --- plumbing ------------------------------------------------------------

def _run_parallel(fn, items: list, *, label: str) -> list[dict]:
    """Map fn over items with a small thread pool, keeping input order and
    printing progress every 25 completions."""
    total = len(items)
    done = 0
    results: list[dict | None] = [None] * total
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        futures = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[{label}] {done}/{total}")
    return [r for r in results if r is not None]


def _dump(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(rows)} rows -> {path}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(
            'Usage: python -m src.eval_knowledge <username> '
            '[--limit N] [--recall-sample N] [--no-recall]'
        )
    username = args[0]
    limit = _arg_int(args, "--limit")
    recall_sample = _arg_int(args, "--recall-sample") or 30
    do_recall = "--no-recall" not in args

    eval_precision(username, limit=limit)
    if do_recall:
        eval_recall(username, sample=recall_sample)


def _arg_int(args: list[str], flag: str) -> int | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return int(args[i + 1])
    return None


if __name__ == "__main__":
    main()
