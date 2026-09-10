"""Validate the knowledge distillation step (src/knowledge.py) at scale.

Reading every distilled Q&A by hand doesn't scale, so a stronger LLM
(config.JUDGE_MODEL, separate from the distiller) checks each unit against its
source thread and flags the bad ones. Two passes:

  precision — for every knowledge unit: is the answer supported by the thread
              (no invention), is it one atomic topic, is it durable reference
              knowledge, is the `type` label right? -> verdict keep/fix/drop.

  recall    — sample threads that produced NOTHING and ask whether reusable
              knowledge was actually there and got missed.

Outputs (next to data/knowledge/<chat>.jsonl):
  <chat>.eval.jsonl    — one judgement per knowledge unit
  <chat>.recall.jsonl  — one judgement per sampled empty thread
plus a summary to stdout and a dump of everything not marked "keep".

Run:  uv run python -m src.eval_knowledge helpgeorgia
      uv run python -m src.eval_knowledge helpgeorgia --limit 40 --recall-sample 30

⚠️ Spends OpenAI tokens: one JUDGE_MODEL call per unit + per sampled thread.
For helpgeorgia (~220 units) this is a few hundred cheap calls.
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
from src.knowledge import _thread_text
from src.preprocess import _load_raw
from src.spam import filter_spam
from src.store import openai_client
from src.threads import build_threads

# --- Judge prompts (Russian: the content and the labels are Russian) ----------

PRECISION_SYSTEM = (
    "Ты — строгий редактор справочной базы знаний о жизни в Грузии. На вход: "
    "исходный тред Telegram-чата и одна пара «вопрос-ответ», извлечённая из него "
    "автоматически, с меткой типа. Оцени пару ТОЛЬКО по этому треду.\n\n"
    "Критерии:\n"
    "1. faithful — КАЖДОЕ утверждение в ответе реально подтверждается тредом. "
    "Если в ответе есть детали, которых в треде нет (додумано, обобщено сверх "
    "сказанного, перепутаны участники) — faithful=false.\n"
    "2. atomic — ответ про ОДНУ тему. Если склеены разные темы (например «права» "
    "и «стоматолог») — atomic=false.\n"
    "3. useful — это долговечная справка, полезная многим (процедуры, документы, "
    "устройство, общие советы «где/как»). Разовое объявление (продажа вещи, "
    "билет, пристройство животного, «кто едет Х числа»), «информации нет», или "
    "пересказ самого вопроса без ответа — useful=false.\n"
    "4. type_ok — метка типа верна: volatile (меняется со временем: законы, "
    "правила, налоги, цены, требования к документам, расписания), stable "
    "(рекомендации и контакты: врач, мастер, магазин, адреса), evergreen "
    "(история, география, культура, язык). Укажи type_suggested в любом случае.\n\n"
    "verdict:\n"
    "  keep — пара точная, атомарная, полезная, тип верный;\n"
    "  fix  — суть полезна, но есть правимый дефект (лишние детали, неверный "
    "тип, слегка размыто, стоит разделить);\n"
    "  drop — не подтверждается тредом, либо не долговечное знание, либо ответа "
    "по сути нет.\n\n"
    "Ответь строго JSON: {\"faithful\": bool, \"atomic\": bool, \"useful\": bool, "
    "\"type_ok\": bool, \"type_suggested\": \"volatile|stable|evergreen\", "
    "\"verdict\": \"keep|fix|drop\", \"note\": \"кратко, что не так (или пусто)\"}"
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
    client = openai_client()
    resp = client.chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


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
    threads = [t for t in build_threads(msgs) if len(t) >= min_thread_size]
    return {t[0]["msg_id"]: t for t in threads}


def _load_knowledge(username: str) -> list[dict]:
    path = config.KNOWLEDGE_DIR / f"{username}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found — run src.knowledge first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Pass 1: precision ------------------------------------------------------

def eval_precision(
    username: str, *, limit: int | None = None, write: bool = True
) -> list[dict]:
    units = _load_knowledge(username)
    if limit is not None:
        units = random.sample(units, min(limit, len(units)))
    threads = _threads_by_root(username)

    def judge_one(u: dict) -> dict:
        thread = threads.get(u["root_msg_id"])
        if thread is None:
            # Distillation ran on a thread set we can't reproduce (e.g. code or
            # spam patterns changed since). Skip rather than judge blind.
            return {**_unit_ref(u), "verdict": "skip", "note": "исходный тред не найден"}
        user = (
            f"ТРЕД:\n{_thread_text(thread)}\n\n"
            f"ПАРА:\nВопрос: {u['question']}\nОтвет: {u['answer']}\n"
            f"Тип: {u['type']}"
        )
        j = _judge(PRECISION_SYSTEM, user)
        return {
            **_unit_ref(u),
            "faithful": j.get("faithful"),
            "atomic": j.get("atomic"),
            "useful": j.get("useful"),
            "type_ok": j.get("type_ok"),
            "type_suggested": j.get("type_suggested"),
            "verdict": j.get("verdict", "error"),
            "note": (j.get("note") or "").strip(),
        }

    out = _run_parallel(judge_one, units, label=f"eval:precision {username}")

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
    }


def _report_precision(rows: list[dict]) -> None:
    judged = [r for r in rows if r["verdict"] in {"keep", "fix", "drop"}]
    n = len(judged) or 1
    print(f"\n=== precision: {len(judged)} judged "
          f"({len(rows) - len(judged)} skipped/errored) ===")
    for key in ("faithful", "atomic", "useful", "type_ok"):
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
