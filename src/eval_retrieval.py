"""Evaluate retrieval quality (src/retrieve.py::search) against the hand-curated
golden query set (eval/golden_queries.jsonl), and sweep RETRIEVAL_MIN_SCORE
candidates on the SAME judged results — no re-querying/re-judging per
threshold, since relevance is judged once against an uncapped candidate list.

Reference-free: no pre-labeled "correct chunk per query" — an LLM judge scores
each retrieved candidate's relevance to the query on the fly (like ragas's
context_precision without a reference), following src.eval_knowledge's
established pattern: one call per query judging ALL its candidates together,
deterministic aggregation in Python (never trust the model's own summary).

Run:  uv run python -m src.eval_retrieval [--limit N] [--category C]

⚠️ Spends OpenAI tokens: one embedding + one config.JUDGE_MODEL call per query
(~2 calls/query — see config.EVAL_DIR/golden_queries.jsonl for the current set
size). The threshold sweep itself is pure Python, zero extra calls.
"""
from __future__ import annotations

import sys

import config
from src.eval_common import load_golden
from src.eval_knowledge import _dump, _run_parallel
from src.retrieve import search
from src.store import chat_json

CANDIDATES_SYSTEM = (
    "Ты оцениваешь качество поиска для RAG-системы про жизнь в Грузии. На "
    "вход: вопрос пользователя и пронумерованный список найденных фрагментов "
    "(вопрос-ответ из Telegram-чатов). Для КАЖДОГО фрагмента реши: помог бы "
    "он реально ответить на вопрос (полностью или частично) — relevant=true, "
    "или он просто похож по теме/словам, но не отвечает на суть — "
    "relevant=false. Не занижай relevant только потому что это лишь один из "
    "нескольких вариантов ответа (места/способы/мнения) — это нормально для "
    "vote_based знаний. Если вопрос по сути про ДРУГУЮ страну, а фрагмент — "
    "про Грузию на похожую тему (например, права/страховка) — relevant=false, "
    "это не тот же вопрос.\n\n"
    "Сначала напиши `note` (кратко почему), потом `relevant`.\n"
    "Ответь строго JSON: {\"judgments\": [{\"index\": 0, \"note\": \"...\", "
    "\"relevant\": bool}, ...]} — ровно один объект на каждый фрагмент, "
    "`index` = номер фрагмента во входном списке (с нуля), в том же порядке."
)


def _judge(system: str, user: str) -> dict:
    return chat_json(
        config.JUDGE_MODEL, system, user,
        temperature=0, reasoning_effort=config.JUDGE_REASONING_EFFORT,
    )


def _candidates_block(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] Вопрос: {h['meta']['question']}\nОтвет: {h['meta']['answer']}"
        for i, h in enumerate(hits)
    )


def judge_candidates(query: str, hits: list[dict]) -> list[dict]:
    """Judge ALL retrieved candidates for ONE query in a SINGLE call (mirrors
    src.eval_knowledge.judge_thread's one-call-per-group pattern). Returns
    [{"relevant": bool|None, "note": str}, ...] in the same order as `hits`."""
    if not hits:
        return []
    user = f"ВОПРОС:\n{query}\n\nФРАГМЕНТЫ:\n{_candidates_block(hits)}"
    data = _judge(CANDIDATES_SYSTEM, user)
    by_index = {j.get("index"): j for j in (data.get("judgments") or []) if isinstance(j, dict)}
    out = []
    for i in range(len(hits)):
        j = by_index.get(i, {})
        out.append({"relevant": j.get("relevant"), "note": (j.get("note") or "").strip()})
    return out


def collect_run(golden: list[dict] | None = None, *, k: int = config.RETRIEVAL_EVAL_K, write: bool = True) -> list[dict]:
    """Per golden query: search(query, k=k, min_score=0.0) ONCE (uncapped, so
    every threshold in sweep_thresholds is just a filter over this one ranked
    list) -> judge_candidates ONCE. Parallelized over queries.

    Returns [{**golden_row, "hits": [{**hit, "relevant", "note"}, ...]}, ...],
    dumped to data/eval/retrieval_judged.jsonl."""
    golden = golden if golden is not None else load_golden()

    def process(row: dict) -> dict:
        hits = search(row["query"], k=k, min_score=0.0)
        judged = judge_candidates(row["query"], hits)
        judged_hits = [{**h, **j} for h, j in zip(hits, judged)]
        return {**row, "hits": judged_hits}

    rows = _run_parallel(process, golden, label="eval:retrieval")
    if write:
        _dump(config.EVAL_DATA_DIR / "retrieval_judged.jsonl", rows)
    return rows


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def sweep_thresholds(rows: list[dict], thresholds: list[float] | None = None) -> list[dict]:
    """PURE PYTHON, zero API calls — filters each query's cached judged hits
    by score per threshold candidate. Per threshold:
      precision        — mean, over queries with >=1 retained hit, of
                          (relevant retained / retained)
      recall           — mean, over queries with >=1 relevant candidate at
                          all (within the full k judged), of
                          (relevant retained / total relevant)
      hit_rate         — fraction of non-no_answer queries retaining >=1
                          relevant hit
      avg_hits         — mean retained-hit count across all queries
      no_answer_leak   — fraction of no_answer queries that STILL retain a
                          judge-relevant hit (should be ~0 — a low threshold
                          failing to say "no answer" when there truly isn't one)
      critical_hit_rate — fraction of critical queries (EXCLUDING
                          subtype=="scope_leakage") retaining >=1 relevant
                          hit. scope_leakage rows are deliberate traps where
                          the corpus has nothing genuinely relevant BY
                          DESIGN (a Georgia-specific fragment on a similar
                          topic isn't a real match for a foreign-country
                          question) — success there is 0 hits, checked by
                          src.eval_rag's scope_ok instead, not this metric.
    """
    thresholds = thresholds if thresholds is not None else config.RETRIEVAL_EVAL_THRESHOLDS
    out = []
    for t in thresholds:
        precisions, recalls, avg_hits_counts = [], [], []
        hit_flags, no_answer_leaks, critical_hits = [], [], []
        for row in rows:
            retained = [h for h in row["hits"] if h["score"] >= t]
            avg_hits_counts.append(len(retained))
            total_relevant = sum(1 for h in row["hits"] if h.get("relevant"))
            retained_relevant = sum(1 for h in retained if h.get("relevant"))
            if retained:
                precisions.append(retained_relevant / len(retained))
            if total_relevant > 0:
                recalls.append(retained_relevant / total_relevant)
            if row["category"] == "no_answer":
                no_answer_leaks.append(1 if retained_relevant > 0 else 0)
            else:
                hit_flags.append(1 if retained_relevant > 0 else 0)
            if row["category"] == "critical" and row.get("subtype") != "scope_leakage":
                critical_hits.append(1 if retained_relevant > 0 else 0)
        precision, recall = _mean(precisions), _mean(recalls)
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        out.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit_rate": _mean(hit_flags),
            "avg_hits": _mean(avg_hits_counts),
            "no_answer_leak": _mean(no_answer_leaks),
            "critical_hit_rate": _mean(critical_hits),
        })
    return out


def sweep_grid(
    rows: list[dict], ks: list[int] | None = None, thresholds: list[float] | None = None,
) -> list[dict]:
    """2D sweep over BOTH candidate pool size (k, i.e. config.TOP_K candidates)
    AND score threshold (config.RETRIEVAL_MIN_SCORE) — still zero extra API
    calls. search() bounds how many candidates Chroma even returns (via `k`)
    BEFORE min_score filters them, so k and the threshold interact: a small k
    can cut off a candidate that would have passed a generous threshold.
    Chroma already returns hits sorted best-first, so "only the top k" is
    just row["hits"][:k] over the same cached, already-judged k=RETRIEVAL_EVAL_K
    pool collect_run fetched — no re-querying, re-embedding, or re-judging."""
    ks = ks if ks is not None else [4, 6, 8, 10, 15, 20]
    thresholds = thresholds if thresholds is not None else config.RETRIEVAL_EVAL_THRESHOLDS
    out = []
    for k in ks:
        capped = [{**row, "hits": row["hits"][:k]} for row in rows]
        for r in sweep_thresholds(capped, thresholds):
            out.append({"k": k, **r})
    return out


def _fmt(x) -> str:
    return f"{x:.2f}" if x is not None else "  - "


def _pick_best(candidate_rows: list[dict]) -> dict | None:
    """Decision rule, in priority order (see module docstring / notebook
    section 15 for why): critical_hit_rate must be 1.0 (a critical query with
    no relevant hit at all has zero chance downstream, no matter the prompt)
    > no_answer_leak must be 0 (never confidently answer something the corpus
    doesn't actually have) > among what's left, maximize F1 (precision/recall
    balance) > tie-break on fewer avg_hits (cheaper, less noise for the
    generator to sort through)."""
    ok = [
        r for r in candidate_rows
        if (r["no_answer_leak"] or 0) == 0 and (r["critical_hit_rate"] in (None, 1.0))
    ]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["f1"] or 0, -r["avg_hits"]))


def _report_grid(grid_rows: list[dict]) -> None:
    print(f"\n{'k':>3} | {'thr':>5} | {'precision':>9} | {'recall':>7} | {'f1':>5} | "
          f"{'hit_rate':>8} | {'avg_hits':>8} | {'no_answer_leak':>14} | {'critical_hit_rate':>17}")
    for r in grid_rows:
        print(f"{r['k']:>3} | {r['threshold']:>5} | {_fmt(r['precision']):>9} | "
              f"{_fmt(r['recall']):>7} | {_fmt(r['f1']):>5} | {_fmt(r['hit_rate']):>8} | "
              f"{_fmt(r['avg_hits']):>8} | {_fmt(r['no_answer_leak']):>14} | "
              f"{_fmt(r['critical_hit_rate']):>17}")

    best = _pick_best(grid_rows)
    if best:
        print(f"\nСовет (не окончательный — смотри таблицу и провалы сама): "
              f"k={best['k']}, порог={best['threshold']} — no_answer_leak=0, "
              f"critical_hit_rate={best['critical_hit_rate']}, f1={_fmt(best['f1'])}, "
              f"avg_hits={_fmt(best['avg_hits'])}")
    else:
        print("\nНи одна комбинация (k, порог) не даёт одновременно no_answer_leak=0 и "
              "critical_hit_rate=1.0 — смотри таблицу и провалившиеся строки внимательно.")


def _arg_int(args: list[str], flag: str) -> int | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return int(args[i + 1])
    return None


def main() -> None:
    args = sys.argv[1:]
    limit = _arg_int(args, "--limit")
    category = None
    if "--category" in args:
        category = args[args.index("--category") + 1]
    golden = load_golden(category=category)
    if limit:
        golden = golden[:limit]
    rows = collect_run(golden)
    _report_grid(sweep_grid(rows))


if __name__ == "__main__":
    main()
