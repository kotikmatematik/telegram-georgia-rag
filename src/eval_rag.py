"""Evaluate RAG-answer quality (src/rag.py::answer) against the hand-curated
golden query set (eval/golden_queries.jsonl): faithfulness (no hallucination),
relevance, cross-country scope compliance, plus two deterministic (zero-LLM-
cost) checks for citation-format and date-hallucination regressions.

Follows src.eval_knowledge's established pattern: one judge call per query
(config.JUDGE_MODEL — same role as stage-3's judge_thread: a first-look judge
of freshly generated output, not stage-4's REVERIFY_MODEL, which specifically
re-checks a PRIOR verdict without seeing it). Verdict is computed in Python
from all signals — never trust the model's own aggregate verdict.

`category == "critical"` queries (subtype documents the real cost of getting
it wrong — legal/medical/financial/scope) are reported in a dedicated,
always-fully-printed block, never folded into the overall pass percentage.

Run:  uv run python -m src.eval_rag [--limit N] [--category C]

⚠️ Spends OpenAI tokens: one embedding + one config.GENERATION_MODEL call (the
system under test) + one config.JUDGE_MODEL call per query.
"""
from __future__ import annotations

import re
import sys
from collections import Counter

import config
from src.eval_common import load_golden
from src.eval_knowledge import _dump, _run_parallel
from src.rag import _CITE_NUM_RX, _MONTHS_RU, _build_context, answer
from src.retrieve import search
from src.store import chat_json

JUDGE_SYSTEM = (
    "Ты — независимый проверяющий ответов RAG-системы про жизнь в Грузии. "
    "На вход: вопрос, использованные фрагменты (пронумерованные) и готовый "
    "ответ ассистента. Оцени:\n"
    "1. faithful — каждая КОНКРЕТНАЯ деталь (место, документ, процедура, "
    "число, срок, причина), поданная как факт ИЗ ЧАТА, должна подтверждаться "
    "фрагментом или быть явно помечена как общее знание — придуманная "
    "конкретика без пометки — нарушение. НЕ считай нарушением: (а) разумное "
    "обобщение/вывод из уже показанных в ответе фактов (например «цена "
    "зависит от района», если ответ сам показывает разные цены в разных "
    "районах) — это синтез, а не новый факт; (б) фоновые фразы о самой "
    "системе поиска («в этих фрагментах нет...», «фрагменты только про "
    "Грузию») — это не утверждение о содержании, а честная оговорка о том, "
    "что нашёл поиск; (в) мягкий неспецифичный совет здравого смысла "
    "(«уточните на месте», «стоит иметь документ при себе»), если он не "
    "подан как факт из чата и не заменяет собой конкретную недостающую "
    "информацию.\n"
    "2. relevant — ответ по существу отвечает на заданный вопрос. ИСКЛЮЧЕНИЕ: "
    "если вопрос на самом деле не про Грузию (см. п.3) и ответ ЧЕСТНО "
    "отказался выдавать грузинские факты за ответ на него, явно объяснив "
    "несоответствие — это тоже relevant=true. Честный, по существу "
    "аргументированный отказ вне охвата системы — правильный ответ на такой "
    "вопрос, а не провал релевантности.\n"
    "3. scope_ok — если вопрос на самом деле про то, как что-то устроено в "
    "ДРУГОЙ стране — ответ должен честно отказаться выдавать грузинские "
    "факты за ответ на него; если другая страна лишь упомянута как "
    "направление/контрагент, а вопрос по сути про жизнь в Грузии — это тоже "
    "scope_ok=true.\n"
    "4. honest_uncertainty — ТОЛЬКО если источники в фрагментах реально "
    "противоречат друг другу или чего-то не хватает: признал ли ответ это "
    "явно, а не выбрал одну сторону как единственно верную. Если источники "
    "не противоречат — верни true (неприменимо, не повод для fail).\n\n"
    "Сначала напиши заметку по каждому пункту, потом сам булев вывод.\n"
    "Ответь строго JSON: {\"faithful_note\": \"...\", \"faithful\": bool, "
    "\"relevant_note\": \"...\", \"relevant\": bool, \"scope_note\": \"...\", "
    "\"scope_ok\": bool, \"honest_uncertainty_note\": \"...\", "
    "\"honest_uncertainty\": bool}"
)

# Reuses src.rag's own regexes/data so "leftover citation" and "date written by
# the model itself" are checked with the exact same patterns _inline_citations
# uses to build/consume them — no separate, potentially-drifting definitions.
_CITATION_PAREN_RX = re.compile(r"\([^()]*https?://\S+[^()]*\)")
_DATE_RX = re.compile(r"\b(" + "|".join(_MONTHS_RU) + r")\s+\d{4}\b")


def _judge(system: str, user: str) -> dict:
    return chat_json(
        config.JUDGE_MODEL, system, user,
        temperature=0, reasoning_effort=config.JUDGE_REASONING_EFFORT,
    )


def _citation_leftover(final_text: str) -> bool:
    """Deterministic, zero LLM cost. True = a [N]-shaped marker survived
    _inline_citations unresolved and is still visible in the final,
    user-facing answer — a citation-format regression."""
    return bool(_CITE_NUM_RX.search(final_text))


def _date_hallucination_suspect(final_text: str) -> bool:
    """Deterministic, zero LLM cost. Strip every citation parenthetical
    _inline_citations actually produces (they all contain a URL); any
    'месяц год' text left over after that was written by the MODEL itself in
    prose, not attached to a real citation — exactly the original bug."""
    stripped = _CITATION_PAREN_RX.sub("", final_text)
    return bool(_DATE_RX.search(stripped))


def judge_answer(query: str, hits: list[dict], final_text: str, *, is_critical: bool = False) -> dict:
    """ONE call (config.JUDGE_MODEL) judging faithful/relevant/scope_ok (and,
    for critical queries, honest_uncertainty) against the fragments actually
    used and the final answer, merged with the two deterministic checks
    above. `verdict`/`fails` are computed HERE, in Python, from all signals —
    matching src.eval_knowledge's rule of never trusting a model's own
    aggregate verdict."""
    context = _build_context(hits) if hits else "(пусто — ничего не нашлось)"
    user = f"ВОПРОС: {query}\n\nФРАГМЕНТЫ:\n{context}\n\nОТВЕТ АССИСТЕНТА:\n{final_text}"
    data = _judge(JUDGE_SYSTEM, user)

    faithful = data.get("faithful")
    relevant = data.get("relevant")
    scope_ok = data.get("scope_ok")
    honest_uncertainty = data.get("honest_uncertainty")
    citation_leftover = _citation_leftover(final_text)
    date_suspect = _date_hallucination_suspect(final_text)

    fails = []
    if faithful is False:
        fails.append("faithful")
    if relevant is False:
        fails.append("relevant")
    if scope_ok is False:
        fails.append("scope_ok")
    if is_critical and honest_uncertainty is False:
        fails.append("honest_uncertainty")
    if citation_leftover:
        fails.append("citation_leftover")
    if date_suspect:
        fails.append("date_suspect")

    return {
        "faithful": faithful, "faithful_note": (data.get("faithful_note") or "").strip(),
        "relevant": relevant, "relevant_note": (data.get("relevant_note") or "").strip(),
        "scope_ok": scope_ok, "scope_note": (data.get("scope_note") or "").strip(),
        "honest_uncertainty": honest_uncertainty,
        "honest_uncertainty_note": (data.get("honest_uncertainty_note") or "").strip(),
        "citation_leftover": citation_leftover,
        "date_suspect": date_suspect,
        "fails": fails,
        "verdict": "fail" if fails else "pass",
    }


def run_eval(golden: list[dict] | None = None, *, k: int = config.TOP_K, write: bool = True) -> list[dict]:
    """Per golden query: answer(query, k=k) on PRODUCTION config (evaluates
    shipped behavior) -> judge_answer against the same fragments search()
    would return for it (deterministic — same query, same index, so this
    matches what answer() used internally without needing rag.py to expose
    its hits). Parallelized via _run_parallel. Dumped to
    data/eval/rag_judged.jsonl."""
    golden = golden if golden is not None else load_golden()

    def process(row: dict) -> dict:
        hits = search(row["query"], k=k)
        result = answer(row["query"], k=k)
        verdict = judge_answer(row["query"], hits, result["answer"], is_critical=row["category"] == "critical")
        return {**row, "answer": result["answer"], "raw_text": result["raw_text"], **verdict}

    rows = _run_parallel(process, golden, label="eval:rag")
    if write:
        _dump(config.EVAL_DATA_DIR / "rag_judged.jsonl", rows)
    return rows


def _report_rag(rows: list[dict]) -> None:
    n = len(rows) or 1
    print(f"\n=== RAG eval: {len(rows)} вопросов ===")
    for key in ("faithful", "relevant", "scope_ok"):
        ok = sum(1 for r in rows if r.get(key) is True)
        print(f"  {key:9}: {ok:4}/{n}  ({100 * ok // n}%)")
    print(f"  citation_leftover: {sum(1 for r in rows if r.get('citation_leftover'))}/{n}   "
          f"date_suspect: {sum(1 for r in rows if r.get('date_suspect'))}/{n}")
    print("  verdict   :", dict(Counter(r["verdict"] for r in rows)))

    failed = [r for r in rows if r["verdict"] == "fail"]
    print(f"\n--- {len(failed)} провалившихся ответов (все категории) ---")
    for r in failed:
        print(f"\n[{r['id']}] category={r['category']}  fails={r['fails']}")
        print(f"  Q: {r['query']}")
        print(f"  A: {r['answer']}")
        for note_key in ("faithful_note", "relevant_note", "scope_note", "honest_uncertainty_note"):
            if r.get(note_key):
                print(f"  {note_key}: {r[note_key]}")

    # Critical queries: ALWAYS printed in full, one line each, regardless of
    # --limit or overall pass rate — a single critical failure must never be
    # invisible inside a good-looking aggregate percentage.
    critical = [r for r in rows if r["category"] == "critical"]
    if critical:
        print(f"\n=== CRITICAL ({len(critical)} вопросов) — построчно, не усредняется ===")
        for r in critical:
            mark = "🚨 CRITICAL FAIL" if r["verdict"] == "fail" else "ok"
            print(f"  [{mark}] {r['id']} ({r.get('subtype', '')}): {r['query']}")
            if r["verdict"] == "fail":
                print(f"      fails={r['fails']}")


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
    rows = run_eval(golden)
    _report_rag(rows)


if __name__ == "__main__":
    main()
