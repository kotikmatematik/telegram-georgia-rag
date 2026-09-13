"""Apply fixes to knowledge units based on eval_knowledge's judgements.

PRECISION_SYSTEM enforces verdict=fix ⟹ faithful=true and useful=true (see
eval_knowledge.py) — so a "fix" verdict only ever means a type, city, or
atomicity defect, never a faithfulness problem. A "drop" verdict means
faithful=false or useful=false.

What this does with each verdict:
  keep — unchanged.
  fix  — adopt type_suggested when type_ok=false; adopt city_suggested when
         city_ok=false (can legitimately be null, e.g. clearing a wrongly
         attached city); re-split into atomic sub-pairs via config.FIX_MODEL
         when atomic=false (any/all of these can apply to the same unit).
  drop — if faithful=false, get a SECOND independent opinion from
         config.REVERIFY_MODEL — a genuine third model, deliberately not shown
         the first judge's verdict/note — before discarding — only drop when
         BOTH agree it's unfaithful. If the second judge disagrees,
         the pair is NOT confidently unfaithful, so it is kept (returned in
         `kept`, and also separately in `rescued` so you can see which
         "keep"s came from a disagreement rather than a clean verdict).
         useful=false drops (one-off/no-answer/restated — already fairly
         rule-based) are NOT re-verified, per the scope asked for.

Run from a notebook on an in-memory batch (units + judged from eval_knowledge,
threads_by_root from src.knowledge.select_threads); no CLI entry point yet —
this is still a prototype step downstream of eval_knowledge.

⚠️ Spends OpenAI tokens: one call per atomize + one call per faithful=false drop.
"""
from __future__ import annotations

import json

import config
from src.eval_knowledge import _run_parallel
from src.knowledge import _thread_text
from src.store import chat_json

ATOMIZE_SYSTEM = (
    "Пара «вопрос-ответ» ниже на самом деле смешивает несколько разных тем. "
    "Раздели её на отдельные АТОМАРНЫЕ пары — одна тема на пару. Опирайся "
    "только на текст исходного ответа, ничего не добавляй от себя. Если после "
    "разбора осталась только одна тема — верни одну пару как есть.\n"
    "Ответь строго JSON: {\"pairs\": [{\"question\": \"...\", \"answer\": \"...\"}]}"
)

REVERIFY_SYSTEM = (
    "Оцени, подтверждается ли ответ тредом ДОСЛОВНО — каждое утверждение "
    "должно быть сказано в тексте треда, ничего не додумано и не обобщено "
    "сверх сказанного.\n"
    "Ответь строго JSON: {\"faithful\": bool, \"note\": \"коротко почему\"}"
)


def _atomize_one(unit: dict) -> list[dict]:
    data = chat_json(
        config.FIX_MODEL, ATOMIZE_SYSTEM,
        f"Вопрос: {unit['question']}\nОтвет: {unit['answer']}",
        temperature=0,
    )
    pairs = data.get("pairs", [])
    out = [
        {**unit, "question": (p.get("question") or "").strip(), "answer": (p.get("answer") or "").strip()}
        for p in pairs
    ]
    out = [u for u in out if u["question"] and u["answer"]]
    return out or [unit]  # fall back to the original pair if the split failed


def _reverify_one(unit: dict, judged_row: dict, thread: list[dict] | None) -> dict:
    if thread is None:
        # Can't re-verify without the source thread — treat as still-dropped
        # rather than silently keeping something we can't check.
        return {"unit": unit, "confirmed": True, "reverify_note": "исходный тред не найден"}
    rv = chat_json(
        config.REVERIFY_MODEL, REVERIFY_SYSTEM,
        f"ТРЕД:\n{_thread_text(thread)}\n\nВопрос: {unit['question']}\nОтвет: {unit['answer']}",
        temperature=0, reasoning_effort=config.REVERIFY_REASONING_EFFORT,
    )
    confirmed = rv.get("faithful") is False  # second judge agrees it's unfaithful
    return {
        "unit": unit,
        "confirmed": confirmed,
        "reverify_note": rv.get("note", ""),
        "judge_note": judged_row.get("note", ""),
    }


def fix_batch(
    units: list[dict], judged: list[dict], threads_by_root: dict[int, list[dict]]
) -> dict[str, list[dict]]:
    """Apply fixes based on judged verdicts.

    `units` and `judged` must be the SAME batch in the SAME order (e.g. the
    exact lists you passed to / got back from eval_knowledge.judge_units).

    Returns {"kept": [...], "fixed": [...], "dropped": [...], "rescued": [...]}
    — `rescued` is the subset of `kept` where the 2nd judge overruled a drop.
    """
    kept: list[dict] = []
    fixed_simple: list[dict] = []
    needs_atomize: list[dict] = []
    needs_reverify: list[tuple[dict, dict, list[dict] | None]] = []
    dropped: list[dict] = []

    for u, j in zip(units, judged):
        verdict = j.get("verdict")
        if verdict == "keep":
            kept.append(u)
        elif verdict == "fix":
            fu = dict(u)
            if not j.get("type_ok") and j.get("type_suggested"):
                fu["type"] = j["type_suggested"]
            if not j.get("city_ok"):
                cs = j.get("city_suggested")
                # the corrected city can legitimately be null (e.g. a wrongly
                # attached city should be cleared), so don't require truthiness
                fu["city"] = None if (not cs or str(cs).strip().lower() in {"null", "none", "-"}) else cs
            if not j.get("atomic"):
                needs_atomize.append(fu)
            else:
                fixed_simple.append(fu)
        elif verdict == "drop":
            if j.get("faithful") is False:
                needs_reverify.append((u, j, threads_by_root.get(u["root_msg_id"])))
            else:
                dropped.append({**u, "drop_reason": j.get("note", "")})
        else:  # skip/error from the judge — don't silently keep
            dropped.append({**u, "drop_reason": j.get("note") or "judge error/skip"})

    atomized = _run_parallel(_atomize_one, needs_atomize, label="fix:atomize")
    fixed = fixed_simple + [sub for group in atomized for sub in group]

    reverified = _run_parallel(
        lambda triple: _reverify_one(*triple), needs_reverify, label="fix:reverify"
    )
    # Second judge disagreed => not confidently unfaithful => keep it. Tagged
    # (not silently merged) so you can still see which "keep"s were rescued
    # this way — returned separately as `rescued`, a subset already folded
    # into `kept`.
    rescued: list[dict] = []
    for r in reverified:
        if r["confirmed"]:
            dropped.append({**r["unit"], "drop_reason": r["judge_note"], "reverify_note": r["reverify_note"]})
        else:
            ru = {**r["unit"], "judge_note": r["judge_note"], "reverify_note": r["reverify_note"]}
            rescued.append(ru)
            kept.append(ru)

    print(
        f"[fix] kept={len(kept)} (of which rescued by 2nd judge: {len(rescued)}) "
        f"fixed={len(fixed)} dropped={len(dropped)}"
    )
    return {"kept": kept, "fixed": fixed, "dropped": dropped, "rescued": rescued}


def save_fixed(username: str, result: dict[str, list[dict]]) -> None:
    """Write the final, corrected knowledge (kept + fixed from fix_batch's
    result) to data/knowledge/<username>.fixed.jsonl — this is the file meant
    to feed retrieve/rag downstream, once that's wired up."""
    final_knowledge = result["kept"] + result["fixed"]
    out_path = config.KNOWLEDGE_DIR / f"{username}.fixed.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for u in final_knowledge:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    print(f"[fix] saved {len(final_knowledge)} items -> {out_path}")
