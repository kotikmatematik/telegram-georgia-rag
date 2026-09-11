"""Apply fixes to knowledge units based on eval_knowledge's judgements.

PRECISION_SYSTEM enforces verdict=fix ⟹ faithful=true and useful=true (see
eval_knowledge.py) — so a "fix" verdict only ever means a type or atomicity
defect, never a faithfulness problem. A "drop" verdict means faithful=false
or useful=false.

What this does with each verdict:
  keep — unchanged.
  fix  — adopt type_suggested when type_ok=false; re-split into atomic
         sub-pairs via LLM when atomic=false (both can apply to the same unit).
  drop — if faithful=false, get a SECOND independent opinion from
         config.CHAT_MODEL (the distiller's own model, not the stronger judge)
         before discarding — agreement confirms the drop, disagreement routes
         the unit to "disputed" for manual review instead of auto-dropping.
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
from src.store import openai_client

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
    client = openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,  # same model as the distiller
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": ATOMIZE_SYSTEM},
            {"role": "user", "content": f"Вопрос: {unit['question']}\nОтвет: {unit['answer']}"},
        ],
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        pairs = data.get("pairs", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pairs = []
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
    client = openai_client()
    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REVERIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"ТРЕД:\n{_thread_text(thread)}\n\n"
                    f"Вопрос: {unit['question']}\nОтвет: {unit['answer']}"
                ),
            },
        ],
    )
    try:
        rv = json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError, TypeError):
        rv = {}
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

    Returns {"kept": [...], "fixed": [...], "dropped": [...], "disputed": [...]}.
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
    disputed: list[dict] = []
    for r in reverified:
        if r["confirmed"]:
            dropped.append({**r["unit"], "drop_reason": r["judge_note"], "reverify_note": r["reverify_note"]})
        else:
            disputed.append({**r["unit"], "judge_note": r["judge_note"], "reverify_note": r["reverify_note"]})

    print(
        f"[fix] kept={len(kept)} fixed={len(fixed)} dropped={len(dropped)} "
        f"disputed={len(disputed)}"
    )
    return {"kept": kept, "fixed": fixed, "dropped": dropped, "disputed": disputed}
