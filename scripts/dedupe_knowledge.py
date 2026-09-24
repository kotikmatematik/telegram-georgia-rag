"""Find and resolve near-duplicate knowledge units within each chat's
data/knowledge/<chat>.fixed.jsonl.

Root cause: thread-building sometimes produces two overlapping threads with
different root_msg_id that both distill the same underlying reply into a
knowledge unit — same source message, same (or reworded) question, near/
exact-duplicate answer. Retrieval then returns both as separate hits, and the
generated answer cites the same source twice.

Two units sharing a `link` (source_link, falling back to root_link — same
field src.index uses) are treated as duplicates if EITHER:
  - their question text matches exactly, OR
  - the cosine similarity of their (question+answer) embedding is >= DEDUPE_SIM_THRESHOLD
A pair sharing a link with neither signal is almost always just two different
facts pulled from the same message — left untouched.

Within a duplicate cluster:
  - if every pairwise similarity is >= DEDUPE_SIM_THRESHOLD, the units are
    near-identical in content — keep one, drop the rest;
  - otherwise (same question, but answers diverge enough — each may hold a
    fact the other lacks) — merge via one LLM call that keeps every distinct
    fact from all versions, instead of picking one and silently losing info.

Only .fixed.jsonl is touched (mirrors save_fixed()/update_knowledge() — the
raw .jsonl distillation output is never rewritten). Run src.index afterwards
to push the change into Chroma (it diffs by id, so deletions/merges there are
picked up automatically — no manual vector surgery needed).

Run:  uv run python -m scripts.dedupe_knowledge          (dry run, reports only)
      uv run python -m scripts.dedupe_knowledge --apply   (rewrites .fixed.jsonl)

Wired into src.weekly_pipeline (between update_knowledge and index) so
future backfills/updates don't quietly re-accumulate duplicates.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import config
from src.store import chat_json, embed_texts

DEDUPE_SIM_THRESHOLD = 0.97

MERGE_SYSTEM = (
    "Тебе даны несколько версий одного и того же факта из Telegram-чата про "
    "жизнь в Грузии — они описывают одно и то же, но с разной степенью "
    "детализации (разные участники сформулировали похожий вопрос, или "
    "распознавание дало немного разный текст одного и того же обсуждения). "
    "Слей их в ОДНУ запись: единый вопрос и единый ответ, который сохраняет "
    "АБСОЛЮТНО ВСЕ различающиеся детали из всех версий (адреса, цены, имена, "
    "контакты, нюансы) — ничего не выбрасывай, если это не дословный повтор. "
    "Не добавляй ничего от себя, не пиши вступлений вроде «в чате обсуждали». "
    "Ответь строго JSON: {\"question\": \"...\", \"answer\": \"...\"}"
)


def _link(u: dict) -> str:
    return u.get("source_link") or u["root_link"]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _merge_cluster(units: list[dict]) -> dict:
    """LLM-merge a cluster of >=2 partially-overlapping units into one,
    keeping the metadata of whichever member has the most recent date
    (freshest source attribution)."""
    payload = "\n\n".join(
        f"Версия {i+1}:\nВопрос: {u['question']}\nОтвет: {u['answer']}"
        for i, u in enumerate(units)
    )
    data = chat_json(config.FIX_MODEL, MERGE_SYSTEM, payload, reasoning_effort=config.FIX_REASONING_EFFORT)
    base = max(units, key=lambda u: u.get("date") or "")
    merged = dict(base)
    if data.get("question") and data.get("answer"):
        merged["question"] = data["question"]
        merged["answer"] = data["answer"]
    return merged


def _dedupe_chat(username: str, apply: bool) -> dict:
    path = config.KNOWLEDGE_DIR / f"{username}.fixed.jsonl"
    if not path.exists():
        return {"units": 0}
    with path.open(encoding="utf-8") as f:
        units = [json.loads(line) for line in f if line.strip()]

    by_link: dict[str, list[int]] = {}
    for idx, u in enumerate(units):
        by_link.setdefault(_link(u), []).append(idx)
    candidate_groups = [idxs for idxs in by_link.values() if len(idxs) > 1]
    if not candidate_groups:
        return {"units": len(units), "groups": 0, "dropped": 0, "merged_groups": 0}

    # Embed only the units inside a candidate group — cheap, most units in a
    # normal file share no link with anyone.
    flat_idxs = [i for g in candidate_groups for i in g]
    embeds = embed_texts([f"{units[i]['question']}\n{units[i]['answer']}" for i in flat_idxs])
    emb_by_idx = {i: np.array(e) for i, e in zip(flat_idxs, embeds)}

    to_drop: set[int] = set()
    merges: list[tuple[list[int], dict]] = []
    dropped_count = 0
    merged_group_count = 0

    for idxs in candidate_groups:
        # union-find over "is a duplicate of" edges within this link-group
        parent = {i: i for i in idxs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pair_sims: dict[tuple[int, int], float] = {}
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                sim = _cos(emb_by_idx[i], emb_by_idx[j])
                same_q = units[i]["question"].strip() == units[j]["question"].strip()
                if same_q or sim >= DEDUPE_SIM_THRESHOLD:
                    union(i, j)
                pair_sims[(i, j)] = sim

        clusters: dict[int, list[int]] = {}
        for i in idxs:
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            if len(members) < 2:
                continue
            sims = [
                pair_sims[(a, b)] if (a, b) in pair_sims else pair_sims[(b, a)]
                for x, a in enumerate(members) for b in members[x + 1:]
            ]
            if all(s >= DEDUPE_SIM_THRESHOLD for s in sims):
                keep = members[0]
                for i in members[1:]:
                    to_drop.add(i)
                dropped_count += len(members) - 1
            else:
                merges.append((members, {}))
                merged_group_count += 1

    if apply:
        for members, _ in merges:
            merged_unit = _merge_cluster([units[i] for i in members])
            keep = members[0]
            units[keep] = merged_unit
            for i in members[1:]:
                to_drop.add(i)

        remaining = [u for i, u in enumerate(units) if i not in to_drop]
        with path.open("w", encoding="utf-8") as f:
            for u in remaining:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")

    return {
        "units": len(units),
        "groups": len(candidate_groups),
        "dropped": dropped_count,
        "merged_groups": merged_group_count,
    }


def run(apply: bool) -> None:
    print(f"[dedupe] mode: {'APPLY' if apply else 'DRY RUN'}")
    totals = {"dropped": 0, "merged_groups": 0}
    for chat in config.CHATS:
        username = chat["username"]
        stats = _dedupe_chat(username, apply)
        if stats.get("groups"):
            print(
                f"[dedupe] {username}: {stats['units']} units, "
                f"{stats['groups']} link-groups checked, "
                f"{stats['dropped']} pure duplicates dropped, "
                f"{stats['merged_groups']} groups merged"
            )
            totals["dropped"] += stats["dropped"]
            totals["merged_groups"] += stats["merged_groups"]
    print(f"[dedupe] TOTAL: {totals['dropped']} dropped, {totals['merged_groups']} merged")
    if not apply:
        print("[dedupe] dry run — nothing written. Re-run with --apply to write, then `uv run python -m src.index`.")


def main() -> None:
    run(apply="--apply" in sys.argv)


if __name__ == "__main__":
    main()
