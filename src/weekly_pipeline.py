"""Incremental knowledge refresh for ALL configured chats — the recurring
"pull new messages, distill new/changed threads, re-embed" job (see
config.REPROCESS_OVERLAP_DAYS and src.update_knowledge). Meant to run on a
schedule (see scripts/weekly_update.sh + the launchd job that calls it), not
manually per chat like the one-off full_pipeline scripts used for the
initial from-scratch collection of a new chat.

Cheap by construction: src.ingest is incremental (only fetches messages
newer than what's on disk), update_knowledge only re-touches threads active
since the last run (see its module docstring), and src.index only re-embeds
content that actually changed — so a normal weekly run costs roughly what
was measured for incremental re-collection (~$0.30-1.30/week across all
chats), not a full from-scratch redo.

One chat failing (e.g. a transient Azure error) is logged and skipped, not
allowed to abort the whole run — the same per-thread/per-chat isolation
already used elsewhere in the pipeline.

Run:  uv run python -m src.weekly_pipeline
"""
from __future__ import annotations

import subprocess
import sys
import time

import config
from src.update_knowledge import update_knowledge


def main() -> None:
    t0 = time.time()
    print("[weekly] === run started ===", flush=True)

    print("[weekly] ingest...", flush=True)
    subprocess.run([sys.executable, "-m", "src.ingest"], check=True)

    for chat in config.CHATS:
        username = chat["username"]
        print(f"[weekly] update_knowledge({username!r})", flush=True)
        try:
            update_knowledge(username)
        except Exception as e:
            print(
                f"[weekly] {username}: FAILED ({type(e).__name__}: {e}) — "
                "skipping, other chats continue",
                flush=True,
            )

    print("[weekly] index...", flush=True)
    subprocess.run([sys.executable, "-m", "src.index"], check=True)

    print(f"[weekly] === run finished in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
