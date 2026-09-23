# telegram-georgia-rag

A RAG assistant (Telegram bot) that answers practical questions about life in
Georgia, distilled from real discussions in Russian-speaking topical
Telegram chats (business/sole proprietorship, driver's licenses, rent,
medicine, services, IT, etc.). Every answer cites the source chat and date;
if the chats don't cover a question, the bot says so honestly instead of
guessing.

> The bot answers in Russian — the source chats and target users are
> Russian-speaking.

**Bot:** [@georgia_insider_bot](https://t.me/georgia_insider_bot)

## How it works

Two separate pipelines share the same knowledge store:

```
                     ┌─ collection (offline, run by hand / weekly cron) ─┐
Telegram chats → ingest → threads → knowledge (LLM) → eval → fix → index
                                                                       │
                                                                       ▼
                                                                 chroma_db
                                                                       │
                     ┌─ serving (always on, on the server) ────────────┘
User message → rag (retrieve + generate + cite) → bot (Telegram) → answer
```

- **`src/ingest.py`** — fetch chat history into `data/raw/<chat>.jsonl`
  (Telethon). Incremental: reruns only fetch messages newer than what's
  already on disk.
- **`src/threads.py`** — group raw messages into threads by reply-chains and
  time-proximity bursts (capped at `config.THREAD_MAX_BURST_SIZE`).
  `src/spam.py` drops ads/one-off classifieds/pet-rehoming before this.
- **`src/knowledge.py`** — the actual "distillation": an LLM reads each
  thread and extracts durable `{question, answer, type, city}` knowledge
  units (chit-chat/one-off threads yield nothing). Checkpointed and
  resumable (`resume=True`), with per-thread error isolation — one thread
  tripping Azure's content filter doesn't kill a multi-hour run.
- **`src/eval_knowledge.py`** / **`src/fix_knowledge.py`** — an independent
  LLM judge checks each unit against its source thread (faithful? atomic?
  right type/city?), then fixes what's fixable and drops what isn't
  (with a second independent opinion before any faithfulness-based drop).
- **`src/index.py`** — embeds `data/knowledge/<chat>.fixed.jsonl` into a
  local Chroma collection. Diffs by content hash — only embeds what's new or
  changed, never a full re-embed.
- **`src/update_knowledge.py`** / **`src/weekly_pipeline.py`** — the
  recurring incremental version of the above: only re-touches threads active
  since the last run, for every configured chat, then re-indexes. See
  [Keeping the knowledge fresh](#keeping-the-knowledge-fresh) below.
- **`src/retrieve.py`** — embed a query, search Chroma (`config.TOP_K`,
  `config.RETRIEVAL_MIN_SCORE`).
- **`src/rag.py`** — build the answer: retrieve, generate with inline
  citations, resolve short follow-ups against conversation history
  (`_rewrite_query`), render Telegram HTML with real source links.
- **`src/bot.py`** — the Telegram interface (aiogram): text questions in
  private chat, voice messages (transcribed via Groq's hosted Whisper, free
  tier, supporters only), guest mode (`@bot вопрос` — mention or reply in ANY
  chat, no need to add the bot there — classified via `classify_guest` in
  `src/rag.py` so "спасибо"/chatter doesn't trigger a full answer), a
  per-user daily rate limit (5/day free, 100/day for supporters — see
  `/support`, `/grant`), and a JSONL interaction log for analytics/future
  eval-set growth.
- **`src/eval_retrieval.py`** / **`src/eval_rag.py`** — a from-scratch
  evaluation harness (not ragas) against `eval/golden_queries.jsonl`: sweeps
  `(k, threshold)` for retrieval, judges faithfulness/relevance/hallucination
  for full answers, and never lets a `critical`-category failure hide inside
  an average pass rate. See that file's docstrings and
  `notebooks/explore.ipynb` (sections 15-16) for the metrics explained.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys
```

See `.env.example` for what each key is for. The chat list lives in
`config.py` (`CHATS`) — the bot's account (`TELEGRAM_PHONE`) must already be
a member of every chat listed there.

## Running the pipeline

```bash
uv run python -m src.ingest   # fetch history for every chat in config.CHATS
```

Full from-scratch collection for one new chat (costs tokens — see
`src/knowledge.py`'s module docstring) is a few function calls, not a single
CLI command — see `notebooks/explore.ipynb`, or a throwaway script like:

```python
from src.knowledge import distill_chat
from src.eval_knowledge import eval_precision, _load_knowledge, _threads_by_root
from src.fix_knowledge import fix_batch, save_fixed
from src.update_knowledge import bootstrap_state

username = "some_chat"
distill_chat(username, write=True, resume=True, checkpoint_every=100)
judged = eval_precision(username, write=True)
result = fix_batch(_load_knowledge(username), judged, _threads_by_root(username))
save_fixed(username, result)
bootstrap_state(username)  # seed the cursor so weekly_pipeline doesn't redo all of this
```

Then, or for the existing chats:

```bash
uv run python -m src.index                           # embed + index everything
uv run python -m src.retrieve "как открыть ип"       # test retrieval
uv run python -m src.rag "какие документы для ип"    # test answer
uv run python -m src.bot                             # start the bot
```

## Keeping the knowledge fresh

Collection runs directly on the production server (not a local machine) —
`georgia_ingest.session`, `data/raw/`, and `data/knowledge/` all live there
now, so nothing depends on any laptop being on. `scripts/georgia-weekly.service`
+ `scripts/georgia-weekly.timer` (deployed to `/etc/systemd/system/`) run
`src.weekly_pipeline` every Monday 04:00 (+ up to 5min random delay) and
restart the bot service afterward. Logs: `data/logs/weekly_update.log`
under the external data dir (see Deployment below). Check it any time with:

```bash
ssh deploy@<server> "systemctl list-timers georgia-weekly.timer; tail -30 /opt/telegram-georgia-data/data/logs/weekly_update.log"
```

Because both `update_knowledge` and `index` are incremental, a normal weekly
run costs roughly $0.30-1.30 (measured), not a full from-scratch redo.

## Evaluation

```bash
uv run python -m src.eval_retrieval [--limit N] [--category C]
uv run python -m src.eval_rag [--limit N] [--category C]
```

Golden set: `eval/golden_queries.jsonl` (tracked in git, unlike `data/`),
hand-curated across every configured chat. The `critical` category
(subtypes documented per-row) covers questions where a confident wrong
answer is genuinely costly — legal/medical/financial — and is reported
separately, never folded into an overall pass rate.

## Exploration notebook

```bash
uv run jupyter lab   # then open notebooks/explore.ipynb
```

Walks through every pipeline step by hand, plus the retrieval/RAG eval demo
with metrics explained inline.

## Deployment

The bot runs as a systemd service (`georgia-bot`, under a dedicated
non-root `deploy` user — not root) on a small Ubuntu VPS, which also runs
collection itself (see above) — `.env` there additionally carries the
`TELEGRAM_API_ID`/`TELEGRAM_API_HASH`/`TELEGRAM_PHONE` used by
`georgia_ingest.session`. That session file grants full access to that
Telegram account (not just bot-scoped access), which is the real reason the
server's baseline hardening matters: SSH is key-only, no root login, `ufw`
allows only SSH in, `fail2ban` bans repeat auth failures and pings Telegram
on every ban (see `scripts/notify_ban.sh`), and the bot service crashing
also pings Telegram (`scripts/notify_on_failure.sh`, wired via
`ExecStopPost=`).

On the server, `data/` and `chroma_db/` live OUTSIDE the code checkout
entirely — at `/opt/telegram-georgia-data/{data,chroma_db}`, pointed to via
`GEORGIA_DATA_DIR`/`GEORGIA_CHROMA_DIR` in `.env` (`config.py` falls back to
the usual repo-relative paths when those aren't set, which is what local dev
uses). This is deliberate, not incidental: a code redeploy only ever touches
`/opt/telegram-georgia-rag/`, so it is now structurally unable to reach the
live vector index or collected knowledge, regardless of what `--exclude`
flags a given rsync command does or doesn't have. This came from a real
incident — a deploy without `--exclude='chroma_db'` once silently overwrote
the server's live vector index with a stale local copy (recovered by
re-running `uv run python -m src.index`, which is diff-based and safe to
re-run any time) — the exclude flags below are kept anyway as a second guard,
but the directory split is the actual fix.

```bash
rsync -az --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='data' --exclude='chroma_db' --exclude='notebooks' --exclude='eval' \
  --exclude='*.session*' --exclude='.env' \
  ./ deploy@<server>:/opt/telegram-georgia-rag/
ssh deploy@<server> "sudo systemctl restart georgia-bot"
```

(`deploy` has full passwordless sudo — broader than originally scoped; the
Contabo web console became the only path back in once a lockout happened
with no fallback, so it was widened rather than risk repeating that.)

## ⚠️ Privacy

The chats contain real people's personal data. `data/`, `chroma_db/`, and
`.env` are gitignored — **never commit them**. `.env.example` must only ever
contain placeholder values, never a real phone number/key/token.
