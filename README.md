# telegram-georgia-rag

A RAG assistant (Telegram bot) that answers questions about life in Georgia
based on conversations from Russian-speaking topical Telegram chats (sole
proprietorship/business, driver's licenses, rent, services, classifieds, etc.).

> The bot answers in Russian, because the source chats and target users are
> Russian-speaking.

## How it works

```
Telegram → ingest → preprocess/chunk → embed+index (Chroma) → retrieve → GPT → bot
```

- **ingest** — fetch chat history into `data/raw/*.jsonl` (Telethon)
- **preprocess** — drop spam (quick-money / drugs / 18+), then group messages
  into dialog chunks `data/chunks/*.jsonl` (time windows + reply chains)
- **index** — OpenAI embeddings → local ChromaDB vector store
- **rag** — retrieve top-k chunks + generate an answer with source links
- **bot** — Telegram interface (aiogram)

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys
```

Fill in `.env`:
- `OPENAI_API_KEY` — OpenAI key
- `BOT_TOKEN` — bot token from [@BotFather](https://t.me/BotFather)
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` — from https://my.telegram.org/apps
- `TELEGRAM_PHONE` — phone number of the account that is a member of the chats

The chat list is configured in `config.py` (`CHATS`).

## Running the pipeline

```bash
uv run python -m src.ingest                          # fetch history
uv run python -m src.preprocess                      # chunking
uv run python -m src.index                           # indexing
uv run python -m src.retrieve "как открыть ип"       # test retrieval
uv run python -m src.rag "какие документы для ип"    # test answer
uv run python -m src.bot                             # start the bot
```

> The example queries are in Russian on purpose — they must match the
> Russian-language content of the chats.

## Exploration notebook

To inspect each pipeline step by hand:

```bash
uv run jupyter lab
```

Then open `notebooks/explore.ipynb`.

## ⚠️ Privacy

The chats contain real people's personal data. The `data/`, `chroma_db/`
directories and the `.env` file are listed in `.gitignore` — **do not commit
them** to a public repository.
