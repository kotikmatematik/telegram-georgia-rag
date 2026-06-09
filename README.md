# telegram-georgia-rag

RAG-ассистент (Telegram-бот), который отвечает на вопросы о жизни в Грузии на
основе переписок из русскоязычных тематических Telegram-чатов (ИП/бизнес, права,
аренда, услуги, объявления и т.д.).

## Как это работает

```
Telegram → ingest → preprocess/chunk → embed+index (Chroma) → retrieve → GPT → бот
```

- **ingest** — выгрузка истории чатов в `data/raw/*.jsonl` (Telethon)
- **preprocess** — очистка и сборка сообщений в чанки-диалоги `data/chunks/*.jsonl`
- **index** — эмбеддинги OpenAI → локальная векторная БД ChromaDB
- **rag** — поиск top-k чанков + генерация ответа со ссылками на источники
- **bot** — интерфейс в Telegram (aiogram)

## Установка

```bash
uv sync
cp .env.example .env   # затем заполнить ключи
```

Нужно заполнить в `.env`:
- `OPENAI_API_KEY` — ключ OpenAI
- `BOT_TOKEN` — токен бота от [@BotFather](https://t.me/BotFather)
- `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` — с https://my.telegram.org/apps
- `TELEGRAM_PHONE` — номер аккаунта, состоящего в чатах

Список чатов настраивается в `config.py` (`CHATS`).

## Запуск пайплайна

```bash
uv run python -m src.ingest                          # выгрузка истории
uv run python -m src.preprocess                      # чанкинг
uv run python -m src.index                           # индексация
uv run python -m src.retrieve "как открыть ип"       # проверка поиска
uv run python -m src.rag "какие документы для ип"    # проверка ответа
uv run python -m src.bot                             # запуск бота
```

## ⚠️ Приватность

Чаты содержат личные данные реальных людей. Каталоги `data/`, `chroma_db/` и
файл `.env` добавлены в `.gitignore` — **не коммить их** в публичный репозиторий.
