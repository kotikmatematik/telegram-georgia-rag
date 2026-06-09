"""Telegram-бот: задаёшь вопрос -> бот отвечает по контенту чатов о Грузии.

Запуск:  uv run python -m src.bot
Нужен BOT_TOKEN в .env (получить у @BotFather).
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

import config
from src.rag import answer

logging.basicConfig(level=logging.INFO)
dp = Dispatcher()

WELCOME = (
    "Привет! Я отвечаю на вопросы о жизни в Грузии на основе переписок из "
    "тематических Telegram-чатов.\n\n"
    "Просто напиши вопрос, например:\n"
    "• Какие документы нужны для открытия ИП?\n"
    "• Как получить водительские права?\n"
    "• В каком районе Тбилиси лучше снять квартиру?"
)


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(WELCOME)


@dp.message()
async def on_question(message: Message) -> None:
    query = (message.text or "").strip()
    if not query:
        await message.answer("Напиши, пожалуйста, текстовый вопрос.")
        return
    await message.chat.do("typing")
    # answer() — синхронный (сетевые вызовы OpenAI), уводим в поток
    result = await asyncio.to_thread(answer, query)
    text = result["answer"] or "Не удалось сформировать ответ."
    await message.answer(text, disable_web_page_preview=True)


async def main() -> None:
    if not config.BOT_TOKEN:
        raise SystemExit("Не задан BOT_TOKEN в .env (получить у @BotFather)")
    bot = Bot(config.BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
