"""Telegram bot: ask a question -> the bot answers from the Georgia chats.

Run:  uv run python -m src.bot
Requires BOT_TOKEN in .env (get it from @BotFather).
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message

import config
from src.rag import answer, to_telegram_html

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
    # answer() is synchronous (blocking OpenAI calls), so run it in a thread
    result = await asyncio.to_thread(answer, query)
    text = result["answer"] or "Не удалось сформировать ответ."
    await message.answer(to_telegram_html(text), parse_mode="HTML", disable_web_page_preview=True)


async def main() -> None:
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN is not set in .env (get it from @BotFather)")
    bot = Bot(config.BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
