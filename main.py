import asyncio
from src.bot import CalendarBot
from loguru import logger

if __name__ == "__main__":
    bot = CalendarBot()
    asyncio.run(bot.start())
