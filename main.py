import asyncio
from src.bot import CalendarBot

if __name__ == "__main__":
    bot = CalendarBot()
    asyncio.run(bot.start())
