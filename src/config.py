import os
from loguru import logger
from dotenv import load_dotenv

def get_settings():
    load_dotenv()

    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY must be set")
        os._exit(1)

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_TOKEN must be set")
        os._exit(1)

    caldav_timezone = os.getenv("CALDAV_TIMEZONE", "Europe/Moscow")

    model = os.getenv("MODEL", "deepseek-chat")

    # Get daily token limit from env or use default
    daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "30000"))

    return {
        "api_key": api_key,
        "model": model,
        "telegram_token": telegram_token,
        "caldav": {
            "timezone": caldav_timezone
        },
        "daily_token_limit": daily_token_limit
    } 