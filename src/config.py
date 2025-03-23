import os
from loguru import logger

def get_settings():
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.error("API_KEY must be set")
        os._exit(1)

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_TOKEN must be set")
        os._exit(1)

    model = os.getenv("MODEL", "deepseek-chat")
    logger.info(f"Using model: {model}")

    return {
        "api_key": api_key,
        "model": model,
        "telegram_token": telegram_token
    } 