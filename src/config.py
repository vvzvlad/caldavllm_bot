import os
from loguru import logger
from dotenv import load_dotenv


def get_settings():
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("LLM_API_KEY must be set")
        os._exit(1)

    telegram_token = os.getenv("BOT_TOKEN")
    if not telegram_token:
        logger.error("BOT_TOKEN must be set")
        os._exit(1)

    caldav_timezone = os.getenv("CALDAV_TIMEZONE", "Europe/Moscow")

    model = os.getenv("MODEL", "openai/gpt-oss-120b")

    # Which LLM provider to use: "deepseek" (default), "groq", etc.
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek")

    # Get daily token limit from env or use default
    daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "30000"))

    # Message batching settings
    batch_timeout = float(os.getenv("MESSAGE_BATCH_TIMEOUT", "0.8"))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "30"))

    return {
        "api_key": api_key,
        "model": model,
        "telegram_token": telegram_token,
        "caldav": {
            "timezone": caldav_timezone
        },
        "daily_token_limit": daily_token_limit,
        "llm_provider": llm_provider,
        "batch_timeout": batch_timeout,
        "max_batch_size": max_batch_size,
    }