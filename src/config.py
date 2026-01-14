import os
from loguru import logger
from dotenv import load_dotenv


def get_settings():
    load_dotenv()


    # Provider-specific API keys with fallback to legacy key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Ensure at least one API key is available
    if not deepseek_api_key and not groq_api_key:
        logger.error("At least one API key must be set: DEEPSEEK_API_KEY, GROQ_API_KEY")
        os._exit(1)

    telegram_token = os.getenv("BOT_TOKEN")
    if not telegram_token:
        logger.error("BOT_TOKEN must be set")
        os._exit(1)

    timezone = os.getenv("TZ", "Europe/Moscow")

    # Which LLM provider to use: "deepseek" (default), "groq", etc.
    llm_provider = os.getenv("LLM_PROVIDER", "groq")

    default_models = {
        "deepseek": "deepseek-reasoner",
        "groq": "openai/gpt-oss-120b",
    }

    if llm_provider not in default_models:
        logger.error(
            "Unsupported LLM provider '%s'. Allowed providers: %s",
            llm_provider,
            ", ".join(sorted(default_models.keys())),
        )
        os._exit(1)

    model = os.getenv("MODEL") or default_models[llm_provider]

    # Get daily token limit from env or use default
    daily_token_limit = int(os.getenv("DAILY_TOKEN_LIMIT", "30000"))

    # Message batching settings
    batch_timeout = float(os.getenv("MESSAGE_BATCH_TIMEOUT", "0.8"))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "30"))

    return {
        "deepseek_api_key": deepseek_api_key,
        "groq_api_key": groq_api_key,
        "model": model,
        "telegram_token": telegram_token,
        "caldav": {
            "timezone": timezone
        },
        "daily_token_limit": daily_token_limit,
        "llm_provider": llm_provider,
        "batch_timeout": batch_timeout,
        "max_batch_size": max_batch_size,
    }