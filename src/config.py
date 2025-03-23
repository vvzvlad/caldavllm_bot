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

    # Check CalDAV settings
    caldav_url = os.getenv("CALDAV_URL")
    caldav_username = os.getenv("CALDAV_USERNAME")
    caldav_password = os.getenv("CALDAV_PASSWORD")
    caldav_calendar_name = os.getenv("CALDAV_CALENDAR_NAME")
    caldav_timezone = os.getenv("CALDAV_TIMEZONE", "Europe/Moscow")
    
    if not all([caldav_url, caldav_username, caldav_password, caldav_calendar_name]):
        logger.error("CalDAV credentials must be set")
        os._exit(1)

    model = os.getenv("MODEL", "deepseek-chat")
    logger.info(f"Using model: {model}")

    return {
        "api_key": api_key,
        "model": model,
        "telegram_token": telegram_token,
        "caldav": {
            "url": caldav_url,
            "username": caldav_username,
            "password": caldav_password,
            "calendar_name": caldav_calendar_name,
            "timezone": caldav_timezone
        }
    } 