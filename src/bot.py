import asyncio
import telebot
from loguru import logger
from .config import get_settings
from .llm import DeepSeekLLM

class CalendarBot:
    def __init__(self):
        self.settings = get_settings()
        self.bot = telebot.TeleBot(self.settings["telegram_token"])
        self.llm = DeepSeekLLM()
        self._setup_handlers()

    def _setup_handlers(self):
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            try:
                logger.info(f"Received message from {message.from_user.id}: {message.text}")
                self.bot.send_chat_action(message.chat.id, 'typing')
                event = asyncio.run(self.llm.parse_calendar_event(message.text))
                
                if not event:
                    self.bot.reply_to(
                        message,
                        "Не удалось обработать сообщение. Попробуйте еще раз."
                    )
                    return
                
                if not event["result"]:
                    self.bot.reply_to(
                        message,
                        f"Не удалось распарсить событие: {event['comment']}"
                    )
                    return
                
                response = (
                    f"✅ Событие успешно распарсено!\n\n"
                    f"📌 {event['title']}\n"
                    f"🕒 Начало: {event['start_time']}\n"
                    f"🕒 Конец: {event['end_time']}\n"
                    f"📍 {event['location']}\n"
                    f"📝 {event['description']}"
                )
                
                self.bot.reply_to(message, response)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.bot.reply_to(message, "Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

    def run(self):
        logger.info("Starting bot...")
        self.bot.infinity_polling()
