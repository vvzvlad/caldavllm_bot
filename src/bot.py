import asyncio
import telebot
from datetime import datetime
from loguru import logger
from .config import get_settings
from .llm import DeepSeekLLM
from .calendar import CalendarManager

class CalendarBot:
    def __init__(self):
        self.settings = get_settings()
        self.bot = telebot.TeleBot(self.settings["telegram_token"])
        self.llm = DeepSeekLLM()
        self.calendar = CalendarManager()
        self._setup_handlers()

    def _format_datetime(self, iso_datetime: str) -> str:
        """Format ISO datetime to human readable format"""
        try:
            dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y %H:%M")
        except Exception as e:
            logger.error(f"Failed to format datetime: {str(e)}")
            return iso_datetime

    def _create_event_message(self, event: dict) -> str:
        """Create formatted event message"""
        parts = []
        
        if event.get("title"):
            parts.append(f"📌 {event['title']}")
            
        if event.get("start_time"):
            parts.append(f"🕒 Начало: {self._format_datetime(event['start_time'])}")
            
        if event.get("end_time"):
            parts.append(f"🕒 Конец: {self._format_datetime(event['end_time'])}")
            
        if event.get("location"):
            parts.append(f"📍 {event['location']}")
            
        if event.get("description"):
            parts.append(f"📝 {event['description']}")
            
        return "\n".join(parts)

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
                
                # Create inline keyboard
                keyboard = telebot.types.InlineKeyboardMarkup()
                keyboard.row(
                    telebot.types.InlineKeyboardButton("✅ Добавить", callback_data="add"),
                    telebot.types.InlineKeyboardButton("❌ Отменить", callback_data="cancel")
                )
                
                # Send event preview with buttons
                self.bot.reply_to(
                    message,
                    f"Проверьте информацию о событии:\n\n{self._create_event_message(event)}",
                    reply_markup=keyboard
                )
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.bot.reply_to(message, "Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

        @self.bot.callback_query_handler(func=lambda call: True)
        def handle_callback(call):
            try:
                action = call.data
                
                if action == 'add':
                    # Get original message text from the message that was replied to
                    original_message = call.message.reply_to_message
                    if not original_message:
                        self.bot.answer_callback_query(call.id, "Ошибка: не удалось найти исходное сообщение")
                        return
                        
                    event = asyncio.run(self.llm.parse_calendar_event(original_message.text))
                    
                    if not event or not event["result"]:
                        self.bot.answer_callback_query(call.id, "Ошибка при добавлении события")
                        return
                    
                    # Add event to calendar
                    success = self.calendar.add_event(
                        title=event["title"],
                        start_time=event["start_time"],
                        end_time=event["end_time"],
                        description=event["description"],
                        location=event["location"]
                    )
                    
                    if success:
                        self.bot.answer_callback_query(call.id, "✅ Событие добавлено в календарь")
                        self.bot.edit_message_reply_markup(
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                            reply_markup=None
                        )
                    else:
                        self.bot.answer_callback_query(call.id, "❌ Ошибка при добавлении события")
                        
                elif action == 'cancel':
                    self.bot.answer_callback_query(call.id, "❌ Добавление отменено")
                    self.bot.edit_message_reply_markup(
                        chat_id=call.message.chat.id,
                        message_id=call.message.message_id,
                        reply_markup=None
                    )
                    
            except Exception as e:
                logger.error(f"Error handling callback: {str(e)}")
                self.bot.answer_callback_query(call.id, "Произошла ошибка")

    def run(self):
        logger.info("Starting bot...")
        self.bot.infinity_polling()
