import asyncio
import telebot
from datetime import datetime
from loguru import logger
from threading import Thread
from .config import get_settings
from .llm import DeepSeekLLM
from .calendar import CalendarManager
from .users import UserManager

class CalendarBot:
    def __init__(self):
        self.settings = get_settings()
        self.bot = telebot.TeleBot(self.settings["telegram_token"])
        self.llm = DeepSeekLLM()
        self.calendar = CalendarManager()
        self.user_manager = UserManager()
        self.parsed_events = {}  # Store parsed events by message_id
        self._setup_handlers()

    def _format_datetime(self, iso_datetime: str) -> str:
        """Format ISO datetime to human readable format"""
        try:
            dt = datetime.fromisoformat(iso_datetime.replace('Z', '+00:00'))
            return dt.strftime("%d.%m.%Y %H:%M")
        except Exception as e:
            logger.error(f"Failed to format datetime: {str(e)}")
            return iso_datetime

    def _format_number(self, number: int) -> str:
        """Format number to human readable format with k suffix"""
        if number >= 1000:
            return f"{number // 1000}к"
        return str(number)

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

    def _send_typing_status(self, chat_id: int, stop_event):
        """Send typing status every 4 seconds until stop_event is set"""
        while not stop_event.is_set():
            try:
                self.bot.send_chat_action(chat_id, 'typing')
                # Sleep for 4 seconds (typing status lasts 5 seconds)
                for _ in range(40):  # 4 seconds with 0.1s checks
                    if stop_event.is_set():
                        break
                    asyncio.run(asyncio.sleep(0.1))
            except Exception as e:
                logger.error(f"Error sending typing status: {str(e)}")
                break

    def _setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            welcome_text = (
                "👋 Привет! Я бот для добавления событий в календарь.\n\n"
                "Сначала нужно настроить подключение к твоему календарю. "
                "Используй команду /caldav с параметрами:\n"
                "/caldav username password url calendar_name\n\n"
                "Например:\n"
                "/caldav username@fastmail.com password https://caldav.fastmail.com/dav/ calendar_name\n\n"
                "После этого просто напиши мне о событии, например:\n"
                "• Завтра в 15:00 встреча с клиентом\n"
                "• 25 марта в 11 утра лекция о японском символизме\n"
                "• Встреча в офисе в понедельник в 10:00\n\n"
                "Я пойму текст и добавлю событие в твой календарь."
            )
            self.bot.reply_to(message, welcome_text)

        @self.bot.message_handler(commands=['stats'])
        def handle_stats(message):
            stats = self.user_manager.get_user_stats(message.from_user.id)
            if not stats:
                self.bot.reply_to(message, "У вас пока нет статистики использования.")
                return
                
            last_request = datetime.fromisoformat(stats["last_request"]) if stats["last_request"] else None
            last_request_str = last_request.strftime("%d.%m.%Y %H:%M:%S") if last_request else "никогда"
            
            remaining_tokens = self.user_manager.get_remaining_tokens(message.from_user.id)
            
            stats_text = (
                "📊 Ваша статистика:\n\n"
                f"Количество запросов: {stats['requests_count']}\n"
                f"Всего использовано токенов: {self._format_number(stats['total_tokens'])}\n"
                f"Осталось токенов сегодня: {self._format_number(remaining_tokens)}\n"
                f"Последний запрос: {last_request_str}"
            )
            self.bot.reply_to(message, stats_text)

        @self.bot.message_handler(commands=['caldav'])
        def handle_caldav(message):
            try:
                # Check if user provided all required parameters
                params = message.text.split()
                if len(params) != 5:
                    self.bot.reply_to(
                        message,
                        "Неверный формат команды. Используйте:\n"
                        "/caldav username password url calendar_name\n\n"
                        "Например:\n"
                        "/caldav user@fastmail.com strong_password https://caldav.fastmail.com/dav/ main_calendar"
                    )
                    return

                # Get parameters
                _, username, password, url, calendar_name = params

                # Show checking message
                self.bot.send_message(message.chat.id, "🔄 Проверка подключения к календарю...")

                # Check calendar access
                success, error = self.calendar.check_calendar_access(url, username, password, calendar_name)
                if not success:
                    self.bot.send_message(message.chat.id, f"❌ {error}")
                    return

                # Save credentials
                success = self.user_manager.save_caldav_credentials(
                    message.from_user.id,
                    username,
                    password,
                    url,
                    calendar_name
                )

                if success:
                    self.bot.reply_to(
                        message,
                        "✅ Календарь доступен, настройки успешно сохранены!\n"
                        "Теперь вы можете добавлять события."
                    )
                else:
                    self.bot.reply_to(
                        message,
                        "❌ Не удалось сохранить настройки календаря. Попробуйте еще раз."
                    )

            except Exception as e:
                logger.error(f"Error setting up CalDAV: {str(e)}")
                self.bot.reply_to(
                    message,
                    "Произошла ошибка при настройке календаря. Попробуйте еще раз."
                )

        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            try:
                # Check if user has CalDAV credentials
                if not self.user_manager.has_caldav_credentials(message.from_user.id):
                    self.bot.reply_to(
                        message,
                        "Сначала нужно настроить подключение к календарю. "
                        "Используйте команду /caldav"
                    )
                    return

                # Check token limit
                if not self.user_manager.check_token_limit(message.from_user.id):
                    remaining_tokens = self.user_manager.get_remaining_tokens(message.from_user.id)
                    self.bot.reply_to(
                        message,
                        f"Достигнут дневной лимит токенов ({self.user_manager.daily_token_limit}). "
                        "Попробуйте завтра."
                    )
                    return

                logger.info(f"Received message from {message.from_user.id}: {message.text}")
                
                # Start typing status in a separate thread
                stop_typing = asyncio.Event()
                typing_thread = Thread(
                    target=self._send_typing_status,
                    args=(message.chat.id, stop_typing)
                )
                typing_thread.start()

                try:
                    event = asyncio.run(self.llm.parse_calendar_event(message.text))
                finally:
                    # Stop typing status
                    stop_typing.set()
                    typing_thread.join()
                
                if not event:
                    self.bot.reply_to(
                        message,
                        "Не удалось обработать сообщение. Попробуйте еще раз."
                    )
                    return
                
                # Update user stats with tokens from LLM response
                tokens_used = event.get("tokens_used", 0) if isinstance(event, dict) else 0
                self.user_manager.update_user_stats(message.from_user.id, tokens_used)
                self.user_manager.add_tokens_used(message.from_user.id, tokens_used)
                
                if not event["result"]:
                    self.bot.reply_to(
                        message,
                        f"Не удалось распарсить событие: {event['comment']}"
                    )
                    return
                
                # Create inline keyboard
                keyboard = telebot.types.InlineKeyboardMarkup()
                keyboard.row(
                    telebot.types.InlineKeyboardButton("✅ Добавить в календарь", callback_data="add")
                )
                
                # Send event preview with buttons
                preview_message = self.bot.reply_to(
                    message,
                    f"Проверьте информацию о событии:\n\n{self._create_event_message(event)}",
                    reply_markup=keyboard
                )
                
                # Store parsed event in memory
                self.parsed_events[preview_message.message_id] = event
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                self.bot.reply_to(message, "Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

        @self.bot.callback_query_handler(func=lambda call: True)
        def handle_callback(call):
            try:
                action = call.data
                
                if action == 'add':
                    # Get parsed event from memory
                    event = self.parsed_events.get(call.message.message_id)
                    if not event:
                        self.bot.answer_callback_query(call.id, "Ошибка: не удалось найти информацию о событии")
                        return
                    
                    # Add event to calendar
                    success, error = self.calendar.add_event(
                        user_id=call.from_user.id,
                        title=event["title"],
                        start_time=event["start_time"],
                        end_time=event["end_time"],
                        description=event["description"],
                        location=event["location"]
                    )
                    
                    if success:
                        self.bot.answer_callback_query(call.id, "✅ Событие добавлено в календарь")
                        # Update button text
                        keyboard = telebot.types.InlineKeyboardMarkup()
                        keyboard.row(
                            telebot.types.InlineKeyboardButton("✅ Успешно добавлено", callback_data="added")
                        )
                        self.bot.edit_message_reply_markup(
                            chat_id=call.message.chat.id,
                            message_id=call.message.message_id,
                            reply_markup=keyboard
                        )
                        # Clean up
                        del self.parsed_events[call.message.message_id]
                    else:
                        self.bot.answer_callback_query(call.id, "❌ Ошибка")
                        self.bot.reply_to(call.message, f"❌ {error}")
                        
                elif action == 'added':
                    self.bot.answer_callback_query(call.id, "Это событие уже добавлено в календарь")
                    
            except Exception as e:
                logger.error(f"Error handling callback: {str(e)}")
                self.bot.answer_callback_query(call.id, "Произошла ошибка")

    def run(self):
        logger.info("Starting bot...")
        self.bot.infinity_polling()
