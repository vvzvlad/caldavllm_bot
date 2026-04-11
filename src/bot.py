import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Awaitable
from loguru import logger
from aiogram import Bot, Dispatcher, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import Command
from .config import get_settings
from .llm import get_llm
from .calendar import CalendarManager
from .users import UserManager


@dataclass
class MessageBatch:
    """Represents a batch of messages from a single user"""
    messages: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # Paths to downloaded images
    timer: asyncio.TimerHandle | None = None
    first_message_time: float = 0.0
    first_message: types.Message | None = None  # Reference to first message for reply
    owner_user_id: int | None = None  # ID of the user who forwarded the dialogue (calendar owner)


class MessageBatcher:
    """Manages message batching with debounce timeout"""
    
    def __init__(
        self,
        process_callback: Callable[["MessageBatch", types.Message], Awaitable[None]],
        batch_timeout: float = 2.0,
        max_batch_size: int = 20
    ):
        """
        Args:
            process_callback: Async function to call when batch is ready
                                Signature: async def callback(batch: MessageBatch, first_message: types.Message)
            batch_timeout: Seconds to wait before processing batch
            max_batch_size: Maximum messages per batch (triggers immediate processing)
        """
        self._batches: dict[int, MessageBatch] = {}  # user_id -> batch
        self._process_callback = process_callback
        self._batch_timeout = batch_timeout
        self._max_batch_size = max_batch_size
    
    def _get_sender_name(self, message: types.Message) -> str:
        """Extract sender name from message, handling forwarded messages"""
        # Check if it's a forwarded message
        if message.forward_from: # Forwarded from a user who allows linking
            return message.forward_from.first_name or message.forward_from.username or "Unknown"
        elif message.forward_sender_name: # Forwarded from a user who hides their account
            return message.forward_sender_name
        elif message.from_user: # Regular message from user
            return message.from_user.first_name or message.from_user.username or "Unknown"
        return "Unknown"
    
    def _get_sender_user_id(self, message: types.Message) -> int | None:
        """Extract sender user_id from message, handling forwarded messages"""
        # Check if it's a forwarded message
        if message.forward_from:   # Forwarded from a user who allows linking
            return message.forward_from.id
        elif message.forward_sender_name: # Forwarded from a user who hides their account - no user_id available
            return None
        elif message.from_user: # Regular message from user
            return message.from_user.id
        return None
    
    def _format_message_text(self, name: str, text: str, is_calendar_owner: bool = False) -> str:
        """Format message as 'Name: text' with newlines removed
        
        Args:
            name: Sender name
            text: Message text
            is_calendar_owner: If True, adds "(пользователь календаря)" marker after name
        """
        # Remove newlines and carriage returns, replace with space
        clean_text = text.replace('\n', ' ').replace('\r', ' ')
        # Remove multiple consecutive spaces
        while '  ' in clean_text:
            clean_text = clean_text.replace('  ', ' ')
        
        if is_calendar_owner:
            return f"{name} (пользователь календаря): {clean_text.strip()}"
        return f"{name}: {clean_text.strip()}"
    
    async def add_message(
        self,
        user_id: int,
        text: str | None,
        image_path: str | None,
        message: types.Message
    ) -> None:
        """Add a message to the user's batch, resetting the timer"""
        
        # Cancel existing timer if any
        if user_id in self._batches:
            batch = self._batches[user_id]
            if batch.timer:
                batch.timer.cancel()
                batch.timer = None
        else:
            # Create new batch
            # owner_user_id is the user who forwards the dialogue (from_user.id of first message)
            batch = MessageBatch(
                first_message_time=asyncio.get_event_loop().time(),
                first_message=message,
                owner_user_id=message.from_user.id if message.from_user else None
            )
            self._batches[user_id] = batch
        
        # Add message content to batch with sender name
        if text:
            sender_name = self._get_sender_name(message)
            # Check if the sender is the calendar owner
            sender_user_id = self._get_sender_user_id(message)
            is_calendar_owner = (
                batch.owner_user_id is not None
                and sender_user_id is not None
                and sender_user_id == batch.owner_user_id
            )
            formatted_message = self._format_message_text(sender_name, text, is_calendar_owner)
            batch.messages.append(formatted_message)
        if image_path:
            batch.images.append(image_path)
        
        # Check if max batch size reached
        total_items = len(batch.messages) + len(batch.images)
        if total_items >= self._max_batch_size:
            logger.info(f"Max batch size ({self._max_batch_size}) reached for user {user_id}, processing immediately")
            await self._process_batch(user_id)
            return
        
        # Schedule processing with debounce
        self._schedule_processing(user_id)
    
    def _schedule_processing(self, user_id: int) -> None:
        """Schedule batch processing after timeout"""
        if user_id not in self._batches:
            return
            
        batch = self._batches[user_id]
        
        # Cancel existing timer
        if batch.timer:
            batch.timer.cancel()
        
        # Create new timer using call_later
        loop = asyncio.get_event_loop()
        batch.timer = loop.call_later(
            self._batch_timeout,
            lambda: asyncio.create_task(self._process_batch(user_id))
        )
    
    async def _process_batch(self, user_id: int) -> None:
        """Process the batch for a user"""
        if user_id not in self._batches:
            return
        
        batch = self._batches.pop(user_id)
        
        # Cancel timer if still active
        if batch.timer:
            batch.timer.cancel()
            batch.timer = None
        
        if not batch.messages and not batch.images:
            logger.warning(f"Empty batch for user {user_id}, skipping")
            return
        
        logger.info(
            f"Processing batch for user {user_id}: "
            f"{len(batch.messages)} messages, {len(batch.images)} images"
        )
        
        try:
            await self._process_callback(batch, batch.first_message)
        except Exception as e:
            logger.error(f"Error processing batch for user {user_id}: {e}")
            # Clean up images on error
            for img_path in batch.images:
                if img_path and os.path.exists(img_path):
                    try:
                        os.unlink(img_path)
                    except Exception as del_err:
                        logger.error(f"Failed to delete temp image: {del_err}")


class CalendarBot:
    def __init__(self):
        self.settings = get_settings()
        api_server = self.settings.get("telegram_bot_api_server")
        # Use custom Telegram Bot API server URL if provided (e.g., local Bot API Server instance)
        # TelegramAPIServer is passed to AiohttpSession via 'api' parameter, not to Bot directly
        server = TelegramAPIServer.from_base(api_server) if api_server else TelegramAPIServer.from_base("https://api.telegram.org")
        self.bot = Bot(token=self.settings["telegram_token"], session=AiohttpSession(api=server))
        self.dp = Dispatcher()
        # LLM backend is selected via configuration in src.config / src.llm
        self.llm = get_llm()
        self.calendar = CalendarManager()
        self.user_manager = UserManager()
        self.parsed_events = {}
        
        # Initialize message batcher with settings from config
        self.message_batcher = MessageBatcher(
            process_callback=self._process_batched_messages,
            batch_timeout=self.settings.get("batch_timeout", 2.0),
            max_batch_size=self.settings.get("max_batch_size", 20)
        )
        
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
        if number >= 1000: return f"{number // 1000}к"
        return str(number)

    def _create_event_message(self, event: dict) -> str:
        """Create formatted event message"""
        parts = []
        
        if event.get("title"): parts.append(f"📌 {event['title']}")
        if event.get("start_time"): parts.append(f"🕒 Начало: {self._format_datetime(event['start_time'])}")
        if event.get("end_time"): parts.append(f"🕒 Конец: {self._format_datetime(event['end_time'])}")
        if event.get("location"): parts.append(f"📍 {event['location']}")
        if event.get("description"): parts.append(f"📝 {event['description']}")
        return "\n".join(parts)

    async def _send_typing_status(self, chat_id: int):
        """Send typing status every 4 seconds until cancelled"""
        while True:
            try:
                await self.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)  # Telegram typing status lasts 5 seconds
            except Exception as e:
                logger.error(f"Error sending typing status: {str(e)}")
                break

    async def _download_photo(self, message: types.Message) -> str:
        """Download photo from message and return the local path"""
        try:
            # Get the photo with highest resolution
            photo = message.photo[-1]
            
            # Create a temporary file to save the photo
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_path = temp_file.name
                
            # Download the photo
            await self.bot.download(photo.file_id, destination=temp_path)
            logger.info(f"Downloaded photo to {temp_path}")
            
            return temp_path
        except Exception as e:
            logger.error(f"Failed to download photo: {e}")
            return None

    async def _process_message_with_image(self, message: types.Message, text: str = None, image_path: str = None):
        """Process a message with optional image and text"""
        try:
            if not self.user_manager.has_caldav_credentials(message.from_user.id):
                await message.reply("Сначала нужно настроить подключение к календарю. Используй команду /caldav, /google или /fastmail")
                return

            if not self.user_manager.check_token_limit(message.from_user.id):
                await message.reply(
                    f"Достигнут дневной лимит токенов ({self.user_manager.daily_token_limit}). "
                    "Попробуйте завтра."
                )
                return
            
            # Use message caption if text is not provided
            if text is None and message.caption:
                text = message.caption
            
            # Default text if none provided
            if not text:
                text = "Добавь это событие в календарь"
                
            logger.info(f"Processing message from {message.from_user.id}: text={text}, has_image={image_path is not None}")

            
            typing_task = asyncio.create_task(self._send_typing_status(message.chat.id))
            try:
                event = await self.llm.parse_calendar_event(text, image_path)
            finally:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

            # Clean up temporary file if it exists
            if image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                    logger.info(f"Deleted temporary file {image_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {e}")

            if not event:
                await message.reply("Internal error processing the message. Please try again later.")
                return
            
            tokens_used = event.get("tokens_used", 0) if isinstance(event, dict) else 0
            self.user_manager.update_user_stats(message.from_user.id, tokens_used)
            self.user_manager.add_tokens_used(message.from_user.id, tokens_used)
            
            if not event["result"]:
                error_text = event.get("comment", "Unknown error")
                await message.reply(f"❌ {error_text}")
                return
            
            keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="✅ Добавить в календарь", callback_data="add")]
            ])
            
            preview_message = await message.reply(
                f"Проверьте информацию о событии:\n\n{self._create_event_message(event)}",
                reply_markup=keyboard
            )
            
            self.parsed_events[preview_message.message_id] = event
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await message.reply("Произошла ошибка при обработке сообщения. Попробуйте еще раз.")

    async def _process_message(self, message: types.Message):
        """Process a regular text message"""
        await self._process_message_with_image(message, text=message.text)

    async def _process_photo(self, message: types.Message):
        """Process a photo message with optional caption"""
        try:
            image_path = await self._download_photo(message)
            if not image_path:
                await message.reply("Failed to process the image. Please try again.")
                return
                
            await self._process_message_with_image(message, text=message.caption, image_path=image_path)
        except Exception as e:
            logger.error(f"Error processing photo: {str(e)}")
            await message.reply("Error processing the image. Please try again.")

    async def _process_callback(self, callback_query: types.CallbackQuery):
        try:
            action = callback_query.data
            
            if action == 'add':
                event = self.parsed_events.get(callback_query.message.message_id)
                if not event:
                    await callback_query.answer("Ошибка: не удалось найти информацию о событии")
                    return
                
                success, error = await self.calendar.add_event(
                    user_id=callback_query.from_user.id,
                    title=event["title"],
                    start_time=event["start_time"],
                    end_time=event["end_time"],
                    description=event["description"],
                    location=event["location"]
                )
                
                if success:
                    await callback_query.answer("✅ Событие добавлено в календарь")
                    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
                        [types.InlineKeyboardButton(text="✅ Успешно добавлено", callback_data="added")]
                    ])
                    await callback_query.message.edit_reply_markup(reply_markup=keyboard)
                    del self.parsed_events[callback_query.message.message_id]
                else:
                    await callback_query.answer("❌ Ошибка")
                    await callback_query.message.reply(f"❌ {error}")
                    
            elif action == 'added':
                await callback_query.answer("Это событие уже добавлено в календарь")
                
        except Exception as e:
            logger.error(f"Error handling callback: {str(e)}")
            await callback_query.answer("Произошла ошибка")

    async def _process_batched_messages(self, batch: MessageBatch, first_message: types.Message) -> None:
        """Process a batch of messages as a single unit"""
        # Combine all message texts - each message is already formatted as "Name: text"
        combined_text = "\n".join(batch.messages) if batch.messages else None
        
        # Use first image if any, or None
        image_path = batch.images[0] if batch.images else None
        
        logger.info(
            f"Processing batched messages for user {first_message.from_user.id}: "
            f"combined_text_length={len(combined_text) if combined_text else 0}, "
            f"using_image={image_path is not None}"
        )
        
        # Process the combined message
        await self._process_message_with_image(
            first_message,
            text=combined_text,
            image_path=image_path
        )
        
        # Clean up any additional images (first one is cleaned up by _process_message_with_image)
        for img_path in batch.images[1:]:
            if img_path and os.path.exists(img_path):
                try:
                    os.unlink(img_path)
                    logger.info(f"Deleted additional temp image {img_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temp image: {e}")

    def _setup_handlers(self):
        @self.dp.message(Command("start"))
        async def handle_start(message: types.Message):
            welcome_text = (
                "👋 Привет! Я бот для добавления событий в календарь.\n\n"
                "Для настройки календаря используй одну из команд:\n\n"
                "📧 Для Google Calendar:\n"
                "/google account password [calendar] - Быстрая настройка Google Calendar\n\n"
                "📧 Для FastMail:\n"
                "/fastmail account password [calendar] - Быстрая настройка FastMail\n\n"
                "🔧 Для других CalDAV календарей:\n"
                "/caldav username password url calendar_name\n\n"
                "После настройки просто напиши мне о событии, например:\n"
                "• Завтра в 15:00 встреча с клиентом\n"
                "• 25 марта в 11 утра лекция о японском символизме\n"
                "• Встреча в офисе в понедельник в 10:00\n\n"
                "Ты также можешь отправить мне изображение приглашения или афиши события.\n\n"
                "Я пойму текст и добавлю событие в твой календарь."
            )
            await message.reply(welcome_text)

        @self.dp.message(Command("google"))
        async def handle_google(message: types.Message):
            try:
                params = message.text.split()
                if len(params) < 3 or len(params) > 4:
                    await message.reply(
                        "/google username password [calendar]\n\n"
                        "❗️ username - ваш имя пользоватея (можно с @gmail.com, можно без)\n"
                        "❗️ password — ваш пароль приложения. Для получения пароля:\n"
                        "1. Включить двухфакторную аутентификацию (2FA)\n"
                        "   • Без 2FA пароли приложений недоступны\n"
                        "   • Обычный пароль от аккаунта не подойдет\n\n"
                        "2. Создать пароль приложения:\n"
                        "   • Перейдите на https://myaccount.google.com/apppasswords или перейдите Security->2-Step Verification->App passwords\n"
                        "   • Введите название (например 'Calendar Bot')\n"
                        "   • Используйте сгенерированный пароль в команде выше\n\n"
                        "❗️ calendar - название вашего календаря (опционально)\n"
                        "   • Если не указано, будет использован основной календарь",
                        disable_web_page_preview=True
                    )
                    return

                _, username, password, *calendar_params = params
                if not username.endswith("@gmail.com"):
                    username = f"{username}@gmail.com"

                url = f"https://www.google.com/calendar/dav/{username}/events"
                calendar_name = calendar_params[0] if calendar_params else username

                status_message = await message.reply("🔄 Проверка подключения к Google Calendar...")

                success, error = await self.calendar.check_calendar_access(url, username, password, calendar_name)
                if not success:
                    await status_message.edit_text(f"❌ {error}")
                    return

                success = self.user_manager.save_caldav_credentials(
                    message.from_user.id,
                    username,
                    password,
                    url,
                    calendar_name
                )

                if success:
                    await status_message.edit_text("✅ Google Calendar подключен успешно! Можете добавлять события.")
                else:
                    await status_message.edit_text("❌ Не удалось сохранить настройки. Попробуйте еще раз.")

            except Exception as e:
                logger.error(f"Error setting up Google Calendar: {str(e)}")
                await message.reply("Произошла ошибка при настройке. Попробуйте еще раз.")

        @self.dp.message(Command("fastmail"))
        async def handle_fastmail(message: types.Message):
            try:
                params = message.text.split()
                if len(params) < 3 or len(params) > 4:
                    await message.reply(
                        "/fastmail username password [calendar]\n\n"
                        "❗️ username - ваш имя пользоватея (можно с @fastmail.com, можно без)\n"
                        "❗️ password — ваш пароль приложения. Для получения пароля:\n"
                        "1. Перейдите на https://app.fastmail.com/settings/security/apps\n"
                        "2. Нажмите 'New App Password'\n"
                        "3. Выберите 'Calendars (CalDAV)'(так доступ у бота будет только к календарю, а не ко всей почте) и выберите название, например 'Calendar Bot'\n"
                        "4. Используйте сгенерированный пароль в команде выше\n\n"
                        "❗️ calendar - название вашего календаря (опционально)\n"
                        "   • Если не указано, будет использован основной календарь",
                        disable_web_page_preview=True
                    )
                    return

                _, username, password, *calendar_params = params
                if not username.endswith("@fastmail.com"):
                    username = f"{username}@fastmail.com"

                # Get username without domain for default calendar name
                default_calendar = username.split('@')[0]
                calendar_name = calendar_params[0] if calendar_params else default_calendar

                url = "https://caldav.fastmail.com/dav/"

                status_message = await message.reply("🔄 Проверка подключения к FastMail...")

                success, error = await self.calendar.check_calendar_access(url, username, password, calendar_name)
                if not success:
                    await status_message.edit_text(f"❌ {error}")
                    return

                success = self.user_manager.save_caldav_credentials(
                    message.from_user.id,
                    username,
                    password,
                    url,
                    calendar_name
                )

                if success:
                    await status_message.edit_text("✅ FastMail подключен успешно! Можете добавлять события.")
                else:
                    await status_message.edit_text("❌ Не удалось сохранить настройки. Попробуйте еще раз.")

            except Exception as e:
                logger.error(f"Error setting up FastMail: {str(e)}")
                await message.reply("Произошла ошибка при настройке. Попробуйте еще раз.")

        @self.dp.message(Command("caldav"))
        async def handle_caldav(message: types.Message):
            try:
                params = message.text.split()
                if len(params) != 5:
                    await message.reply(
                        "Неверный формат команды. Используйте:\n /caldav username password url calendar_name\n\n"
                        "Например:\n/caldav user@fastmail.com strong_password https://caldav.fastmail.com/dav/ main_calendar",
                        disable_web_page_preview=True
                    )
                    return

                _, username, password, url, calendar_name = params

                status_message = await message.reply("🔄 Проверка подключения к календарю...")

                success, error = await self.calendar.check_calendar_access(url, username, password, calendar_name)
                if not success:
                    await status_message.edit_text(f"❌ {error}")
                    return

                success = self.user_manager.save_caldav_credentials(
                    message.from_user.id,
                    username,
                    password,
                    url,
                    calendar_name
                )

                if success:
                    await status_message.edit_text(
                        "✅ Календарь доступен, настройки успешно сохранены! Теперь вы можете добавлять события."
                    )
                else:
                    await status_message.edit_text(
                        "❌ Не удалось сохранить настройки календаря. Попробуйте еще раз."
                    )

            except Exception as e:
                logger.error(f"Error setting up CalDAV: {str(e)}")
                await message.reply(
                    "Произошла ошибка при настройке календаря. Попробуйте еще раз.",
                    disable_web_page_preview=True
                )

        @self.dp.message(Command("stats"))
        async def handle_stats(message: types.Message):
            stats = self.user_manager.get_user_stats(message.from_user.id)
            if not stats:
                await message.reply("У вас пока нет статистики использования.")
                return
                
            
            remaining_tokens = self.user_manager.get_remaining_tokens(message.from_user.id)
            
            stats_text = (
                "📊 Ваша статистика:\n\n"
                f"Потрачено токенов сегодня: {self._format_number(self.user_manager.daily_token_limit - remaining_tokens)} из лимита {self._format_number(self.user_manager.daily_token_limit)}\n"
                f"Вы сделали {stats['requests_count']} запросов, всего использовали токенов: {self._format_number(stats['total_tokens'])}, "
                f"в среднем {stats['total_tokens'] // max(1, stats['requests_count'])} токенов на запрос"
            )
            await message.reply(stats_text)

        @self.dp.message(lambda message: message.photo)
        async def handle_photo(message: types.Message):
            # Download image first, then add to batch
            image_path = await self._download_photo(message)
            if not image_path:
                await message.reply("Failed to process the image. Please try again.")
                return
            await self.message_batcher.add_message(
                user_id=message.from_user.id,
                text=message.caption,
                image_path=image_path,
                message=message
            )

        @self.dp.message()
        async def handle_message(message: types.Message):
            # Add message to batch for debounced processing
            await self.message_batcher.add_message(
                user_id=message.from_user.id,
                text=message.text,
                image_path=None,
                message=message
            )

        @self.dp.callback_query()
        async def handle_callback(callback_query: types.CallbackQuery):
            # Create task for processing callback
            asyncio.create_task(self._process_callback(callback_query))

    async def _advertise_commands(self):
        """Register bot commands in Telegram"""
        commands = [
            types.BotCommand(command="start", description="Начать работу"),
            types.BotCommand(command="google", description="Настройка Google Calendar"),
            types.BotCommand(command="fastmail", description="Настройка FastMail"),
            types.BotCommand(command="caldav", description="Настройка CalDAV"),
            types.BotCommand(command="stats", description="Показать статистику использования")
        ]
        await self.bot.set_my_commands(commands)

    async def start(self):
        logger.info("Starting bot...")
        try:
            await self._advertise_commands()
            await self.dp.start_polling(self.bot)
        finally:
            await self.bot.session.close()
