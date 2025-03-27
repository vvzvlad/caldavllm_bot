import asyncio
from datetime import datetime
from loguru import logger
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from .config import get_settings
from .llm import DeepSeekLLM
from .calendar import CalendarManager
from .users import UserManager

class CalendarBot:
    def __init__(self):
        self.settings = get_settings()
        self.bot = Bot(token=self.settings["telegram_token"])
        self.dp = Dispatcher()
        self.llm = DeepSeekLLM()
        self.calendar = CalendarManager()
        self.user_manager = UserManager()
        self.parsed_events = {}
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

    async def _process_message(self, message: types.Message):
        try:
            if not self.user_manager.has_caldav_credentials(message.from_user.id):
                await message.reply( "Сначала нужно настроить подключение к календарю. Используй команду /caldav, /google или /fastmail" )
                return

            if not self.user_manager.check_token_limit(message.from_user.id):
                await message.reply(
                    f"Достигнут дневной лимит токенов ({self.user_manager.daily_token_limit}). "
                    "Попробуйте завтра."
                )
                return

            logger.info(f"Received message from {message.from_user.id}: {message.text}")

            llm = DeepSeekLLM()
            
            typing_task = asyncio.create_task(self._send_typing_status(message.chat.id))
            try:
                event = await llm.parse_calendar_event(message.text)
            finally:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

            if not event:
                await message.reply( "Внутренняя ошибка при обработке сообщения. Попробуйте позже." )
                return
            
            tokens_used = event.get("tokens_used", 0) if isinstance(event, dict) else 0
            self.user_manager.update_user_stats(message.from_user.id, tokens_used)
            self.user_manager.add_tokens_used(message.from_user.id, tokens_used)
            
            if not event["result"]:
                error_text = event.get("comment", "Неизвестная ошибка")
                await message.reply(  f"❌ {error_text}" )
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
                "Я пойму текст и добавлю событие в твой календарь."
            )
            await message.reply(welcome_text)

        @self.dp.message(Command("google"))
        async def handle_google(message: types.Message):
            try:
                params = message.text.split()
                if len(params) < 3 or len(params) > 4:
                    await message.reply(
                        "Неверный формат команды. Используйте:\n"
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
                        "Неверный формат команды. Используйте:\n"
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

        @self.dp.message()
        async def handle_message(message: types.Message):
            # Создаем таск для обработки сообщения
            asyncio.create_task(self._process_message(message))

        @self.dp.callback_query()
        async def handle_callback(callback_query: types.CallbackQuery):
            # Создаем таск для обработки колбэка
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
