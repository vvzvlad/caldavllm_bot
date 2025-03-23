from datetime import datetime, timedelta
from caldav import DAVClient
from loguru import logger
from .config import get_settings
from .users import UserManager

class CalendarManager:
    def __init__(self):
        self.settings = get_settings()
        self.user_manager = UserManager()

    async def check_calendar_access(self, url, username, password, calendar_name):
        """Check if we can connect to calendar with given credentials"""
        try:
            # Try to connect
            client = DAVClient(
                url=url,
                username=username,
                password=password
            )
            
            # Try to get principal
            principal = client.principal()
            calendars = principal.calendars()
            
            # Try to find calendar
            calendar = None
            for cal in calendars:
                if cal.name == calendar_name:
                    calendar = cal
                    break
                    
            if not calendar:
                return False, f"Календарь '{calendar_name}' не найден. Доступные календари: {', '.join(cal.name for cal in calendars)}"

            return True, None

        except Exception as e:
            return False, f"Ошибка подключения к календарю: {str(e)}"

    async def add_event(self, user_id, title, start_time, end_time=None, description=None, location=None):
        """Add event to user's calendar"""
        try:
            # Get user credentials
            creds = self.user_manager.get_caldav_credentials(user_id)
            if not creds:
                error = f"Не найдены данные для подключения к календарю. Используйте команду /caldav для настройки"
                logger.error(f"No CalDAV credentials found for user {user_id}")
                return False, error

            # Create client and get calendar
            client = DAVClient(
                url=creds["url"],
                username=creds["username"],
                password=creds["password"]
            )
            
            principal = client.principal()
            calendars = principal.calendars()
            
            calendar = None
            for cal in calendars:
                if cal.name == creds["calendar_name"]:
                    calendar = cal
                    break
                    
            if not calendar:
                error = f"Календарь '{creds['calendar_name']}' не найден. Проверьте название календаря в настройках"
                logger.error(f"Calendar '{creds['calendar_name']}' not found for user {user_id}")
                return False, error

            # Get timezone from settings
            timezone = self.settings["caldav"]["timezone"]

            # If no end time provided, make event 1 hour long
            if not end_time:
                end_time = (datetime.fromisoformat(start_time) + timedelta(hours=1)).isoformat()

            # Format times for iCal
            start_time = start_time.replace('-', '').replace(':', '')
            end_time = end_time.replace('-', '').replace(':', '')

            # Create event data
            event_data = f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:{title}
DTSTART;TZID={timezone}:{start_time}
DTEND;TZID={timezone}:{end_time}"""

            if description:
                event_data += f"\nDESCRIPTION:{description}"
            if location:
                event_data += f"\nLOCATION:{location}"

            event_data += "\nEND:VEVENT\nEND:VCALENDAR"

            # Add event to calendar
            calendar.save_event(event_data)
            logger.info(f"Added event '{title}' to calendar for user {user_id}")
            return True, None

        except Exception as e:
            error = f"Ошибка при добавлении события в календарь: {str(e)}"
            logger.error(f"Error adding event for user {user_id}: {str(e)}")
            return False, error
