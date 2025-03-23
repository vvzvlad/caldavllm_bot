import os
from datetime import datetime
from caldav import DAVClient
from loguru import logger
from .config import get_settings

class CalendarManager:
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.calendar = None
        self._init_calendar()

    def _init_calendar(self):
        try:
            # Get settings from environment variables
            caldav_url = os.getenv("CALDAV_URL")
            caldav_username = os.getenv("CALDAV_USERNAME")
            caldav_password = os.getenv("CALDAV_PASSWORD")
            caldav_calendar_name = os.getenv("CALDAV_CALENDAR_NAME")

            if not all([caldav_url, caldav_username, caldav_password, caldav_calendar_name]):
                logger.error("CalDAV credentials not set")
                return

            # Connect to server
            self.client = DAVClient(
                url=caldav_url,
                username=caldav_username,
                password=caldav_password
            )

            # Get list of calendars
            calendars = self.client.principal().calendars()
            
            # Find the required calendar
            for calendar in calendars:
                if calendar.name == caldav_calendar_name:
                    self.calendar = calendar
                    logger.info(f"Successfully connected to calendar: {caldav_calendar_name}")
                    return

            # If calendar not found, use the first available one
            if calendars:
                self.calendar = calendars[0]
                logger.info(f"Using first available calendar: {self.calendar.name}")
            else:
                logger.error("No calendars found")
                raise Exception("No calendars available")

        except Exception as e:
            logger.error(f"Failed to initialize calendar: {str(e)}")
            raise

    def add_event(self, title: str, start_time: str, end_time: str, description: str, location: str) -> bool:
        """
        Adds an event to the calendar
        
        Args:
            title: event title
            start_time: start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: end time in ISO format (YYYY-MM-DDTHH:MM:SS)
            description: event description
            location: event location
            
        Returns:
            bool: True if event was successfully added, False otherwise
        """
        try:
            if not self.calendar:
                logger.error("Calendar not initialized")
                return False

            # Check required fields
            if not title or not start_time or not end_time:
                logger.error("Missing required fields: title, start_time, or end_time")
                return False

            # Get timezone from settings
            timezone = self.settings["caldav"]["timezone"]

            # Create iCal event
            event_data = f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART;TZID={timezone}:{start_time.replace('-', '').replace(':', '')}
DTEND;TZID={timezone}:{end_time.replace('-', '').replace(':', '')}
SUMMARY:{title}"""

            # Add optional fields only if they are not None
            if description:
                event_data += f"\nDESCRIPTION:{description}"
            if location:
                event_data += f"\nLOCATION:{location}"

            event_data += """
END:VEVENT
END:VCALENDAR"""

            # Add event to calendar
            self.calendar.save_event(event_data)
            logger.info(f"Successfully added event: {title}")
            return True

        except Exception as e:
            logger.error(f"Failed to add event: {str(e)}")
            return False
