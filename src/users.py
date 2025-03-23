import os
import json
from loguru import logger

class UserManager:
    def __init__(self):
        self.data_dir = "data"
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

    def _get_user_file(self, user_id: int) -> str:
        """Get path to user's data file"""
        return os.path.join(self.data_dir, f"user_{user_id}.json")

    def save_caldav_credentials(self, user_id: int, username: str, password: str, url: str, calendar_name: str) -> bool:
        """Save CalDAV credentials for user"""
        try:
            user_file = self._get_user_file(user_id)
            data = {
                "caldav": {
                    "username": username,
                    "password": password,
                    "url": url,
                    "calendar_name": calendar_name
                }
            }
            
            with open(user_file, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Saved CalDAV credentials for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CalDAV credentials for user {user_id}: {str(e)}")
            return False

    def get_caldav_credentials(self, user_id: int) -> dict:
        """Get CalDAV credentials for user"""
        try:
            user_file = self._get_user_file(user_id)
            if not os.path.exists(user_file):
                return None
                
            with open(user_file, 'r') as f:
                data = json.load(f)
                return data.get("caldav")
                
        except Exception as e:
            logger.error(f"Failed to get CalDAV credentials for user {user_id}: {str(e)}")
            return None

    def has_caldav_credentials(self, user_id: int) -> bool:
        """Check if user has CalDAV credentials"""
        return self.get_caldav_credentials(user_id) is not None 