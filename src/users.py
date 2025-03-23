import os
import json
from datetime import datetime, date
from typing import Dict, Tuple

from loguru import logger
from .config import get_settings

class UserManager:
    def __init__(self):
        self.data_dir = "data"
        self._ensure_data_dir()
        # Store daily token usage in memory: {user_id: (date, tokens)}
        self.daily_token_usage: Dict[int, Tuple[date, int]] = {}
        self.settings = get_settings()
        self.daily_token_limit = self.settings["daily_token_limit"]

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
            
            # Load existing data if file exists
            data = {}
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Update caldav credentials
            data["caldav"] = {
                "username": username,
                "password": password,
                "url": url,
                "calendar_name": calendar_name
            }
            
            with open(user_file, 'w', encoding='utf-8') as f:
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
                
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("caldav")
                
        except Exception as e:
            logger.error(f"Failed to get CalDAV credentials for user {user_id}: {str(e)}")
            return None

    def has_caldav_credentials(self, user_id: int) -> bool:
        """Check if user has CalDAV credentials"""
        return self.get_caldav_credentials(user_id) is not None

    def update_user_stats(self, user_id: int, tokens_used: int = None) -> bool:
        """Update user statistics"""
        try:
            user_file = self._get_user_file(user_id)
            
            # Load existing data if file exists
            data = {}
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Initialize stats if not exists
            if "stats" not in data:
                data["stats"] = {
                    "requests_count": 0,
                    "total_tokens": 0,
                    "last_request": None
                }
            
            # Update stats
            data["stats"]["requests_count"] += 1
            if tokens_used:
                data["stats"]["total_tokens"] += tokens_used
            data["stats"]["last_request"] = datetime.now().isoformat()
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            
            logger.info(f"Updated stats for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update stats for user {user_id}: {str(e)}")
            return False

    def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics"""
        try:
            user_file = self._get_user_file(user_id)
            if not os.path.exists(user_file):
                return None
                
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("stats")
                
        except Exception as e:
            logger.error(f"Failed to get stats for user {user_id}: {str(e)}")
            return None

    def check_token_limit(self, user_id: int) -> bool:
        """Check if user has not exceeded daily token limit"""
        today = date.today()
        
        if user_id not in self.daily_token_usage:
            return True
            
        last_date, tokens = self.daily_token_usage[user_id]
        
        # Reset counter if it's a new day
        if last_date < today:
            self.daily_token_usage[user_id] = (today, 0)
            return True
            
        return tokens < self.daily_token_limit

    def get_remaining_tokens(self, user_id: int) -> int:
        """Get remaining tokens for today"""
        today = date.today()
        
        if user_id not in self.daily_token_usage:
            return self.daily_token_limit
            
        last_date, tokens = self.daily_token_usage[user_id]
        
        if last_date < today:
            return self.daily_token_limit
            
        return max(0, self.daily_token_limit - tokens)

    def add_tokens_used(self, user_id: int, tokens: int) -> None:
        """Add tokens to user's daily usage"""
        today = date.today()
        
        if user_id not in self.daily_token_usage:
            self.daily_token_usage[user_id] = (today, tokens)
        else:
            last_date, current_tokens = self.daily_token_usage[user_id]
            if last_date < today:
                self.daily_token_usage[user_id] = (today, tokens)
            else:
                self.daily_token_usage[user_id] = (today, current_tokens + tokens) 