import json
import httpx
from loguru import logger
from typing import Dict, Any, Optional
from .config import get_settings

class DeepSeekLLM:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings["api_key"]
        self.model = self.settings["model"]
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _make_request(self, messages: list[Dict[str, str]], temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                    return None
                
                response_json = response.json()
                logger.debug(f"DeepSeek API response: {response_json}")
                return response_json
        except httpx.TimeoutException:
            logger.error("DeepSeek API request timed out")
            return None
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {str(e)}")
            return None

    async def parse_calendar_event(self, text: str) -> Optional[Dict[str, Any]]:
        system_prompt = """You are a calendar event parser. Extract the following information from the text and return it in valid JSON format.
        Current date and time: 2024-03-22 16:30:00

        Date parsing logic:
        1. If no date is specified, use current day
        2. If only day is specified (e.g. "15th"):
            - If day is in the past for current month, use next month
            - Otherwise use current month
        3. If month is specified (e.g. "September"):
            - If month is in the past for current year, use next year
            - Otherwise use current year

        Required fields:
        - title: event title
        - start_time: event start time (in ISO format)
        - end_time: event end time (in ISO format). If not specified, set it to 1 hour after start_time
        - description: event description
        - location: event location (if any)
        
        Return ONLY the JSON object without any additional text or explanation. Use null for missing fields.
        Example response format:
        {
            "title": "Meeting with client",
            "start_time": "2024-03-22T15:00:00",
            "end_time": "2024-03-22T16:00:00",  # If not specified, set to start_time + 1 hour
            "description": "Project discussion",
            "location": "Office"
        }"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        response = await self._make_request(messages)
        if not response:
            return None
            
        try:
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse DeepSeek response: {str(e)}")
            return None
