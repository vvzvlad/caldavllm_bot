import json
import httpx
from loguru import logger
from typing import Dict, Any, Optional
from datetime import datetime
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
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = f"""You are a calendar event parser. Extract the following information from the text and return it in valid JSON format.
Current date and time: {current_datetime}

WARNING! 200 points are deducted for each mistake. You have 600 points left. Be very attentive
Input date parsing logic:
1. If no date is specified, use current day
2. If only day is specified (e.g. "15th" or "15-го"):
    - If day is in the past for current month, use next month
    - If day is today or in the future for current month, use current month
    - Example: if today is March 15, 2024, and event is "15-го", use March 15, 2024
    - Example: if today is March 20, 2024, and event is "15-го", use April 15, 2024
3. If month is specified (e.g. "September"):
    - Month without specific day is NOT enough information, return result: false
    - If month with day is in the past for current year, use next year
    - Otherwise use current year
4. If date is in the past (including today with past time), move it to next occurrence:
    - If only time is in past for today, move to tomorrow
    - If day is in past for current month, move to next month
    - If full date (day and month) is in the past for current year, move to next year
    - Example: if today is March 20, 2024, and event is "15 марта", use March 15, 2025
    - Example: if today is March 20, 2024, and event is "15-го", use April 15, 2024
5. For specific date with day and month:
    - If date is today or in the future for current year, use current year
    - If date is in the past for current year, use next year
    - Example: if today is March 15, 2024, and event is "15 марта", use March 15, 2024
    - Example: if today is March 20, 2024, and event is "15 марта", use March 15, 2025
    - IMPORTANT: When checking if date is in the past, compare the full date (day and month) with current date.
        If the date has already passed this year, use next year

Required output fields:
- title: event title. Format based on event type (keep it as short as possible):
    * For haircuts/beauty: "Парикмахер"
    * For doctor appointments: "Доктор //full_name//" (if doctor type is not specified) or "//doctor_type//" (use genitive case, e.g. "Дерматолог", "Психолог", "Хирург", "Стоматолог")
    * For masterclasses: "//short_title//" (without prefixes like "Онлайн мастер-класс:")
    * ALWAYS use Russian language! (e.g. "Встреча с клиентом")
- start_time: event start time (in ISO format)
- end_time: event end time (in ISO format). If duration is specified, use it, otherwise set to 1 hour after start_time
- description: detailed description of the event:
    * For meetings: use the exact text from input that describes the meeting (including location if mentioned)
    * For doctor appointments: "Прием у доктора //surname//" (do not decline the word "доктора" or surname, do not add dot at the end)
    * For online sessions: "Сессия с психологом //psychologist_name//" (use instrumental case for psychologist name)
    * For masterclasses and courses: use the full title without prefixes
    * For events with speakers/guests: "Встреча с //guest_name//, //guest_role//. //short_topic//" (keep it under 100 characters)
    * ALWAYS use Russian language! 
    * ALWAYS keep descriptions short and concise (under 100 characters)
- location: event location. Format based on event type:
    * For physical locations: "//address//" (do not include clinic name at the start unless it's part of the official address)
    * For online events: //link// or, if link is not available, "Онлайн" (if it's an online event)
    * ALWAYS use Russian language! 
- result: boolean, true if event was successfully parsed, false if parsed failed, there is not enough information
- comment: string, explanation why parsing failed if result is false, null if result is true

Return ONLY the JSON object without any additional text or explanation. Use null for missing fields.
Example response format:
{{
    "title": "Встреча с клиентом",
    "start_time": "2024-03-22T15:00:00",
    "end_time": "2024-03-22T16:00:00",  # If not specified, set to start_time + 1 hour
    "description": "Встреча с клиентом",  # Same as title if no specific description
    "location": "Офис",  # Use nominative case
    "result": true,
    "comment": null
}}

Example of failed parsing (if there is not enough information, e.g. only month without day):
{{
    "result": false,
    "comment": "Недостаточно информации о дате"
}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        response = await self._make_request(messages)
        if not response:
            return None
            
        try:
            content = response["choices"][0]["message"]["content"]
            # Remove markdown code block if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            return json.loads(content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse DeepSeek response: {str(e)}")
            return None
