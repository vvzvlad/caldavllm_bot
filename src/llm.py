import json
import httpx
import asyncio
from loguru import logger
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
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

    def _return_datetime(self) -> datetime:
        return datetime.now()

    async def _generate_calendar(self) -> str:
        from calendar import day_name
        
        calendar_text = []
        current_date = self._return_datetime()
        current_weekday = current_date.weekday()
        
        for i in range(14):
            date = current_date + timedelta(days=i)
            day_info = {
                'Monday': ('понедельник', 'этот', 'следующий'),
                'Tuesday': ('вторник', 'этот', 'следующий'),
                'Wednesday': ('среда', 'эта', 'следующая'),
                'Thursday': ('четверг', 'этот', 'следующий'),
                'Friday': ('пятница', 'эта', 'следующая'),
                'Saturday': ('суббота', 'эта', 'следующая'),
                'Sunday': ('воскресенье', 'это', 'следующее')
            }[day_name[date.weekday()]]
            
            if i == 0:
                calendar_text.append(f"{date.strftime('%d %B')} — {day_info[0]} (сегодня)")
            elif i <= 6 - current_weekday:  # если день на этой неделе
                calendar_text.append(f"{date.strftime('%d %B')} — {day_info[1]} {day_info[0]}")
            else:
                calendar_text.append(f"{date.strftime('%d %B')} — {day_info[2]} {day_info[0]}")
            
        return "\n".join(calendar_text)

    async def _make_request(self, messages: list[Dict[str, str]], temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature
                    }
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
        # Метка времени начала обработки
        start_time = datetime.now()
        request_id = str(hash(text))[:8]  # Короткий ID для отслеживания запроса
        logger.info(f"[{request_id}] Starting LLM processing for: {text}")
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = f"""You are a calendar event parser. Extract the following information from the text and return it in valid JSON format.


WARNING! 200 points are deducted for each mistake. You have 600 points left. Be very attentive

IMPORTANT TIMEZONE HANDLING:
1. If timezone is specified (e.g. "по иркутскому времени", "по московскому времени", etc.):
   - Convert all times to Moscow time (UTC+3)
   - Example: "22:22 по иркутскому времени" (UTC+8) should be converted to "17:22" Moscow time
2. If no timezone is specified, assume Moscow time (UTC+3)
3. DO NOT include timezone offset in the output
4. Always return times in Moscow timezone in ISO format without timezone information

Required output fields:
- title: event title. Format based on event type (keep it as short as possible):
    * For haircuts/beauty: "Парикмахер"
    * For doctor appointments: "Доктор //full_name//" (if doctor type is not specified) or "//doctor_type//" (use genitive case, e.g. "Дерматолог", "Психолог", "Хирург", "Стоматолог")
    * For masterclasses: "//short_title//" (without prefixes like "Онлайн мастер-класс:")
    * ALWAYS use Russian language! 
- start_time: event start time (in ISO format, Moscow time)
- end_time: event end time (in ISO format, Moscow time). If duration is specified, use it, otherwise set to 1 hour after start_time
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
6. For relative dates:
    - "в эту субботу" means the next Saturday from today
    - "на субботу" means the next Saturday from today
    - "в следующую субботу" means the Saturday after the next one
    - "в прошлую субботу" means the last Saturday
    - Example: if today is Wednesday March 20, 2024:
        * "в эту субботу" = March 23, 2024
        * "на субботу" = March 23, 2024
        * "в следующую субботу" = March 30, 2024

7.  - If there is no time statement, only a date statement, and it is one day, then return the business hours: 10:00-18:00
    - If the text specifies multiple days (March 20-26), then you should return 00:00:00 March 20-23:59:59 March 26, i.e. full days.
        
Return ONLY the JSON object without any additional text or explanation. Use null for missing fields.
Example response format:
{{
    "title": "//название события//",
    "start_time": "2024-03-22T15:00:00",
    "end_time": "2024-03-22T16:00:00",  # If not specified, set to start_time + 1 hour
    "description": "//описание события//",  # blank if no specific description
    "location": "//место события//",  # Use nominative case
    "result": true,
    "comment": null
}}

Example of failed parsing (if there is not enough information, e.g. only month without day):
{{
    "result": false,
    "comment": "Недостаточно информации о дате" #Описание того, почему не удалось распознать событие
}}


Current date and time: {current_datetime}
    Calendar for the next 14 days:
{await self._generate_calendar()}

"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            api_start_time = datetime.now()
            response = await self._make_request(messages)
            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()
            
            logger.info(f"[{request_id}] API request completed in {api_duration:.2f} seconds")
            
            if not response:
                logger.error(f"[{request_id}] API request failed")
                return {
                    "result": False,
                    "comment": "Ошибка при обращении к LLM: таймаут запроса или сервис недоступен"
                }
                
            content = response["choices"][0]["message"]["content"]
            # Remove markdown code block if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(content)
            
            # Add token usage information
            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)
            
            # Общее время обработки
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            logger.info(f"[{request_id}] Total processing completed in {total_duration:.2f} seconds")
            
            return result
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"[{request_id}] Failed to parse DeepSeek response: {str(e)}")
            return {
                "result": False,
                "comment": f"Ошибка при обработке ответа LLM: {str(e)}"
            }
