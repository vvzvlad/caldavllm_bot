#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# mypy: disable-error-code="import-untyped"

import json
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List

import httpx
from loguru import logger
from .config import get_settings

class DeepSeekLLM:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings["deepseek_api_key"]
        self.model = self.settings["model"]
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _now(self) -> datetime:
        """
        Time provider used for tests; it prefers src.llm.datetime if available,
        so that `patch("src.llm.datetime")` in tests continues to work.
        """
        try:
            # Local import to avoid circular imports at module load time
            from . import llm as llm_module  # type: ignore
            dt_cls = getattr(llm_module, "datetime", None)
            if dt_cls is not None:
                return dt_cls.now()
        except Exception:
            # Fallback to standard datetime if anything goes wrong
            pass

        from datetime import datetime as _dt
        return _dt.now()

    def _return_datetime(self) -> datetime:
        return self._now()

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
            
            if i == 0: calendar_text.append(f"{date.strftime('%d %B')} — {day_info[0]} (сегодня)")
            elif i <= 6 - current_weekday: calendar_text.append(f"{date.strftime('%d %B')} — {day_info[1]} {day_info[0]}")
            else: calendar_text.append(f"{date.strftime('%d %B')} — {day_info[2]} {day_info[0]}")
            
        return "\n".join(calendar_text)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image file not found: {image_path}")
                return ""
                
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = "image/jpeg"  # Default mime type
                
                if path.suffix.lower() in [".png"]:
                    mime_type = "image/png"
                elif path.suffix.lower() in [".jpg", ".jpeg"]:
                    mime_type = "image/jpeg"
                elif path.suffix.lower() in [".gif"]:
                    mime_type = "image/gif"
                elif path.suffix.lower() in [".webp"]:
                    mime_type = "image/webp"
                    
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            return ""

    async def _make_request(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> Optional[Dict[str, Any]]:
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
                logger.debug(f"DeepSeek API response: {response_json['choices'][0]['message']['content']}")
                return response_json

        except httpx.TimeoutException:
            logger.error("DeepSeek API request timed out")
            return None
        except Exception as e:
            logger.error(f"DeepSeek API request failed: {str(e)}")
            return None

    async def process_with_image(self, image_path: str, text: str, temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        """Send a request with both text and image to the LLM API."""
        try:
            request_id = str(hash(text))[:8]
            logger.info(f"[{request_id}] Starting LLM processing with image: {image_path}")
            
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                logger.error(f"[{request_id}] Failed to encode image")
                return {
                    "result": False,
                    "comment": "Failed to encode image"
                }
            
            # Create multimodal content format
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
            
            messages = [
                {"role": "user", "content": content}
            ]
            
            api_start_time = self._now()
            response = await self._make_request(messages, temperature)
            api_end_time = self._now()
            api_duration = (api_end_time - api_start_time).total_seconds()
            
            logger.info(f"[{request_id}] API request with image completed in {api_duration:.2f} seconds")
            
            if not response:
                logger.error(f"[{request_id}] API request with image failed")
                return {
                    "result": False,
                    "comment": "LLM API error: request timeout or service unavailable"
                }
            
            content = response["choices"][0]["message"]["content"]
            
            # Add token usage information if available
            result = {"result": True, "content": content}
            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image request: {str(e)}")
            return {
                "result": False,
                "comment": f"Error processing image request: {str(e)}"
            }

    async def parse_calendar_event(self, text: str, image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Start time timestamp
        start_time = self._now()
        request_id = str(hash(text))[:8]  # Short ID for request tracking
        logger.info(f"[{request_id}] Starting LLM processing for: {text}")
        
        if image_path:
            logger.info(f"[{request_id}] Image provided: {image_path}")
        
        current_datetime = self._now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = f"""
You are a calendar event parser. Extract the following information from the text and return it in valid JSON format.

WARNING! 200 points are deducted for each mistake. You have 600 points left. Be very attentive

IMPORTANT: INPUT FORMAT
The input may contain:
1. A single message with event information
2. A dialogue/conversation with multiple participants in format:
   Name1: message text
   Name2 (пользователь календаря): message text
   ...
   
When analyzing a dialogue:
- The person marked as "(пользователь календаря)" is the calendar owner
- Events should be created from the perspective of the calendar owner
- Pay attention to WHO is inviting WHOM - the calendar owner's events are what matter
- Example: If "Маша: Давай встретимся в пятницу в 15:00 Петя: Давай" and Петя is the calendar owner, event "has a meeting with Маша on Friday at 15:00"
- Extract event details from the conversation context
- Different parts of the event info may be spread across multiple messages
- You must combine all this information into a single event

IMPORTANT TIMEZONE HANDLING:
1. If timezone is specified (e.g. "по иркутскому времени", "по московскому времени", etc.):
   * Convert all times to Moscow time (UTC+3)
   * Example: "22:22 по иркутскому времени" (UTC+8) should be converted to "17:22" Moscow time
2. If no timezone is specified, assume Moscow time (UTC+3)
3. DO NOT include timezone offset in the output
4. Always return times in Moscow timezone in ISO format without timezone information

Required output fields:
- title: event title. Format based on event type (keep it as short as possible):
    * The headline is the most concise description of what the event is about. 
    * It should be as short as possible, but not so short as to lose information. 
    * Don't write generic words like "Встреча", "Звонок", always be specific about who exactly the meeting is with and who exactly the call is with. Often, you can do without common words at all: For example, not "Доктор", but "Дерматолог". Not "Встреча" but "Обсуждение работы". Not "встреча с HR" but "собеседование".  
    * If I'm asking to be reminded of something, such as "напомни мне вывести деньги", I should write "Вывести деньги". 
    * Use abbreviations: instead of "День рождения Иры", write "ДР Иры". 
    * Don't write long phrases: "Звонок с коллегами по поводу уточнения новых требований к ПО" will be cut off by any calendar and там останется только "Звонок с колл....", и по нему вообще нельзя будет ничего понять, о чем встреча. Вместо этого лучше написать "Звонок Требования ПО"
    *DON'T FANTASIZE. you are obliged to write ONLY WHAT IS in the text given to you. Any fantasy will get you points when it is discovered.
    * Start with capital letters
    * ALWAYS use Russian language! 
    * ALWAYS keep title short and concise (under 100 characters)
- start_time: event start time (in ISO format, Moscow time)
- end_time: event end time (in ISO format, Moscow time). If duration is specified, use it, otherwise set to 1 hour after start_time
- description: detailed description of the event:
    * Any additional information that is not duplicated in the title. If you receive an appointment with a doctor ("Запись к врачу-дерматологу в 14 часов, адрес большая шихстинская, с собой надо взять медкарту, не есть 12 часов, оплата 5000р"), you should put the most important thing in the title: "Дерматолог", time and address - in the time and date fields, and all other information - in the description field: "Взять медкарту, не есть 12 часов, оплата 5000₽". 
    * A good description should fit in 300 characters or less.
    *DON'T FANTASIZE. you are obliged to write ONLY WHAT IS in the text given to you. Any fantasy will get you points when it is discovered.
    * Start with capital letters 
    * ALWAYS use Russian language! 
    * ALWAYS keep descriptions short and concise (under 300 characters)
- location: event location. Format based on event type:
    * For physical locations: "//name//, //address//" (include point name!)
    * For online events: //link// or, if link is not available, blank.
    * Start with capital letters
- result: boolean, true if event was successfully parsed, false if parsed failed, there is not enough information
- comment: string, explanation why parsing failed if result is false, null if result is true

Input date parsing logic:
1. If no date is specified, use current day
2. If only day is specified (e.g. "15th" or "15-го"):
    - If day is in the past for current month, use next month
    - If day is today or in the future for current month, use current month
    - Example: if today is March 14, 2024, and event is "15-го", use March 15, 2024
    - Example: if today is March 20, 2024, and event is "15-го", use April 15, 2024
3. If month is specified (e.g. "September"):
    - Month without specific day is NOT enough information, return result: false
    - If month with day is in the past for current year, use next year
    - Otherwise use current year
4. If date is in the past (including today with past time), move it to next occurrence:
    - If only time is in past for today, move to tomorrow
    - If day is in past for current month, move to next month
    - If full date (day and month) is in the past for current year, use next year
    - Example: if today is March 20, 2024, and event is "15 марта", use March 15, 2025
    - Example: if today is March 20, 2024, and event is "15-го", use April 15, 2024
    - IMPORTANT: When checking if date is in the past, compare the full date (day and month) with current date.
        If the date has already passed this year, use next year
    - CRITICAL: For example, if today is March 20, 2024, and event is "15 марта в 15:00", you MUST use March 15, 2025 because March 15, 2024 is in the past!
5. For relative dates:
    - "в эту субботу" means the next Saturday from today
    - "на субботу" means the next Saturday from today
    - "в следующую субботу" means the Saturday after the next one
    - "в прошлую субботу" means the last Saturday
    - Example: if today is Wednesday March 20, 2024:
        * "в эту субботу" = March 23, 2024
        * "на субботу" = March 23, 2024
        * "в следующую субботу" = March 30, 2024

7.  - If there is no time statement, only a date statement, and it is one day, then return the business hours: 10:00-18:00
    - If the text specifies multiple days (March 20-26), then you MUST return 00:00:00 for start_time and 23:59:59 for end_time
    - Example: "20–28 августа" should be "2024-08-20T00:00:00" to "2024-08-28T23:59:59"
        
Current date and time: {current_datetime}
    Calendar for the next 14 days:
{await self._generate_calendar()}


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

"""
        
        # Prepare messages for API request
        if image_path:
            # If image is provided, create multimodal content for both system and user messages
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                logger.error(f"[{request_id}] Failed to encode image")
                return {
                    "result": False,
                    "comment": "Failed to encode image"
                }
            
            system_content = [{"type": "text", "text": system_prompt}]
            system_message = {"role": "system", "content": system_content}
            
            # If caption exists (not default text), prepend it with "#" marker for high priority
            caption_text = f"# {text}" if text and text != "Добавь это событие в календарь" else text
            user_content = [
                {"type": "text", "text": caption_text},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
            user_message = {"role": "user", "content": user_content}
        else:
            # Regular text request
            system_message = {"role": "system", "content": system_prompt}
            user_message = {"role": "user", "content": text}
        
        messages = [system_message, user_message]
        
        try:
            api_start_time = self._now()
            response = await self._make_request(messages)
            api_end_time = self._now()
            api_duration = (api_end_time - api_start_time).total_seconds()
            
            logger.info(f"[{request_id}] API request completed in {api_duration:.2f} seconds")
            
            if not response:
                logger.error(f"[{request_id}] API request failed")
                return {
                    "result": False,
                    "comment": "LLM API error: request timeout or service unavailable"
                }
                
            content = response["choices"][0]["message"]["content"]
            # Remove markdown code block if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(content)
            
            # Add token usage information
            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)
            
            # Total processing time
            end_time = self._now()
            total_duration = (end_time - start_time).total_seconds()
            logger.info(f"[{request_id}] Total processing completed in {total_duration:.2f} seconds")
            
            return result
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"[{request_id}] Failed to parse DeepSeek response: {str(e)}")
            return {
                "result": False,
                "comment": f"Failed to parse LLM response: {str(e)}"
            }
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {str(e)}")
            return {
                "result": False,
                "comment": f"Unexpected error: {str(e)}"
            }