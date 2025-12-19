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
from typing import Dict, Any, Optional, List, cast
import importlib
import logging

from .config import get_settings

logger = logging.getLogger(__name__)
httpx = importlib.import_module('httpx')


class GroqLLM:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings["api_key"]
        self.model_text = "openai/gpt-oss-120b"
        self.model_ocr = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
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
            elif i <= 6 - current_weekday:
                calendar_text.append(f"{date.strftime('%d %B')} — {day_info[1]} {day_info[0]}")
            else:
                calendar_text.append(f"{date.strftime('%d %B')} — {day_info[2]} {day_info[0]}")

        return "\n".join(calendar_text)

    def _encode_image_to_base64(self, image_path: str) -> str:
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error("Image file not found in _encode_image_to_base64: %s", image_path)
                return ""

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                mime_type = "image/jpeg"

                if path.suffix.lower() in [".png"]:
                    mime_type = "image/png"
                elif path.suffix.lower() in [".jpg", ".jpeg"]:
                    mime_type = "image/jpeg"
                elif path.suffix.lower() in [".gif"]:
                    mime_type = "image/gif"
                elif path.suffix.lower() in [".webp"]:
                    mime_type = "image/webp"

                return f"data:{mime_type};base64,{encoded_string}"
        except OSError as e:
            logger.error("Failed to encode image in _encode_image_to_base64: %s", e)
            return ""

    async def _make_request(self, messages: List[Dict[str, Any]], temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model_text,
                        "messages": messages,
                        "temperature": temperature,
                        "reasoning_effort": "high",
                        "stream": False,
                        "response_format": {"type": "json_object"},
                    }
                )

                if response.status_code != 200:
                    logger.error("Groq API error %s in _make_request: %s", response.status_code, response.text)
                    return None

                response_json = response.json()
                try:
                    logger.debug(
                        "Groq API response in _make_request: %s",
                        response_json['choices'][0]['message']['content']
                    )
                except (KeyError, TypeError):
                    logger.debug("Groq API response in _make_request: content is missing in choices[0]")
                return response_json

        except httpx.TimeoutException:
            logger.error("Groq API request timeout in _make_request (30s)")
            return None
        except httpx.RequestError as e:
            logger.error("Groq API request failed in _make_request: %s", e)
            return None

    async def _make_ocr_request(self, messages: List[Dict[str, Any]], temperature: float = 0.3) -> Optional[Dict[str, Any]]:
        """Make OCR request using model_ocr without JSON response format."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model_ocr,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False,
                    }
                )

                if response.status_code != 200:
                    logger.error("Groq OCR API error %s in _make_ocr_request: %s", response.status_code, response.text)
                    return None

                response_json = response.json()
                try:
                    logger.debug(
                        "Groq OCR API response in _make_ocr_request: %s",
                        response_json['choices'][0]['message']['content']
                    )
                except (KeyError, TypeError):
                    logger.debug("Groq OCR API response in _make_ocr_request: content is missing in choices[0]")
                return response_json

        except httpx.TimeoutException:
            logger.error("Groq OCR API request timeout in _make_ocr_request (60s)")
            return None
        except httpx.RequestError as e:
            logger.error("Groq OCR API request failed in _make_ocr_request: %s", e)
            return None

    async def _ocr_image(self, image_path: str, request_id: str) -> Optional[str]:
        """Extract text from image using OCR model.
        
        Args:
            image_path: Path to the image file
            request_id: Request ID for logging traceability
            
        Returns:
            Extracted text from the image, or None on failure
        """
        logger.info("[%s] Starting OCR extraction for image: %s", request_id, image_path)
        
        base64_image = self._encode_image_to_base64(image_path)
        if not base64_image:
            logger.error("[%s] Failed to encode image in _ocr_image", request_id)
            return None
        
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Распознай и извлеки весь текст с этого изображения. Верни только распознанный текст без дополнительных комментариев."},
                    {"type": "image_url", "image_url": {"url": base64_image}}
                ]
            }
        ]
        
        try:
            ocr_start_time = datetime.now()
            response = await self._make_ocr_request(messages)
            ocr_end_time = datetime.now()
            ocr_duration = (ocr_end_time - ocr_start_time).total_seconds()
            
            logger.info("[%s] OCR request completed in %.2f seconds", request_id, ocr_duration)
            
            if not response:
                logger.error("[%s] OCR request failed in _ocr_image", request_id)
                return None
            
            extracted_text = response["choices"][0]["message"]["content"]
            logger.info("[%s] OCR extracted text: %s", request_id, extracted_text[:200] if len(extracted_text) > 200 else extracted_text)
            
            return extracted_text
            
        except (KeyError, TypeError) as e:
            logger.error("[%s] Failed to parse OCR response in _ocr_image: %s", request_id, e)
            return None
        except (httpx.RequestError, ValueError) as e:
            logger.error("[%s] Unexpected error in _ocr_image: %s", request_id, e)
            return None

    async def parse_calendar_event(self, text: str, image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        start_time = datetime.now()
        request_id = str(hash(text))[:8]
        logger.info("[%s] Starting Groq processing for: %s", request_id, text)

        # OCR-first pipeline: if image is provided, extract text first
        combined_text = text
        if image_path:
            logger.info("[%s] Image provided in parse_calendar_event: %s, starting OCR extraction", request_id, image_path)
            
            ocr_text = await self._ocr_image(image_path, request_id)
            if ocr_text is None:
                logger.error("[%s] OCR extraction failed in parse_calendar_event", request_id)
                return {
                    "result": False,
                    "comment": "Failed to extract text from image (OCR error)"
                }
            
            # Combine original text with OCR-extracted text
            combined_text = f"{text}\n\nТекст с изображения:\n{ocr_text}"
            logger.info("[%s] Combined text for parsing: %s", request_id, combined_text[:200] if len(combined_text) > 200 else combined_text)

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prompt = f"""
You are a calendar event parser. Extract the following information from the text and return it in valid JSON format.

WARNING! 200 points are deducted for each mistake. You have 600 points left. Be very attentive

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
    * Don't write long phrases: "Звонок с коллегами по поводу уточнения новых требований к ПО" will be cut off by any calendar and there will remain just "Звонок с колл....", and it doesn't allow to understand what the meeting is about. Instead, it would be better to write "Звонок Требования ПО"
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
    - If full date (day and month) is in the past for current year, move to next year
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

        # Text-only parsing (image text already extracted via OCR if provided)
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": combined_text}

        messages = cast(List[Dict[str, Any]], [system_message, user_message])

        try:
            api_start_time = datetime.now()
            response = await self._make_request(messages)
            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()

            logger.info("[%s] Groq API request completed in %.2f seconds", request_id, api_duration)

            if not response:
                logger.error("[%s] Groq API request failed in parse_calendar_event", request_id)
                return {
                    "result": False,
                    "comment": "LLM API error: request timeout or service unavailable"
                }

            content = response["choices"][0]["message"]["content"]
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(content)

            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)

            end_time_dt = datetime.now()
            total_duration = (end_time_dt - start_time).total_seconds()
            logger.info("[%s] Total Groq processing completed in %.2f seconds", request_id, total_duration)

            return result
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("[%s] Failed to parse Groq response in parse_calendar_event: %s", request_id, e)
            return {
                "result": False,
                "comment": f"Failed to parse LLM response: {str(e)}"
            }
        except (httpx.RequestError, ValueError, TypeError) as e:
            logger.error("[%s] Unexpected error in parse_calendar_event: %s", request_id, e)
            return {
                "result": False,
                "comment": f"Unexpected error: {str(e)}"
            }


