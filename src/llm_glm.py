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


class GLMLLM:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings["glm_api_key"]
        # One model for everything: glm-4.6v is multimodal and takes text-only messages too, so the
        # image and the parsing prompt travel in a SINGLE request and there is no OCR step at all
        # (unlike src/llm_groq.py, which needs a separate vision model first). Its text-only sibling
        # glm-4.6 rejects image content with code 1210, so it must not be substituted here.
        self.model = "glm-4.6v"
        self.base_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
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

    async def _make_request(self, messages: List[Dict[str, Any]], temperature: float = 0.7, json_response: bool = True) -> Optional[Dict[str, Any]]:
        """Post one chat completion to the Z.ai endpoint.

        `json_response` asks the model for a JSON object; glm-4.6v honours it and answers without
        markdown fences. It is turned off for process_with_image(), whose callers want free text.
        The timeout is 60 s rather than the 30 s src/llm_groq.py uses for its text call, because
        THIS request is the one carrying the base64 image.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )

                if response.status_code != 200:
                    logger.error("GLM API error %s in _make_request: %s", response.status_code, response.text)
                    # Return a sentinel dict so the caller can distinguish an API-level error
                    # from a network/timeout failure (which returns None)
                    try:
                        error_body = response.json()
                        error_detail = error_body.get("error", {}).get("message", response.text)
                    except Exception:
                        error_detail = response.text
                    return {"_error": True, "_error_detail": error_detail}

                response_json = response.json()
                try:
                    logger.debug(
                        "GLM API response in _make_request: %s",
                        response_json['choices'][0]['message']['content']
                    )
                except (KeyError, TypeError):
                    logger.debug("GLM API response in _make_request: content is missing in choices[0]")
                return response_json

        except httpx.TimeoutException:
            logger.error("GLM API request timeout in _make_request (60s)")
            return None
        except httpx.RequestError as e:
            logger.error("GLM API request failed in _make_request: %s", e)
            return None

    async def process_with_image(self, image_path: str, text: str, temperature: float = 0.7) -> Optional[Dict[str, Any]]:
        """Send a request with both text and image to the LLM API."""
        try:
            request_id = str(hash(text))[:8]
            logger.info("[%s] Starting GLM processing with image: %s", request_id, image_path)

            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("[%s] Failed to encode image in process_with_image", request_id)
                return {
                    "result": False,
                    "comment": "Failed to encode image"
                }

            content: List[Dict[str, Any]] = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]

            messages = cast(List[Dict[str, Any]], [{"role": "user", "content": content}])

            api_start_time = datetime.now()
            response = await self._make_request(messages, temperature, json_response=False)
            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()

            logger.info("[%s] GLM API request with image completed in %.2f seconds", request_id, api_duration)

            if response is None:
                logger.error("[%s] GLM API request with image failed (timeout/network) in process_with_image", request_id)
                return {
                    "result": False,
                    "comment": "LLM API error: request timeout or service unavailable"
                }

            if isinstance(response, dict) and response.get("_error"):
                error_detail = response.get("_error_detail", "unknown API error")
                logger.error("[%s] GLM API returned error in process_with_image: %s", request_id, error_detail)
                return {
                    "result": False,
                    "comment": f"LLM API error: {error_detail}"
                }

            # glm-4.6v also returns `reasoning_content`; the answer is in `content`, sometimes with a
            # leading newline in front of it.
            answer = response["choices"][0]["message"]["content"].strip()

            result: Dict[str, Any] = {"result": True, "content": answer}
            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)

            return result

        except (KeyError, TypeError) as e:
            logger.error("Failed to parse GLM response in process_with_image: %s", e)
            return {
                "result": False,
                "comment": f"Failed to parse LLM response: {str(e)}"
            }
        except (httpx.RequestError, ValueError) as e:
            logger.error("Error processing image request in process_with_image: %s", e)
            return {
                "result": False,
                "comment": f"Error processing image request: {str(e)}"
            }

    async def parse_calendar_event(self, text: str, image_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        start_time = datetime.now()
        request_id = str(hash(text))[:8]
        logger.info("[%s] Starting GLM processing for: %s", request_id, text)

        if image_path:
            logger.info("[%s] Image provided in parse_calendar_event: %s, sending it with the prompt", request_id, image_path)

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
Example response format (end_time defaults to start_time + 1 hour if not specified; description is blank if no specific info; location uses nominative case):
{{
    "title": "название события",
    "start_time": "2024-03-22T15:00:00",
    "end_time": "2024-03-22T16:00:00",
    "description": "описание события",
    "location": "место события",
    "result": true,
    "comment": null
}}

Example of failed parsing (if there is not enough information, e.g. only month without day; comment should explain why parsing failed):
{{
    "result": false,
    "comment": "Недостаточно информации о дате"
}}
"""

        # Single-request pipeline: glm-4.6v reads the image itself, so the picture and the text go
        # to the same model that returns the event JSON. No OCR round trip.
        system_message = {"role": "system", "content": system_prompt}

        if image_path:
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                logger.error("[%s] Failed to encode image in parse_calendar_event", request_id)
                return {
                    "result": False,
                    "comment": "Failed to encode image"
                }
            user_content: Any = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        else:
            user_content = text

        user_message = {"role": "user", "content": user_content}

        messages = cast(List[Dict[str, Any]], [system_message, user_message])

        try:
            api_start_time = datetime.now()
            response = await self._make_request(messages)
            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()

            logger.info("[%s] GLM API request completed in %.2f seconds", request_id, api_duration)

            if response is None:
                # Network-level failure: timeout or connection error
                logger.error("[%s] GLM API request failed (timeout/network) in parse_calendar_event", request_id)
                return {
                    "result": False,
                    "comment": "LLM API error: request timeout or service unavailable"
                }

            if isinstance(response, dict) and response.get("_error"):
                # API returned a non-200 HTTP status; surface the actual error detail
                error_detail = response.get("_error_detail", "unknown API error")
                logger.error("[%s] GLM API returned error in parse_calendar_event: %s", request_id, error_detail)
                return {
                    "result": False,
                    "comment": f"LLM API error: {error_detail}"
                }

            # The model also returns `reasoning_content`; the answer lives in `content` as usual, and
            # it sometimes starts with a newline — strip it before the JSON parse.
            content = response["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
            result = json.loads(content)

            if "usage" in response:
                result["tokens_used"] = response["usage"].get("total_tokens", 0)

            end_time_dt = datetime.now()
            total_duration = (end_time_dt - start_time).total_seconds()
            logger.info("[%s] Total GLM processing completed in %.2f seconds", request_id, total_duration)

            return result
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("[%s] Failed to parse GLM response in parse_calendar_event: %s", request_id, e)
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
