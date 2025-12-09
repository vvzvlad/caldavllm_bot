#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Dict, Any, Optional


class LLMProvider(Protocol):
    async def parse_calendar_event(
        self,
        text: str,
        image_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    async def process_with_image(
        self,
        image_path: str,
        text: str,
        temperature: float = 0.7,
    ) -> Optional[Dict[str, Any]]:
        ...