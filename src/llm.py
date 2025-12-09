#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any
from datetime import datetime  # kept for tests that patch `src.llm.datetime`

from loguru import logger

from .config import get_settings
from .llm_base import LLMProvider
from .llm_deepseek import DeepSeekLLM
from .llm_groq import GroqLLM


_provider: Optional[LLMProvider] = None


def _create_provider_from_settings() -> LLMProvider:
    """
    Create an LLM provider instance based on configuration.

    Supported providers (via settings["llm_provider"] or LLM_PROVIDER env):

    - "deepseek" (default): DeepSeekLLM from src.llm_deepseek
    - "groq": GroqLLM from src.llm_groq
    """
    settings = get_settings()
    provider_name = settings.get("llm_provider", "deepseek").lower()

    if provider_name == "groq":
        logger.info("Initializing LLM provider: groq")
        return GroqLLM()

    # Fallback/default
    if provider_name != "deepseek":
        logger.warning(
            "Unknown LLM provider '%s', falling back to 'deepseek'",
            provider_name,
        )
    logger.info("Initializing LLM provider: deepseek")
    return DeepSeekLLM()


def get_llm() -> LLMProvider:
    """
    Return a singleton LLM provider instance.

    This is the main entry point for the rest of the application.
    """
    global _provider
    if _provider is None:
        _provider = _create_provider_from_settings()
    return _provider


async def parse_calendar_event(
    text: str,
    image_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Thin facade that delegates to the configured provider.

    Kept as a module-level function so existing imports like
    `from src.llm import parse_calendar_event` continue to work.
    """
    llm = get_llm()
    return await llm.parse_calendar_event(text, image_path)


async def process_with_image(
    image_path: str,
    text: str,
    temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """
    Thin facade that delegates image+text processing to the provider.
    """
    llm = get_llm()
    return await llm.process_with_image(image_path, text, temperature)
