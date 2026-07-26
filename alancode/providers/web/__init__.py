"""Web backend - drive online AI assistant chat UIs as LLM providers.

Instead of an API, requests go through a real browser window automated at
the X11 level (xdotool + xclip): prompts are pasted into the assistant's
composer, and answers are read back by copying the page text and extracting
tagged blocks. Site-specific details live in ``WebAssistant`` entries;
``WebProvider`` implements the ``LLMProvider`` contract on top.
"""

from alancode.providers.web.assistant import ASSISTANTS, WebAssistant
from alancode.providers.web.provider import WebProvider

__all__ = ["ASSISTANTS", "WebAssistant", "WebProvider"]
