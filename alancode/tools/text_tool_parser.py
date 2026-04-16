"""Text-based tool call parser for models without native tool calling.

When a model doesn't support the OpenAI tool_calls response format, it may
still output tool calls as text in its own format. This module extracts
those tool calls from the text and converts them to ToolUseBlock objects.

Supported formats:
- ``hermes``: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
- ``glm``: ``<tool_call>Name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
- ``alan``: ``<tool_use>{"name": "...", "input": {...}}</tool_use>``

Each format is implemented as a ToolCallFormat class with:
- ``parse(text)`` → extract well-formed tool calls
- ``detect_malformed(text)`` → detect attempted but incorrectly formatted tool calls
- ``format_error(text)`` → return error feedback for the model
- ``system_prompt(tool_schemas)`` → return format instructions for the system prompt
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_TEXT_TOOL_RETRIES = 3


@dataclass
class ParsedToolCall:
    """A tool call extracted from text."""
    name: str
    input: dict[str, Any]
    raw_match: str  # The full matched text (for removal from content)


@dataclass
class ParseResult:
    """Result of parsing text for tool calls.

    Attributes:
        tool_calls: Successfully parsed tool calls.
        cleaned_text: Text with tool call markup and thinking tags removed.
        thinking: Extracted thinking content (from ``<think>`` tags), or None.
        error: If non-None, the model attempted a tool call but the format
            was wrong. This message should be fed back to the model.
    """
    tool_calls: list[ParsedToolCall]
    cleaned_text: str
    thinking: str | None = None
    error: str | None = None


# ── Base class ───────────────────────────────────────────────────────────────


class ToolCallFormat(ABC):
    """Base class for text-based tool call format parsers."""

    @abstractmethod
    def parse(self, text: str) -> list[ParsedToolCall]:
        """Extract well-formed tool calls from text."""
        ...

    @abstractmethod
    def detect_malformed(self, text: str) -> str | None:
        """Detect attempted but malformed tool calls.

        Returns a description of what went wrong, or None if no
        malformed attempt was detected.
        """
        ...

    @abstractmethod
    def format_error(self, malformed_description: str) -> str:
        """Return an error message to feed back to the model.

        Includes the expected format and what was wrong.
        """
        ...

    @abstractmethod
    def system_prompt(self, tool_schemas: list[dict]) -> str:
        """Return system prompt instructions for this format."""
        ...


# ── Format: hermes ───────────────────────────────────────────────────────────


_HERMES_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Loose patterns to detect attempted but malformed tool calls
_HERMES_LOOSE_PATTERN = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_call>[^<]{10,}",
    re.DOTALL,
)


class HermesFormat(ToolCallFormat):
    """Hermes/Qwen format: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``"""

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _HERMES_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                arguments = data.get("arguments", data.get("input", {}))
                if name:
                    results.append(ParsedToolCall(
                        name=name, input=arguments, raw_match=match.group(0),
                    ))
            except json.JSONDecodeError:
                pass  # Detected as malformed below
        return results

    def detect_malformed(self, text: str) -> str | None:
        # Check for <tool_call> tags that didn't parse as valid JSON
        for match in _HERMES_LOOSE_PATTERN.finditer(text):
            snippet = match.group(0)
            if not _HERMES_PATTERN.match(snippet):
                return f"Found <tool_call> block but content is not valid JSON: {snippet[:150]}"
        return None

    def format_error(self, malformed_description: str) -> str:
        return (
            f"Tool call format error: {malformed_description}\n\n"
            f"Expected format:\n"
            f"<tool_call>\n"
            f'{{"name": "tool_name", "arguments": {{"param": "value"}}}}\n'
            f"</tool_call>\n\n"
            f"Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, output a JSON object inside <tool_call> tags:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "You may call multiple tools by outputting multiple <tool_call> blocks.\n"
            "After a tool call, wait for the result before continuing."
        )


# ── Format: glm ──────────────────────────────────────────────────────────────


_GLM_PATTERN = re.compile(
    # Closing </tool_call> is REQUIRED. A partial mid-stream match without
    # the closing tag used to parse as a complete tool call and execute
    # with truncated arguments.
    r"<tool_call>(\w+)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)+)</tool_call>",
    re.DOTALL,
)

_GLM_ARG_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>",
    re.DOTALL,
)

# Loose patterns: any <tool_call> that didn't match the strict pattern
_GLM_LOOSE_PATTERN = re.compile(
    r"<tool_call>.*?(?:</tool_call>|$)",
    re.DOTALL,
)


class GLMFormat(ToolCallFormat):
    """GLM format: ``<tool_call>Name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``"""

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _GLM_PATTERN.finditer(text):
            name = match.group(1)
            args_text = match.group(2)
            args = {}
            for arg_match in _GLM_ARG_PATTERN.finditer(args_text):
                key = arg_match.group(1).strip()
                value = arg_match.group(2).strip()
                args[key] = value
            if name:
                results.append(ParsedToolCall(
                    name=name, input=args, raw_match=match.group(0),
                ))
        return results

    def detect_malformed(self, text: str) -> str | None:
        # Find any <tool_call> that the strict parser didn't match
        strict_matches = {m.group(0) for m in _GLM_PATTERN.finditer(text)}
        for match in _GLM_LOOSE_PATTERN.finditer(text):
            snippet = match.group(0)
            if snippet not in strict_matches:
                return f"Found <tool_call> block but format is incorrect: {snippet[:150]}"
        return None

    def format_error(self, malformed_description: str) -> str:
        return (
            f"Tool call format error: {malformed_description}\n\n"
            f"Expected format:\n"
            f"<tool_call>ToolName"
            f"<arg_key>parameter_name</arg_key>"
            f"<arg_value>parameter_value</arg_value>"
            f"</tool_call>\n\n"
            f"Example:\n"
            f"<tool_call>Bash"
            f"<arg_key>command</arg_key>"
            f"<arg_value>ls -la</arg_value>"
            f"</tool_call>\n\n"
            f"Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_desc = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in tool_schemas
        )
        return (
            "\n\n# Available Tools\n\n"
            f"{tools_desc}\n\n"
            "Use <tool_call> tags to call tools with this exact format:\n"
            "<tool_call>ToolName"
            "<arg_key>param</arg_key>"
            "<arg_value>value</arg_value>"
            "</tool_call>"
        )


# ── Format: alan ─────────────────────────────────────────────────────────────


_ALAN_PATTERN = re.compile(
    r"<tool_use>\s*(\{.*?\})\s*</tool_use>",
    re.DOTALL,
)

_ALAN_LOOSE_PATTERN = re.compile(
    r"<tool_use>.*?</tool_use>|<tool_use>[^<]{10,}",
    re.DOTALL,
)


class AlanFormat(ToolCallFormat):
    """Alan format: ``<tool_use>{"name": ..., "input": ...}</tool_use>``"""

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _ALAN_PATTERN.finditer(text):
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                input_data = data.get("input", data.get("arguments", {}))
                if name:
                    results.append(ParsedToolCall(
                        name=name, input=input_data, raw_match=match.group(0),
                    ))
            except json.JSONDecodeError:
                pass
        return results

    def detect_malformed(self, text: str) -> str | None:
        for match in _ALAN_LOOSE_PATTERN.finditer(text):
            snippet = match.group(0)
            if not _ALAN_PATTERN.match(snippet):
                return f"Found <tool_use> block but content is not valid JSON: {snippet[:150]}"
        return None

    def format_error(self, malformed_description: str) -> str:
        return (
            f"Tool call format error: {malformed_description}\n\n"
            f"Expected format:\n"
            f"<tool_use>\n"
            f'{{"name": "tool_name", "input": {{"param": "value"}}}}\n'
            f"</tool_use>\n\n"
            f"Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, output a JSON object inside <tool_use> tags:\n"
            "<tool_use>\n"
            '{"name": "tool_name", "input": {"param": "value"}}\n'
            "</tool_use>\n\n"
            "You may call multiple tools by outputting multiple <tool_use> blocks.\n"
            "After a tool call, wait for the result before continuing."
        )


# ── Registry ─────────────────────────────────────────────────────────────────


FORMATS: dict[str, ToolCallFormat] = {
    "hermes": HermesFormat(),
    "glm": GLMFormat(),
    "alan": AlanFormat(),
}


def get_format(name: str) -> ToolCallFormat:
    """Get a ToolCallFormat by name.

    Raises:
        ValueError: If the format name is not recognized.
    """
    fmt = FORMATS.get(name)
    if fmt is None:
        raise ValueError(f"Unknown tool call format: {name!r}. Supported: {list(FORMATS.keys())}")
    return fmt


# ── Public API ───────────────────────────────────────────────────────────────


def _extract_thinking(text: str) -> tuple[str | None, str]:
    """Extract thinking content from ``<think>...</think>`` tags.

    Returns (thinking_text, remaining_text).
    If no thinking tags found, returns (None, original_text).
    """
    # Handle both <think>...</think> and just </think> (opening tag sometimes missing)
    import re
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = text[:match.start()] + text[match.end():]
        return (thinking or None, remaining.strip())

    # Handle </think> without opening tag (model sometimes just closes)
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].strip()
        remaining = parts[1].strip() if len(parts) > 1 else ""
        return (thinking or None, remaining)

    return (None, text.strip())


def extract_tool_calls_from_text(
    text: str,
    format: str = "hermes",
) -> ParseResult:
    """Extract tool calls from model text output.

    Returns a ParseResult with:
    - ``tool_calls``: successfully parsed tool calls
    - ``cleaned_text``: text with markup removed
    - ``error``: if non-None, the model attempted a tool call but used
      the wrong format. This message should be sent back as a tool
      result error so the model can retry.
    """
    fmt = get_format(format)

    # Try strict parsing first
    tool_calls = fmt.parse(text)
    cleaned = text
    for tc in tool_calls:
        cleaned = cleaned.replace(tc.raw_match, "")
    thinking, cleaned = _extract_thinking(cleaned)

    if tool_calls:
        return ParseResult(tool_calls=tool_calls, cleaned_text=cleaned, thinking=thinking)

    # No valid tool calls — check for malformed attempts
    malformed = fmt.detect_malformed(text)
    if malformed:
        error_msg = fmt.format_error(malformed)
        thinking, cleaned = _extract_thinking(text)
        return ParseResult(tool_calls=[], cleaned_text=cleaned, thinking=thinking, error=error_msg)

    # No tool call attempt at all — normal text response
    thinking, cleaned = _extract_thinking(text)
    return ParseResult(tool_calls=[], cleaned_text=cleaned, thinking=thinking)


def get_tool_format_system_prompt(format: str, tool_schemas: list[dict]) -> str:
    """Generate system prompt instructions for text-based tool calling."""
    return get_format(format).system_prompt(tool_schemas)
