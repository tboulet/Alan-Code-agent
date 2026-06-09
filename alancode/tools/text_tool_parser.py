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
- ``format_error()`` → return error feedback for the model
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
    def detect_malformed(self, text: str) -> bool:
        """Detect attempted but malformed tool calls.

        Returns True if the text contains a tool-call-like block that did
        not parse cleanly, False otherwise.
        """
        ...

    @abstractmethod
    def format_error(self) -> str:
        """Return an error message to feed back to the model.

        Includes the expected format and an example. The model's own
        output is intentionally NOT echoed back — doing so confuses the
        model about what is its message and what is the tool feedback.
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

# Loose pattern: requires BOTH opening and closing tags. A bare <tool_call>
# mentioned in prose (e.g. when the model apologizes for a previous error and
# quotes the tag) must NOT trigger malformed detection — that caused a
# self-perpetuating retry loop where each error message got quoted back.
_HERMES_LOOSE_PATTERN = re.compile(
    r"<tool_call>.*?</tool_call>",
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

    def detect_malformed(self, text: str) -> bool:
        for match in _HERMES_LOOSE_PATTERN.finditer(text):
            if not _HERMES_PATTERN.match(match.group(0)):
                return True
        return False

    def format_error(self) -> str:
        return (
            "Found <tool_call> block but content is not valid.\n\n"
            "Expected format:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
            "</tool_call>\n\n"
            "Example:\n"
            "<tool_call>\n"
            '{"name": "Read", "arguments": {"file_path": "/path/to/file.py"}}\n'
            "</tool_call>\n\n"
            "Please retry with the correct format."
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

# Loose pattern: requires BOTH opening and closing tags so that a bare
# <tool_call> mentioned in prose does not trigger malformed detection.
_GLM_LOOSE_PATTERN = re.compile(
    r"<tool_call>.*?</tool_call>",
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

    def detect_malformed(self, text: str) -> bool:
        strict_matches = {m.group(0) for m in _GLM_PATTERN.finditer(text)}
        for match in _GLM_LOOSE_PATTERN.finditer(text):
            if match.group(0) not in strict_matches:
                return True
        return False

    def format_error(self) -> str:
        return (
            "Found <tool_call> block but format is incorrect.\n\n"
            "Expected format:\n"
            "<tool_call>ToolName"
            "<arg_key>parameter_name</arg_key>"
            "<arg_value>parameter_value</arg_value>"
            "</tool_call>\n\n"
            "Example:\n"
            "<tool_call>Bash"
            "<arg_key>command</arg_key>"
            "<arg_value>ls -la</arg_value>"
            "</tool_call>\n\n"
            "Please retry with the correct format."
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

# Loose pattern: requires BOTH opening and closing tags so that a bare
# <tool_use> mentioned in prose does not trigger malformed detection.
_ALAN_LOOSE_PATTERN = re.compile(
    r"<tool_use>.*?</tool_use>",
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

    def detect_malformed(self, text: str) -> bool:
        for match in _ALAN_LOOSE_PATTERN.finditer(text):
            if not _ALAN_PATTERN.match(match.group(0)):
                return True
        return False

    def format_error(self) -> str:
        return (
            "Found <tool_use> block but content is not valid.\n\n"
            "Expected format:\n"
            "<tool_use>\n"
            '{"name": "tool_name", "input": {"param": "value"}}\n'
            "</tool_use>\n\n"
            "Example:\n"
            "<tool_use>\n"
            '{"name": "Read", "input": {"file_path": "/path/to/file.py"}}\n'
            "</tool_use>\n\n"
            "Please retry with the correct format."
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


# ── Format: hermes_xml ────────────────────────────────────────────────────────
#
# Qwen3-Coder-Next / Qwen3-Next-family (and other Hermes-FunctionCalling-Lite
# trained models) emit `<tool_call><function=NAME><parameter=KEY>VAL</parameter>
# </function></tool_call>` instead of the JSON-inside-tag the plain `hermes`
# format expects. The two are visually similar — same outer `<tool_call>` tag
# — but the body is XML-shaped not JSON-shaped.


_HERMES_XML_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)

_HERMES_XML_ARG_PATTERN = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL,
)

# Loose: only fires malformed-detection when both <tool_call> and </tool_call>
# are present (so prose mentions of "<tool_call>" don't trigger).
_HERMES_XML_LOOSE_PATTERN = re.compile(
    r"<tool_call>.*?</tool_call>",
    re.DOTALL,
)


def _coerce_arg(raw: str) -> object:
    """Try JSON-decode (numbers, bools, lists, objects) — else strip + return str."""
    s = raw.strip()
    if not s:
        return ""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


class HermesXMLFormat(ToolCallFormat):
    """Hermes-XML format: ``<tool_call><function=N><parameter=K>V</parameter></function></tool_call>``

    Used by Qwen3-Coder-Next and other Hermes-FunctionCalling-Lite trained
    models — the body of the <tool_call> tag is XML-shaped, NOT JSON.
    """

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _HERMES_XML_PATTERN.finditer(text):
            name = match.group(1).strip()
            body = match.group(2)
            args: dict[str, object] = {}
            for arg_match in _HERMES_XML_ARG_PATTERN.finditer(body):
                k = arg_match.group(1).strip()
                v = arg_match.group(2)
                args[k] = _coerce_arg(v)
            if name:
                results.append(ParsedToolCall(
                    name=name, input=args, raw_match=match.group(0),
                ))
        return results

    def detect_malformed(self, text: str) -> bool:
        # A <tool_call> block that doesn't satisfy the strict pattern AND
        # isn't a valid `hermes` JSON-body either is malformed.
        for match in _HERMES_XML_LOOSE_PATTERN.finditer(text):
            blk = match.group(0)
            if _HERMES_XML_PATTERN.match(blk):
                continue
            # Maybe it's JSON-body hermes-style — that's the sibling format's
            # problem, not ours. Don't double-report.
            if re.match(r"<tool_call>\s*\{.*?\}\s*</tool_call>", blk, re.DOTALL):
                continue
            return True
        return False

    def format_error(self) -> str:
        return (
            "Found <tool_call> block but content is not valid Hermes-XML.\n\n"
            "Expected format:\n"
            "<tool_call>\n"
            "<function=tool_name>\n"
            "<parameter=param>value</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "Example:\n"
            "<tool_call>\n"
            "<function=Read>\n"
            "<parameter=file_path>/path/to/file.py</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, output one or more <tool_call> blocks. The body of "
            "each block uses <function=NAME> and <parameter=KEY>VALUE</parameter> "
            "(NOT JSON):\n"
            "<tool_call>\n"
            "<function=tool_name>\n"
            "<parameter=param>value</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "You may call multiple tools by outputting multiple <tool_call> blocks.\n"
            "After a tool call, wait for the result before continuing."
        )


# ── Format: meta_json ─────────────────────────────────────────────────────────
#
# Llama-3.1+, Llama-3.3, and Meta tool-calling models emit a bare JSON object:
#   {"type": "function", "name": "Read", "parameters": {"file_path": "..."}}
# without any wrapping tag. The model relies on its chat template to wrap with
# <|python_tag|>...<|eom_id|> tokens; when those aren't injected (default for
# SGLang served as plain openai/* via LiteLLM), the JSON leaks into the
# response content and nothing parses it.


# Match a top-level {...} that contains "type": "function" and "name".
# Non-greedy + balanced-brace matching is not in the stdlib regex engine, so
# we capture the broadest plausible bracketed chunk and JSON-decode it.
_META_JSON_PATTERN = re.compile(
    r'\{\s*"type"\s*:\s*"function"\s*,.*?\}\s*\}',
    re.DOTALL,
)


def _scan_meta_json_objects(text: str) -> list[tuple[str, dict]]:
    """Find all top-level JSON objects in *text* that have type=function.

    Uses a brace-counting scan because regex alone can't match balanced braces.
    Returns a list of (raw_match, decoded_dict).
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        # Find the next opening brace.
        start = text.find("{", i)
        if start == -1:
            break
        # Walk forward, balancing braces, respecting strings.
        depth = 0
        j = start
        in_str = False
        esc = False
        while j < n:
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\" and in_str:
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : j + 1]
                        try:
                            obj = json.loads(chunk)
                        except (json.JSONDecodeError, ValueError):
                            obj = None
                        if isinstance(obj, dict) and obj.get("type") == "function" and obj.get("name"):
                            out.append((chunk, obj))
                        i = j + 1
                        break
            j += 1
        else:
            # No balanced close found; bail.
            break
    return out


class MetaJSONFormat(ToolCallFormat):
    """Meta-Llama raw-JSON format: ``{"type":"function","name":N,"parameters":{...}}``.

    Used by Llama-3.1+, Llama-3.3 and other Meta-family tool-calling models when
    their chat template's <|python_tag|>...<|eom_id|> wrappers are not injected
    (typical for SGLang served as plain openai/* via LiteLLM).
    """

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for raw, obj in _scan_meta_json_objects(text):
            name = obj.get("name") or ""
            params = obj.get("parameters", obj.get("arguments", {}))
            if not isinstance(params, dict):
                params = {}
            if name:
                results.append(ParsedToolCall(name=name, input=params, raw_match=raw))
        return results

    def detect_malformed(self, text: str) -> bool:
        # If text mentions {"type": "function" but no valid object decodes,
        # treat as malformed so we can return a corrective error to the model.
        if '"type"' in text and '"function"' in text:
            return not bool(_scan_meta_json_objects(text))
        return False

    def format_error(self) -> str:
        return (
            'Found a "type":"function" hint but no valid JSON tool call.\n\n'
            "Expected format (one JSON object per call, no wrapping tags):\n"
            '{"type": "function", "name": "tool_name", '
            '"parameters": {"param": "value"}}\n\n'
            "Example:\n"
            '{"type": "function", "name": "Read", '
            '"parameters": {"file_path": "/path/to/file.py"}}\n\n'
            "Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, output ONE JSON object on its own (no wrapping "
            "tags, no prose around it on the same line):\n"
            '{"type": "function", "name": "tool_name", '
            '"parameters": {"param": "value"}}\n\n'
            "You may call multiple tools by outputting multiple JSON objects, "
            "one per line. After a tool call, wait for the result before "
            "continuing."
        )


# ── Registry ─────────────────────────────────────────────────────────────────


FORMATS: dict[str, ToolCallFormat] = {
    "hermes": HermesFormat(),
    "hermes_xml": HermesXMLFormat(),
    "glm": GLMFormat(),
    "alan": AlanFormat(),
    "meta_json": MetaJSONFormat(),
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
    if fmt.detect_malformed(text):
        thinking, cleaned = _extract_thinking(text)
        return ParseResult(tool_calls=[], cleaned_text=cleaned, thinking=thinking, error=fmt.format_error())

    # No tool call attempt at all — normal text response
    thinking, cleaned = _extract_thinking(text)
    return ParseResult(tool_calls=[], cleaned_text=cleaned, thinking=thinking)


def get_tool_format_system_prompt(format: str, tool_schemas: list[dict]) -> str:
    """Generate system prompt instructions for text-based tool calling."""
    return get_format(format).system_prompt(tool_schemas)
