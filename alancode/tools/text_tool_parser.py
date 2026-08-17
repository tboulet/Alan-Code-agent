"""Text-based tool call parser for models without native tool calling.

When a model doesn't support the OpenAI tool_calls response format, it may
still output tool calls as text in its own format. This module extracts
those tool calls from the text and converts them to ToolUseBlock objects.

Supported formats:
- ``hermes``: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
- ``hermes_xml``: ``<tool_call><function=N><parameter=K>V</parameter></function></tool_call>``
- ``glm``: ``<tool_call>Name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
- ``alan``: ``<tool_use>{"name": "...", "input": {...}}</tool_use>``
- ``meta_json``: ``{"type": "function", "name": "...", "parameters": {...}}``
- ``bash_block``: one fenced ```` ```bash ```` code block, run as the Bash tool
- ``kimi``: ``<|tool_call_begin|>id<|tool_call_argument_begin|>{...}<|tool_call_end|>``
- ``deepseek``: DSML ``invoke``/``parameter`` markup (fullwidth-bar delimited)
- ``minimax``: ``<minimax:tool_call>`` envelope around plain ``invoke``/``parameter``
- ``auto``: accept ANY of the above, whichever strict-parses (teaches bash_block)

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

    # Whether reasoning content may carry this format's tool calls (some
    # OpenAI-compatible reasoning models emit their protocol there).
    parse_thinking: bool = True

    # API stop strings that end generation the moment a call is complete
    # (one tool call per turn, no post-call rambling). Servers cut the
    # output BEFORE the stop string, so a stop-terminated call arrives
    # without its closing marker - repair_stop_truncation restores it.
    # The loop only repairs when stop_reason is not "max_tokens": a
    # length-truncated call must stay unparseable.
    stop_sequences: tuple[str, ...] = ()

    def repair_stop_truncation(self, text: str) -> str:
        """Re-append the closing marker a stop sequence stripped."""
        return text

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


def _close_unbalanced_tag(text: str, open_tag: str, close_tag: str) -> str:
    """Append *close_tag* when an *open_tag* is left unclosed."""
    if text.count(open_tag) > text.count(close_tag):
        return text + close_tag
    return text


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

    stop_sequences = ("</tool_call>",)

    def repair_stop_truncation(self, text: str) -> str:
        return _close_unbalanced_tag(text, "<tool_call>", "</tool_call>")

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

    stop_sequences = ("</tool_call>",)

    def repair_stop_truncation(self, text: str) -> str:
        return _close_unbalanced_tag(text, "<tool_call>", "</tool_call>")

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

    stop_sequences = ("</tool_use>",)

    def repair_stop_truncation(self, text: str) -> str:
        return _close_unbalanced_tag(text, "<tool_use>", "</tool_use>")

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


# The trailing </tool_call> is optional: models that stop generation right
# after </function> (observed on Qwen2.5-72B) must still parse.
_HERMES_XML_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)\s*>(.*?)</function>(?:\s*</tool_call>)?",
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

    # The parser already tolerates a missing </tool_call> (stop-stripped
    # or model-omitted), so no repair is needed.
    stop_sequences = ("</tool_call>",)

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
            "Parameter values are RAW text: never JSON-encode or escape them. "
            "Multi-line values (scripts, file contents, heredocs) go between the "
            "parameter tags verbatim:\n"
            "<tool_call>\n"
            "<function=Bash>\n"
            "<parameter=command>\n"
            "cat > hello.py <<'EOF'\n"
            "print(\"hello\")\n"
            "EOF\n"
            "python3 hello.py\n"
            "</parameter>\n"
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


# ── Format: bash_block ───────────────────────────────────────────────────────
#
# Mini-SWE-Agent convention: the model writes free-form reasoning
# followed by ONE fenced ```bash code block, whose content is the Bash tool's
# command. No structured markup. Only the first block of an answer runs; the
# teaching prompt says "exactly one block", so extras are model confusion and
# are ignored. The opening fence may be glued to prose (GLM-5.2 emits
# "prose.```bash\n...") but must be followed by a newline - that is what
# keeps a prose MENTION of "```bash" from matching; the closing fence must
# sit at line start. An unclosed block (stream still arriving, or output
# truncated) is NOT a call - the length-truncation recovery handles the
# truncated case.


# Fence lines tolerate a trailing \r: GLM-5.2 intermittently emits CRLF
# line endings, which otherwise kill both anchors on a well-formed block.
_BASH_BLOCK_PATTERN = re.compile(
    r"```bash[ \t]*\r?\n(.*?)\n```[ \t\r]*$",
    re.DOTALL | re.MULTILINE,
)


class BashBlockFormat(ToolCallFormat):
    """Bash-block format: one fenced ```bash block, run as the Bash tool."""

    # Fenced code in reasoning is a draft, not an action: the convention
    # puts the executable block in the visible answer only.
    parse_thinking = False

    # "\n```\n" only matches a CLOSING fence line - the opening fence is
    # "```bash" so the newline right after the backticks cannot match it.
    stop_sequences = ("\n```\n",)

    def repair_stop_truncation(self, text: str) -> str:
        if _BASH_BLOCK_PATTERN.search(text):
            return text
        if re.search(r"```bash[ \t]*\r?\n", text):
            return text.rstrip("\r\n") + "\n```"
        return text

    def parse(self, text: str) -> list[ParsedToolCall]:
        matches = list(_BASH_BLOCK_PATTERN.finditer(text))
        if not matches:
            return []
        if len(matches) > 1:
            logger.debug(
                "bash_block: ignoring %d extra fenced block(s), running the "
                "first only", len(matches) - 1,
            )
        first = matches[0]
        return [ParsedToolCall(
            name="Bash",
            input={"command": first.group(1)},
            raw_match=first.group(0),
        )]

    def detect_malformed(self, text: str) -> bool:
        # No-block and unclosed-block answers are normal turns (reasoning
        # only, or truncated output), never a malformed call to retry.
        return False

    def format_error(self) -> str:
        return (
            "No valid bash block found.\n\n"
            "Write exactly one fenced bash code block; its content is "
            "executed as a shell command:\n"
            "```bash\n"
            "ls -la\n"
            "```"
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        return (
            "\n\n# Tool Calling\n\n"
            "You act by writing shell commands. After your reasoning, "
            "include exactly ONE fenced bash code block; its content is "
            "executed in the terminal and the output is returned to you:\n"
            "```bash\n"
            "ls -la\n"
            "```\n\n"
            "Multi-line commands, heredocs and && chains all go inside the "
            "single block. Only the first block of an answer is executed.\n"
            "After the block, wait for the result before continuing."
        )


# ── Format: kimi ─────────────────────────────────────────────────────────────
#
# Kimi K2-family special-token format, emitted by Kimi-K2.7-Code regardless
# of the taught convention (195/196 turns observed on the MiniGrid bench):
# <|tool_calls_section_begin|><|tool_call_begin|>functions.Name:0
# <|tool_call_argument_begin|>{"param": ...}<|tool_call_end|>
# <|tool_calls_section_end|>. Tool-id shapes vary (functions.Name:idx,
# Name:idx, bare Name); the name is extracted tolerantly. Built from the
# K2 spec plus bench transcripts - refine against raw samples if a
# variant shows up.


_KIMI_CALL_PATTERN = re.compile(
    r"(?:<\|tool_calls_section_begin\|>\s*)?"
    r"<\|tool_call_begin\|>\s*(.*?)\s*"
    r"<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>"
    r"(?:\s*<\|tool_calls_section_end\|>)?",
    re.DOTALL,
)

_KIMI_NAME_PATTERN = re.compile(r"([A-Za-z_][\w-]*)(?::\d+)?\s*$")


def _kimi_tool_name(raw_id: str) -> str:
    match = _KIMI_NAME_PATTERN.search(raw_id.split(".")[-1])
    return match.group(1) if match else raw_id.strip()


class KimiFormat(ToolCallFormat):
    """Kimi special-token format: ``<|tool_call_begin|>id<|tool_call_argument_begin|>{...}<|tool_call_end|>``"""

    stop_sequences = ("<|tool_call_end|>",)

    def repair_stop_truncation(self, text: str) -> str:
        return _close_unbalanced_tag(
            text, "<|tool_call_begin|>", "<|tool_call_end|>",
        )

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _KIMI_CALL_PATTERN.finditer(text):
            name = _kimi_tool_name(match.group(1))
            try:
                args = json.loads(match.group(2))
            except (json.JSONDecodeError, ValueError):
                continue  # Detected as malformed below
            if not isinstance(args, dict):
                continue
            if name:
                results.append(ParsedToolCall(
                    name=name, input=args, raw_match=match.group(0),
                ))
        return results

    def detect_malformed(self, text: str) -> bool:
        if "<|tool_call_begin|>" not in text:
            return False
        return not self.parse(text)

    def format_error(self) -> str:
        return (
            "Found Kimi tool-call tokens but the call did not parse.\n\n"
            "Expected format (arguments must be a valid JSON object):\n"
            "<|tool_call_begin|>functions.tool_name:0"
            '<|tool_call_argument_begin|>{"param": "value"}<|tool_call_end|>\n\n'
            "Example:\n"
            "<|tool_call_begin|>functions.Bash:0"
            '<|tool_call_argument_begin|>{"command": "ls -la"}<|tool_call_end|>\n\n'
            "Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, use your tool-call tokens with a JSON object "
            "as the argument payload:\n"
            "<|tool_call_begin|>functions.tool_name:0"
            '<|tool_call_argument_begin|>{"param": "value"}<|tool_call_end|>\n\n'
            "After a tool call, wait for the result before continuing."
        )


# ── Format: deepseek ─────────────────────────────────────────────────────────
#
# DeepSeek DSML markup, emitted by DeepSeek-V4-Flash regardless of the
# taught convention (21/22 turns observed on the MiniGrid bench). The
# delimiter bar is FULLWIDTH VERTICAL LINE U+FF5C, not the ASCII pipe:
# <(bar)DSML(bar)tool_calls> / <(bar)DSML(bar)invoke name="Bash"> /
# <(bar)DSML(bar)parameter name="command" string="true">VALUE</...>.
# Values are raw element text (multi-line, heredocs) unless the
# string="true" attribute is absent, in which case they are JSON-coerced.


_DS_BAR = "｜"  # FULLWIDTH VERTICAL LINE, the actual delimiter DeepSeek emits
_DS_OPEN = f"<{_DS_BAR}DSML{_DS_BAR}"
_DS_CLOSE = f"</{_DS_BAR}DSML{_DS_BAR}"

_DEEPSEEK_INVOKE_PATTERN = re.compile(
    rf"(?:{re.escape(_DS_OPEN)}tool_calls>\s*)?"
    rf'{re.escape(_DS_OPEN)}invoke name="([^"]+)"\s*>(.*?)'
    rf"{re.escape(_DS_CLOSE)}invoke>"
    rf"(?:\s*{re.escape(_DS_CLOSE)}tool_calls>)?",
    re.DOTALL,
)

_DEEPSEEK_PARAM_PATTERN = re.compile(
    rf'{re.escape(_DS_OPEN)}parameter name="([^"]+)"([^>]*)>(.*?)'
    rf"{re.escape(_DS_CLOSE)}parameter>",
    re.DOTALL,
)


class DeepSeekFormat(ToolCallFormat):
    """DeepSeek DSML format: fullwidth-bar ``invoke``/``parameter`` markup."""

    stop_sequences = (f"{_DS_CLOSE}tool_calls>",)

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _DEEPSEEK_INVOKE_PATTERN.finditer(text):
            name = match.group(1).strip()
            args: dict[str, object] = {}
            for param in _DEEPSEEK_PARAM_PATTERN.finditer(match.group(2)):
                key = param.group(1).strip()
                attrs = param.group(2)
                value = param.group(3)
                args[key] = value if 'string="true"' in attrs else _coerce_arg(value)
            if name:
                results.append(ParsedToolCall(
                    name=name, input=args, raw_match=match.group(0),
                ))
        return results

    def detect_malformed(self, text: str) -> bool:
        if _DS_OPEN not in text:
            return False
        return not self.parse(text)

    def format_error(self) -> str:
        return (
            "Found DSML tool-call markup but the call did not parse.\n\n"
            "Expected format:\n"
            f"{_DS_OPEN}tool_calls>\n"
            f'{_DS_OPEN}invoke name="tool_name">\n'
            f'{_DS_OPEN}parameter name="param" string="true">value'
            f"{_DS_CLOSE}parameter>\n"
            f"{_DS_CLOSE}invoke>\n"
            f"{_DS_CLOSE}tool_calls>\n\n"
            "Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, use DSML markup:\n"
            f"{_DS_OPEN}tool_calls>\n"
            f'{_DS_OPEN}invoke name="tool_name">\n'
            f'{_DS_OPEN}parameter name="param" string="true">value'
            f"{_DS_CLOSE}parameter>\n"
            f"{_DS_CLOSE}invoke>\n"
            f"{_DS_CLOSE}tool_calls>\n\n"
            "After a tool call, wait for the result before continuing."
        )


# ── Format: minimax ──────────────────────────────────────────────────────────
#
# MiniMax M2-family markup, observed live from MiniMax-M2.7 (7/8 turns):
# a <minimax:tool_call> envelope around plain-ASCII invoke/parameter tags:
# <minimax:tool_call>
# <invoke name="Bash">
# <parameter name="command">CMD</parameter>
# </invoke>
# </minimax:tool_call>
# Same invoke/parameter shape as DeepSeek DSML but with regular < > and no
# fullwidth bars. Parameter values are raw element text.


_MINIMAX_INVOKE_PATTERN = re.compile(
    r"(?:<minimax:tool_call>\s*)?"
    r'<invoke name="([^"]+)"\s*>(.*?)</invoke>'
    r"(?:\s*</minimax:tool_call>)?",
    re.DOTALL,
)

_MINIMAX_PARAM_PATTERN = re.compile(
    r'<parameter name="([^"]+)"[^>]*>(.*?)</parameter>',
    re.DOTALL,
)


class MiniMaxFormat(ToolCallFormat):
    """MiniMax format: ``<minimax:tool_call>`` envelope around ``invoke``/``parameter`` tags."""

    stop_sequences = ("</minimax:tool_call>",)

    def parse(self, text: str) -> list[ParsedToolCall]:
        results = []
        for match in _MINIMAX_INVOKE_PATTERN.finditer(text):
            name = match.group(1).strip()
            args: dict[str, object] = {}
            for param in _MINIMAX_PARAM_PATTERN.finditer(match.group(2)):
                args[param.group(1).strip()] = param.group(2)
            if name:
                results.append(ParsedToolCall(
                    name=name, input=args, raw_match=match.group(0),
                ))
        return results

    def detect_malformed(self, text: str) -> bool:
        if "<minimax:tool_call>" not in text and '<invoke name="' not in text:
            return False
        return not self.parse(text)

    def format_error(self) -> str:
        return (
            "Found invoke markup but the tool call did not parse.\n\n"
            "Expected format:\n"
            "<minimax:tool_call>\n"
            '<invoke name="tool_name">\n'
            '<parameter name="param">value</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>\n\n"
            "Please retry with the correct format."
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            "\n\n# Tool Calling\n\n"
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To call a tool, use invoke markup:\n"
            "<minimax:tool_call>\n"
            '<invoke name="tool_name">\n'
            '<parameter name="param">value</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>\n\n"
            "After a tool call, wait for the result before continuing."
        )


# ── Format: auto ─────────────────────────────────────────────────────────────
#
# Frontier models routinely ignore the taught convention and emit their
# TRAINED tool markup instead (observed: Qwen3-Coder emitting hermes_xml
# under a bash_block prompt, 104/108 turns). A skill benchmark should never
# drop a real tool call over markup, so ``auto`` strict-parses every
# concrete format and uses whichever yields calls. The markups are mutually
# distinctive, so the first strict match is unambiguous in practice.


_AUTO_ORDER = [
    "bash_block", "hermes_xml", "hermes", "glm", "kimi", "deepseek",
    "minimax", "alan", "meta_json",
]


class AutoFormat(ToolCallFormat):
    """Auto-detecting format: accept any registered markup, teach bash_block."""

    # Only unambiguous stops: models emit stray <tool_call>-style label
    # chatter BEFORE their real call (observed on GLM-5.2), so a
    # </tool_call> stop can cut the turn before the call is written.
    # Formats configured directly keep their own tag stops.
    stop_sequences = (
        "\n```\n", "<|tool_call_end|>",
        f"{_DS_CLOSE}tool_calls>", "</minimax:tool_call>",
    )

    def repair_stop_truncation(self, text: str) -> str:
        # Each member repair is balance-guarded, so this is a no-op for
        # every format whose markers are absent or already closed.
        for name in _AUTO_ORDER:
            text = FORMATS[name].repair_stop_truncation(text)
        return text

    def _formats(self, source: str) -> list[tuple[str, ToolCallFormat]]:
        return [
            (name, FORMATS[name])
            for name in _AUTO_ORDER
            if source == "text" or FORMATS[name].parse_thinking
        ]

    def parse(self, text: str, source: str = "text") -> list[ParsedToolCall]:
        for name, fmt in self._formats(source):
            calls = fmt.parse(text)
            if calls:
                if name != _AUTO_ORDER[0]:
                    logger.info("auto tool format matched %r", name)
                return calls
        return []

    def detect_malformed(self, text: str) -> bool:
        return any(
            fmt.detect_malformed(text) for _, fmt in self._formats("text")
        )

    def format_error(self) -> str:
        return (
            "Your tool call did not parse in any supported format.\n\n"
            "Preferred format - exactly one fenced bash code block; its "
            "content is executed as a shell command:\n"
            "```bash\n"
            "ls -la\n"
            "```"
        )

    def system_prompt(self, tool_schemas: list[dict]) -> str:
        return FORMATS["bash_block"].system_prompt(tool_schemas)


# ── Registry ─────────────────────────────────────────────────────────────────


FORMATS: dict[str, ToolCallFormat] = {
    "hermes": HermesFormat(),
    "hermes_xml": HermesXMLFormat(),
    "glm": GLMFormat(),
    "alan": AlanFormat(),
    "meta_json": MetaJSONFormat(),
    "bash_block": BashBlockFormat(),
    "kimi": KimiFormat(),
    "deepseek": DeepSeekFormat(),
    "minimax": MiniMaxFormat(),
    "auto": AutoFormat(),
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
    source: str = "text",
) -> ParseResult:
    """Extract tool calls from model text output.

    ``source`` is ``"text"`` for visible content or ``"thinking"`` for
    reasoning content - ``auto`` restricts its thinking pass to formats
    whose ``parse_thinking`` allows it.

    Returns a ParseResult with:
    - ``tool_calls``: successfully parsed tool calls
    - ``cleaned_text``: text with markup removed
    - ``error``: if non-None, the model attempted a tool call but used
      the wrong format. This message should be sent back as a tool
      result error so the model can retry.
    """
    fmt = get_format(format)

    # Try strict parsing first
    if isinstance(fmt, AutoFormat):
        tool_calls = fmt.parse(text, source=source)
    else:
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
