"""Scripted LLM provider for testing.

``ScriptedProvider`` is the single test provider for Alan Code.  It supports
two modes of operation:

**Sequential** — responses consumed in FIFO order (simple tests)::

    provider = ScriptedProvider.from_responses([
        text("Hello!"),
        tool_call("Bash", {"command": "ls"}),
        text("Done."),
    ])

**Reactive** — responses chosen by rules that inspect the conversation::

    provider = ScriptedProvider(rules=[
        rule(turn=0, respond=tool_call("Bash", {"command": "ls"})),
        rule(condition=lambda ctx: ctx.last_tool_result_contains("error"),
             respond=text("Something went wrong.")),
        rule(respond=text("Done.")),
    ])

Both modes can be combined via ``from_responses(..., fallback=...)`` or by
mixing turn-indexed rules with condition-based ones.
"""

from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from alancode.providers.base import (
    LLMProvider,
    ModelInfo,
    ProviderStreamEvent,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    StreamToolUseStop,
    ThinkingConfig,
    ToolSchema,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Response builders
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScriptedResponse:
    """What the provider should respond with."""

    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    error: str | None = None
    stop_reason: str | None = None  # Auto-inferred if None
    usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.stop_reason is None:
            if self.tool_calls:
                self.stop_reason = "tool_use"
            elif self.error:
                self.stop_reason = "error"
            else:
                self.stop_reason = "end_turn"


def text(content: str) -> ScriptedResponse:
    """Build a text-only response."""
    return ScriptedResponse(text=content)


def tool_call(
    name: str, input: dict[str, Any], *, id: str | None = None
) -> ScriptedResponse:
    """Build a single tool-call response."""
    return ScriptedResponse(
        tool_calls=[{
            "name": name,
            "input": input,
            "id": id or f"toolu_{_uuid.uuid4().hex[:16]}",
        }]
    )


def multi_tool_call(*calls: tuple[str, dict[str, Any]]) -> ScriptedResponse:
    """Build a response with multiple tool calls (executed concurrently if read-only)."""
    return ScriptedResponse(
        tool_calls=[
            {"name": name, "input": inp, "id": f"toolu_{_uuid.uuid4().hex[:16]}"}
            for name, inp in calls
        ]
    )


def error(message: str) -> ScriptedResponse:
    """Build an error response."""
    return ScriptedResponse(error=message)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation context — structured accessors for rule conditions
# ═══════════════════════════════════════════════════════════════════════════════


class ConversationContext:
    """Parsed view of the conversation for writing rule conditions.

    Passed to condition functions so they can inspect the conversation
    without manually parsing message dicts.
    """

    def __init__(self, messages: list[dict[str, Any]], turn: int) -> None:
        self.messages = messages
        self.turn = turn

    # ── Tool results ──────────────────────────────────────────────────

    @property
    def last_tool_result(self) -> str:
        """Text of the most recent tool result (empty string if none)."""
        for msg in reversed(self.messages):
            # OpenAI format
            if msg.get("role") == "tool":
                return msg.get("content", "")
            # Anthropic format
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        c = block.get("content", "")
                        return c if isinstance(c, str) else str(c)
        return ""

    def last_tool_result_contains(self, substring: str) -> bool:
        """Check if the last tool result contains a substring (case-insensitive)."""
        return substring.lower() in self.last_tool_result.lower()

    @property
    def last_tool_result_is_error(self) -> bool:
        """Check if the last tool result was an error."""
        for msg in reversed(self.messages):
            if msg.get("role") == "tool":
                # Can't reliably detect errors in OpenAI format
                return "error" in msg.get("content", "").lower()[:100]
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return block.get("is_error", False)
        return False

    # ── Tool calls ────────────────────────────────────────────────────

    def tool_was_called(self, tool_name: str) -> bool:
        """Check if a specific tool was called anywhere in the conversation."""
        for msg in self.messages:
            # Anthropic format
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name") == tool_name:
                            return True
            # OpenAI format
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if fn.get("name") == tool_name:
                        return True
        return False

    def tool_call_count(self, tool_name: str | None = None) -> int:
        """Count how many times a tool was called (all tools if name is None)."""
        count = 0
        for msg in self.messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if tool_name is None or block.get("name") == tool_name:
                            count += 1
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if tool_name is None or fn.get("name") == tool_name:
                        count += 1
        return count

    # ── User messages ─────────────────────────────────────────────────

    @property
    def last_user_text(self) -> str:
        """Text of the last user message."""
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    return " ".join(parts)
        return ""

    # ── Message counts ────────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def assistant_message_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")


# ═══════════════════════════════════════════════════════════════════════════════
# Rule system
# ═══════════════════════════════════════════════════════════════════════════════

# Condition function receives ConversationContext → bool
ConditionFn = Callable[[ConversationContext], bool]


@dataclass
class Rule:
    """A conditional response rule.

    Rules are evaluated in order; the first match wins.  A rule matches when
    ALL its conditions are satisfied:

    - ``turn`` — matches a specific turn number (0-indexed API call count)
    - ``condition`` — a callable receiving ``ConversationContext``
    - If neither is set, the rule always matches (default fallback).
    """

    respond: ScriptedResponse
    turn: int | None = None
    condition: ConditionFn | None = None

    def matches(self, ctx: ConversationContext) -> bool:
        if self.turn is not None and self.turn != ctx.turn:
            return False
        if self.condition is not None and not self.condition(ctx):
            return False
        return True


def rule(
    respond: ScriptedResponse,
    *,
    turn: int | None = None,
    condition: ConditionFn | None = None,
) -> Rule:
    """Create a response rule."""
    return Rule(respond=respond, turn=turn, condition=condition)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider
# ═══════════════════════════════════════════════════════════════════════════════


class ScriptedProvider(LLMProvider):
    """Test provider with scripted responses — sequential or reactive.

    Every call to ``stream()`` is recorded in ``call_log`` for assertions.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules: list[Rule] = list(rules or [])
        self._call_count: int = 0
        self.call_log: list[dict[str, Any]] = []

    # ── Factory methods ───────────────────────────────────────────────

    @classmethod
    def from_responses(
        cls,
        responses: list[ScriptedResponse],
        *,
        fallback: ScriptedResponse | None = None,
    ) -> ScriptedProvider:
        """Create a provider with FIFO responses (and optional fallback).

        Each response is consumed in order.  If a fallback is provided,
        it's used when the queue is exhausted.  Otherwise, an error is
        yielded.
        """
        rules = [Rule(respond=r, turn=i) for i, r in enumerate(responses)]
        if fallback is not None:
            rules.append(Rule(respond=fallback))
        return cls(rules=rules)

    # ── Convenience helpers ───────────────────────────────────────────

    def add_rule(self, r: Rule) -> None:
        """Append a rule."""
        self._rules.append(r)

    # ── LLMProvider interface ─────────────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ProviderStreamEvent, None]:
        """Evaluate rules and yield the matching response."""

        turn = self._call_count
        self._call_count += 1

        self.call_log.append({
            "messages": messages,
            "system": system,
            "tools": tools,
            "model": model,
            "turn": turn,
        })

        # Build conversation context for rule evaluation
        ctx = ConversationContext(messages, turn)

        # Find first matching rule
        resp: ScriptedResponse | None = None
        for r in self._rules:
            if r.matches(ctx):
                resp = r.respond
                break

        if resp is None:
            yield StreamError(
                error="ScriptedProvider: no matching rule for this conversation state",
                error_type="api_error",
            )
            return

        effective_model = model or "scripted-model"

        # Error response
        if resp.error is not None:
            yield StreamError(error=resp.error, error_type="api_error")
            return

        # Normal response
        yield StreamMessageStart(
            model=effective_model,
            request_id=f"scripted-req-{turn}",
        )

        if resp.text is not None:
            yield StreamTextDelta(text=resp.text)

        if resp.tool_calls:
            for tc in resp.tool_calls:
                tool_id = tc.get("id", f"toolu_{_uuid.uuid4().hex[:16]}")
                tool_name = tc["name"]
                tool_input = tc["input"]

                yield StreamToolUseStart(id=tool_id, name=tool_name)
                yield StreamToolUseInputDelta(
                    id=tool_id, partial_json=json.dumps(tool_input)
                )
                yield StreamToolUseStop(
                    id=tool_id, name=tool_name, input=tool_input
                )

        yield StreamMessageDelta(
            stop_reason=resp.stop_reason,
            usage=resp.usage or {"input_tokens": 100, "output_tokens": 50},
        )
        yield StreamMessageStop()

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            context_window=200_000,
            max_output_tokens=8_192,
            supports_tool_use=True,
            supports_streaming=True,
        )

