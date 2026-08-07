"""Self-contained instruments for the adversarial context-budget tests."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    BackendStreamEvent,
    LLMBackend,
    ModelInfo,
    StreamError,
    StreamMessageStop,
    StreamTextDelta,
    ThinkingConfig,
    ToolSchema,
)
from alancode.backends.scripted_backend import ScriptedBackend
from alancode.messages.types import AssistantMessage, UserMessage
from alancode.tools.base import Tool, ToolResult, ToolUseContext


SUMMARY_TEXT = "<summary>SUMMARY-MARKER: compact prior work.</summary>"
CJK_START = 0x3000


def word_soup(n_chars: int, seed: str = "word") -> str:
    """Return varied ASCII text of exactly ``n_chars`` characters."""
    words: list[str] = []
    total = 0
    index = 0
    while total < n_chars:
        word = f"{seed}{index}"
        words.append(word)
        total += len(word) + 1
        index += 1
    return " ".join(words)[:n_chars]


def cjk_soup(n_chars: int) -> str:
    """Return a CJK payload that exposes chars-per-token underestimates."""
    block = "".join(chr(0x4E00 + (index % 2000)) for index in range(200))
    return (block * (n_chars // len(block) + 1))[:n_chars]


class AuditedBackend(LLMBackend):
    """Backend-boundary context sensor wrapped around a scripted backend."""

    def __init__(
        self,
        inner: ScriptedBackend,
        context_window: int,
        *,
        max_output_tokens: int = 8_192,
        summarizer_mode: str = "ok",
        main_ptl_first_n: int = 0,
    ) -> None:
        self.inner = inner
        self.cw = context_window
        self.max_output = max_output_tokens
        self.summarizer_mode = summarizer_mode
        self.main_ptl_first_n = main_ptl_first_n
        self.violations: list[dict[str, Any]] = []
        self.journal: list[dict[str, Any]] = []
        self.summarizer_calls = 0
        self.main_calls = 0

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            context_window=self.cw,
            max_output_tokens=self.max_output,
        )

    @staticmethod
    def _underestimate_text(text: str) -> int:
        cjk_chars = sum(1 for char in text if ord(char) >= CJK_START)
        return cjk_chars + (len(text) - cjk_chars) // 4

    @classmethod
    def _estimate(cls, messages: list[dict], system: list[str]) -> int:
        message_tokens = sum(
            cls._underestimate_text(str(message.get("content", "")))
            for message in messages
        )
        return message_tokens + sum(cls._underestimate_text(s) for s in system)

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
    ) -> AsyncGenerator[BackendStreamEvent, None]:
        estimate = self._estimate(messages, system)
        is_summarizer = bool(system) and "summariz" in system[0].lower()
        illegal = max_tokens is not None and estimate + max_tokens > self.cw
        entry = {
            "kind": "summarizer" if is_summarizer else "main",
            "est": estimate,
            "max_tokens": max_tokens,
            "cw": self.cw,
            "illegal": illegal,
            "messages": messages,
            "system": system,
            "payload_str": str(messages),
        }
        self.journal.append(entry)
        if illegal:
            self.violations.append(entry)

        if is_summarizer:
            self.summarizer_calls += 1
            if self.summarizer_mode == "error":
                yield StreamError(
                    error="scripted summarizer failure",
                    error_type="api_error",
                )
                return
            yield StreamTextDelta(text=SUMMARY_TEXT)
            yield StreamMessageStop()
            return

        self.main_calls += 1
        if self.main_calls <= self.main_ptl_first_n:
            yield StreamError(
                error="prompt is too long for this model",
                error_type="api_error",
            )
            return

        async for event in self.inner.stream(
            messages,
            system,
            tools,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            stop_sequences=stop_sequences,
            **kwargs,
        ):
            yield event


class FloodTool(Tool):
    """Test tool returning fixed, per-call, or generated payloads."""

    def __init__(
        self,
        payload: str | list[str] | Callable[[int], str],
        *,
        tool_name: str = "Dummy",
    ) -> None:
        self._payload = payload
        self._name = tool_name
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Test tool returning a configurable payload."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(
        self, args: dict, context: ToolUseContext
    ) -> ToolResult:
        index = self.calls
        self.calls += 1
        payload = self._payload
        if callable(payload):
            data = payload(index)
        elif isinstance(payload, list):
            data = payload[min(index, len(payload) - 1)]
        else:
            data = payload
        return ToolResult(data=data)

    def permission_level(self, args: dict) -> str:
        return "read"


def make_agent(
    tmp_path,
    backend: LLMBackend,
    *,
    tools: list[Tool] | None = None,
    payload_chars: int = 9_000,
    settings: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AlanCodeAgent:
    if tools is None:
        tools = [FloodTool(word_soup(payload_chars))]
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        permission_mode="yolo",
        custom_system_prompt="You are a test agent.",
        tools=tools,
        **kwargs,
    )
    if settings:
        agent._settings.update(settings)
    return agent


async def run_turn(agent: AlanCodeAgent, prompt: str) -> list:
    return [event async for event in agent.query_events_async(prompt)]


def final_text(events: list) -> str:
    texts = [
        event.text
        for event in events
        if isinstance(event, AssistantMessage)
        and not event.hide_in_api
        and event.text.strip()
    ]
    return texts[-1] if texts else ""


def error_messages(events: list) -> list[AssistantMessage]:
    return [
        event
        for event in events
        if isinstance(event, AssistantMessage) and event.is_api_error_message
    ]


def user_notices(events: list, needle: str) -> list[UserMessage]:
    return [
        event
        for event in events
        if isinstance(event, UserMessage)
        and isinstance(event.content, str)
        and needle in event.content
    ]


def main_payloads(backend: AuditedBackend) -> list[str]:
    return [
        call["payload_str"]
        for call in backend.journal
        if call["kind"] == "main"
    ]


def summarizer_payloads(backend: AuditedBackend) -> list[str]:
    return [
        call["payload_str"]
        for call in backend.journal
        if call["kind"] == "summarizer"
    ]
