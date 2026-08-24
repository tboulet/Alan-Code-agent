"""Recovery paths for malformed native calls and opaque server failures."""

from typing import Any, AsyncGenerator

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    BackendStreamEvent,
    LLMBackend,
    ModelInfo,
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    ThinkingConfig,
    ToolSchema,
)
from alancode.messages.types import AssistantMessage, UserMessage


class MalformedThenValidBackend(LLMBackend):
    def __init__(self):
        self.calls: list[list[dict[str, Any]]] = []

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(context_window=32_768, max_output_tokens=8_192)

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
        self.calls.append(messages)
        yield StreamMessageStart(
            model="test",
            usage={"input_tokens": len(self.calls) * 10},
        )
        if len(self.calls) == 1:
            yield StreamToolUseStart(id="call_1", name="Read")
            yield StreamToolUseInputDelta(
                id="call_1", partial_json="not-json"
            )
            yield StreamError(
                error="Tool call 'Read' returned invalid JSON arguments.",
                error_type="invalid_tool_call",
            )
            return
        yield StreamTextDelta(text="recovered")
        yield StreamMessageDelta(
            stop_reason="end_turn",
            usage={"output_tokens": 3},
        )
        yield StreamMessageStop()


class AlwaysMalformedBackend(MalformedThenValidBackend):
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        **kwargs: Any,
    ) -> AsyncGenerator[BackendStreamEvent, None]:
        self.calls.append(messages)
        yield StreamMessageStart(
            model="test",
            usage={"input_tokens": len(self.calls) * 10},
        )
        yield StreamToolUseStart(id="bad", name="Read")
        yield StreamToolUseInputDelta(id="bad", partial_json="not-json")
        yield StreamError(
            error="Tool call 'Read' returned invalid JSON arguments.",
            error_type="invalid_tool_call",
        )


class OpaqueServerFailureBackend(LLMBackend):
    def __init__(self):
        self.main_calls = 0
        self.summary_calls = 0

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(context_window=32_768, max_output_tokens=8_192)

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
        if system and "summariz" in system[0].lower():
            self.summary_calls += 1
            yield StreamTextDelta(text="<summary>Keep the current task.</summary>")
            yield StreamMessageStop()
            return

        self.main_calls += 1
        if self.main_calls <= 4:
            yield StreamError(
                error="internal inference failure",
                error_type="server_error",
                status_code=500,
            )
            return
        yield StreamTextDelta(text="recovered after compaction")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()


class PartialServerFailureBackend(OpaqueServerFailureBackend):
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: list[str],
        tools: list[ToolSchema],
        **kwargs: Any,
    ) -> AsyncGenerator[BackendStreamEvent, None]:
        if system and "summariz" in system[0].lower():
            self.summary_calls += 1
            yield StreamTextDelta(text="<summary>must not run</summary>")
            return
        self.main_calls += 1
        yield StreamMessageStart(model="test")
        yield StreamTextDelta(text="partial")
        yield StreamError(
            error="internal inference failure",
            error_type="server_error",
            status_code=500,
        )


@pytest.mark.asyncio
async def test_malformed_native_tool_call_gets_feedback_and_retry(tmp_path):
    backend = MalformedThenValidBackend()
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        custom_system_prompt="test",
    )

    events = [
        event async for event in agent.query_events_async("read a file")
    ]

    assert len(backend.calls) == 2
    feedback = [
        event
        for event in events
        if isinstance(event, UserMessage)
        and isinstance(event.content, str)
        and "valid JSON object" in event.content
    ]
    assert feedback
    assert "valid JSON object" in str(backend.calls[1])
    finals = [
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    ]
    assert finals[-1].text == "recovered"
    # The malformed first response was still a billable model call. Preserve
    # the input usage reported before native JSON validation failed.
    assert agent.usage.input_tokens == 30
    assert agent.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_exhausted_native_recovery_updates_last_usage(tmp_path):
    backend = AlwaysMalformedBackend()
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        custom_system_prompt="test",
    )

    events = [
        event async for event in agent.query_events_async("read a file")
    ]

    assert len(backend.calls) == 3
    assert agent.usage.input_tokens == 60
    assert agent.last_usage.input_tokens == 30
    final = [
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    ][-1]
    assert final.api_error == "invalid_tool_call"
    assert final.usage.input_tokens == 30


@pytest.mark.asyncio
async def test_exhausted_opaque_5xx_attempts_emergency_compaction(
    tmp_path, monkeypatch,
):
    backend = OpaqueServerFailureBackend()

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("alancode.api.retry.asyncio.sleep", no_sleep)
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        custom_system_prompt="test",
    )

    events = [
        event async for event in agent.query_events_async("solve the task")
    ]

    assert backend.main_calls == 5
    assert backend.summary_calls == 1
    finals = [
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    ]
    assert finals[-1].text == "recovered after compaction"


@pytest.mark.asyncio
async def test_partial_5xx_is_not_replayed_through_compaction(tmp_path):
    backend = PartialServerFailureBackend()
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        custom_system_prompt="test",
    )

    events = [
        event async for event in agent.query_events_async("solve the task")
    ]

    assert backend.main_calls == 1
    assert backend.summary_calls == 0
    errors = [
        event
        for event in events
        if isinstance(event, AssistantMessage) and event.is_api_error_message
    ]
    assert errors
