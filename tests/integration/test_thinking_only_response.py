"""Thinking-only responses: in-send nudge, then a distinct (non-API-error) surface."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
)
from alancode.messages.types import AssistantMessage, ThinkingBlock, UserMessage
from alancode.query.loop import EMPTY_RESPONSE_NUDGE


class ThinkingOnlyBackend(LLMBackend):
    """Always answers with private reasoning and no visible text."""

    def __init__(self):
        self.calls = 0

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="thinking-only")
        yield StreamThinkingDelta(thinking="Private reasoning without an answer")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


class InlineThinkingOnlyBackend(LLMBackend):
    async def stream(self, messages, system, tools, **kwargs):
        yield StreamMessageStart(model="inline-thinking-only")
        yield StreamTextDelta(text="<think>Private inline reasoning</think>")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


class ThinkThenAnswerBackend(LLMBackend):
    """Thinking-only on the first call, a real answer once nudged."""

    def __init__(self):
        self.calls = 0

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="think-then-answer")
        if self.calls == 1:
            yield StreamThinkingDelta(thinking="Deep thought, no action")
        else:
            yield StreamTextDelta(text="Recovered answer.")
        yield StreamMessageDelta(
            stop_reason="end_turn",
            usage={"input_tokens": self.calls, "output_tokens": 1},
        )
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


def _final_assistant(events):
    return [
        e for e in events
        if isinstance(e, AssistantMessage) and not e.hide_in_api
    ][-1]


@pytest.mark.asyncio
async def test_nudge_recovers_a_thinking_only_turn(tmp_path):
    backend = ThinkThenAnswerBackend()
    agent = AlanCodeAgent(backend=backend, cwd=str(tmp_path))

    events = [event async for event in agent.query_events_async("answer me")]

    assert backend.calls == 2
    nudges = [
        e for e in events
        if isinstance(e, UserMessage)
        and isinstance(e.content, str)
        and EMPTY_RESPONSE_NUDGE in e.content
    ]
    assert len(nudges) == 1
    final = _final_assistant(events)
    assert final.api_error is None
    assert "Recovered answer." in final.text
    # The thinking-only call that triggered the nudge is still billable and
    # must not disappear from session accounting.
    assert agent.usage.input_tokens == 3
    assert agent.usage.output_tokens == 2
    await agent.close()


@pytest.mark.asyncio
async def test_nudges_exhaust_then_surface_distinct_signal(tmp_path):
    backend = ThinkingOnlyBackend()
    agent = AlanCodeAgent(backend=backend, cwd=str(tmp_path))

    events = [event async for event in agent.query_events_async("answer me")]

    # Default empty_response_retries=2 -> initial call + 2 nudged retries.
    assert backend.calls == 3
    final = _final_assistant(events)
    assert final.api_error == "empty_response"
    assert not final.is_api_error_message
    assert final.text == "Model returned reasoning but no visible answer or tool call."
    assert "Private reasoning" not in final.text
    assert any(isinstance(block, ThinkingBlock) for block in final.content)
    await agent.close()


@pytest.mark.asyncio
async def test_zero_retries_surfaces_immediately(tmp_path):
    backend = ThinkingOnlyBackend()
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), empty_response_retries=0,
    )

    events = [event async for event in agent.query_events_async("answer me")]

    assert backend.calls == 1
    final = _final_assistant(events)
    assert final.api_error == "empty_response"
    assert not final.is_api_error_message
    assert final.text == "Model returned reasoning but no visible answer or tool call."
    await agent.close()


@pytest.mark.asyncio
async def test_inline_thinking_only_turn_surfaces_without_reasoning_leak(tmp_path):
    agent = AlanCodeAgent(
        backend=InlineThinkingOnlyBackend(), cwd=str(tmp_path),
        empty_response_retries=0,
    )

    events = [event async for event in agent.query_events_async("answer me")]

    final = _final_assistant(events)
    assert final.api_error == "empty_response"
    assert not final.is_api_error_message
    assert final.text == "Model returned reasoning but no visible answer or tool call."
    assert "Private inline reasoning" not in final.text
    assert any(isinstance(block, ThinkingBlock) for block in final.content)
    await agent.close()
