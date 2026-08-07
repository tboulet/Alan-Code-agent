"""Thinking-only responses must not look like successful empty turns."""

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
from alancode.messages.types import AssistantMessage, ThinkingBlock


class ThinkingOnlyBackend(LLMBackend):
    async def stream(self, messages, system, tools, **kwargs):
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


@pytest.mark.asyncio
async def test_thinking_only_turn_is_visible_error_without_reasoning_leak(tmp_path):
    agent = AlanCodeAgent(backend=ThinkingOnlyBackend(), cwd=str(tmp_path))

    events = [event async for event in agent.query_events_async("answer me")]

    message = next(
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    )
    assert message.is_api_error_message
    assert message.api_error == "empty_response"
    assert message.text == "Model returned reasoning but no visible answer or tool call."
    assert "Private reasoning" not in message.text
    assert any(isinstance(block, ThinkingBlock) for block in message.content)
    await agent.close()


@pytest.mark.asyncio
async def test_inline_thinking_only_turn_is_also_a_visible_error(tmp_path):
    agent = AlanCodeAgent(backend=InlineThinkingOnlyBackend(), cwd=str(tmp_path))

    events = [event async for event in agent.query_events_async("answer me")]

    message = next(
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    )
    assert message.is_api_error_message
    assert message.api_error == "empty_response"
    assert message.text == "Model returned reasoning but no visible answer or tool call."
    assert "Private inline reasoning" not in message.text
    assert any(isinstance(block, ThinkingBlock) for block in message.content)
    await agent.close()
