"""Tool-call extraction from reasoning_content streams."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.messages.types import (
    AssistantMessage,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
)
from tests.conftest import EchoTool


class ReasoningToolBackend(LLMBackend):
    def __init__(self):
        self.calls = 0

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="reasoning-test")
        if self.calls == 1:
            yield StreamThinkingDelta(
                thinking=(
                    "I should use the echo tool.\n"
                    '<tool_call>{"name":"Echo","arguments":'
                    '{"text":"hello"}}</tool_call>'
                )
            )
            yield StreamTextDelta(text="Checking that now.")
        else:
            yield StreamTextDelta(text="Done.")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


@pytest.mark.asyncio
async def test_reasoning_tool_call_executes_without_leaking_markup(tmp_path):
    backend = ReasoningToolBackend()
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        tools=[EchoTool()],
        tool_call_format="hermes",
        permission_mode="yolo",
    )

    events = [
        event async for event in agent.query_events_async("Echo hello")
    ]

    assert backend.calls == 2
    assistant_messages = [
        event
        for event in events
        if isinstance(event, AssistantMessage) and not event.hide_in_api
    ]
    tool_message = assistant_messages[0]
    assert any(isinstance(block, ToolUseBlock) for block in tool_message.content)
    thinking = next(
        block
        for block in tool_message.content
        if isinstance(block, ThinkingBlock)
    )
    assert "I should use the echo tool" in thinking.thinking
    assert "<tool_call>" not in thinking.thinking
    assert "<tool_call>" not in tool_message.text

    tool_results = [
        block
        for event in events
        if isinstance(event, UserMessage) and isinstance(event.content, list)
        for block in event.content
        if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].content == "Echo: hello"
    assert assistant_messages[-1].text == "Done."
