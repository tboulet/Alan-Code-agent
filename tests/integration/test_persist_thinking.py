"""persist_thinking: past turns' reasoning re-rendered inline in API history."""

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


class RecordingBackend(LLMBackend):
    """Thinks on the first call; records the message history it receives."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def stream(self, messages, system, tools, **kwargs):
        self.calls.append(messages)
        yield StreamMessageStart(model="recording")
        if len(self.calls) == 1:
            yield StreamThinkingDelta(thinking="my hidden plan")
            yield StreamTextDelta(text="Working on it.")
        else:
            yield StreamTextDelta(text="Done.")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


def _assistant_contents(messages):
    return [m.get("content", "") for m in messages if m.get("role") == "assistant"]


@pytest.mark.asyncio
async def test_thinking_stripped_from_history_by_default(tmp_path):
    backend = RecordingBackend()
    agent = AlanCodeAgent(backend=backend, cwd=str(tmp_path))

    async for _ in agent.query_events_async("first"):
        pass
    async for _ in agent.query_events_async("second"):
        pass

    assert len(backend.calls) == 2
    history = "\n".join(_assistant_contents(backend.calls[1]))
    assert "my hidden plan" not in history
    assert "Working on it." in history
    await agent.close()


@pytest.mark.asyncio
async def test_persist_thinking_reinjects_inline_think_text(tmp_path):
    backend = RecordingBackend()
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), persist_thinking=True,
    )

    async for _ in agent.query_events_async("first"):
        pass
    async for _ in agent.query_events_async("second"):
        pass

    assert len(backend.calls) == 2
    history = "\n".join(_assistant_contents(backend.calls[1]))
    assert "<think>my hidden plan</think>" in history
    assert "Working on it." in history
    await agent.close()
