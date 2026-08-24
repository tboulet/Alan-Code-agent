"""no_verbalize_warning: remind the model when it acts without narrating."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    StreamToolUseStop,
)
from alancode.messages.types import UserMessage

REMINDER_MARK = "called tools without any visible text"


class SilentToolThenAnswerBackend(LLMBackend):
    """First call runs a tool with no visible text, second call answers."""

    def __init__(self, preamble: str | None = None):
        self.calls = 0
        self._preamble = preamble

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="silent-tool")
        if self.calls == 1:
            if self._preamble:
                yield StreamTextDelta(text=self._preamble)
            yield StreamToolUseStart(id="call_1", name="Bash")
            yield StreamToolUseInputDelta(
                id="call_1", partial_json='{"command": "echo hi"}'
            )
            yield StreamToolUseStop(
                id="call_1", name="Bash", input={"command": "echo hi"}
            )
            yield StreamMessageDelta(stop_reason="tool_use")
        else:
            yield StreamTextDelta(text="Done.")
            yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


def _reminders(events):
    return [
        e for e in events
        if isinstance(e, UserMessage)
        and isinstance(e.content, str)
        and REMINDER_MARK in e.content
    ]


@pytest.mark.asyncio
async def test_reminder_absent_by_default(tmp_path):
    backend = SilentToolThenAnswerBackend()
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), permission_mode="yolo"
    )

    events = [e async for e in agent.query_events_async("run it")]

    assert _reminders(events) == []


@pytest.mark.asyncio
async def test_reminder_injected_when_enabled(tmp_path):
    backend = SilentToolThenAnswerBackend()
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        permission_mode="yolo",
        no_verbalize_warning=True,
    )

    events = [e async for e in agent.query_events_async("run it")]

    assert len(_reminders(events)) == 1
    # Not a retry: the silent turn's tool call still ran, so the model was
    # called again only to continue after the results.
    assert backend.calls == 2


@pytest.mark.asyncio
async def test_no_reminder_when_the_model_narrates(tmp_path):
    backend = SilentToolThenAnswerBackend(preamble="Checking the shell first.")
    agent = AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        permission_mode="yolo",
        no_verbalize_warning=True,
    )

    events = [e async for e in agent.query_events_async("run it")]

    assert _reminders(events) == []
