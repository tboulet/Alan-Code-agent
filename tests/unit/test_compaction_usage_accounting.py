"""Layer C bills like any other call: the summarizer must reach cost tracking."""

import pytest

from alancode.compact.compact_auto import compaction_auto
from alancode.backends.base import (
    StreamError,
    StreamMessageDelta,
    StreamMessageStart,
    StreamTextDelta,
)
from alancode.messages.types import AssistantMessage, TextBlock, UserMessage


class RecordingTracker:
    """Minimal CostTracker stand-in: only add_usage is exercised here."""

    def __init__(self):
        self.calls = []

    def add_usage(self, usage, model, duration_ms=0.0):
        self.calls.append((usage, model))


class SummarizerBackend:
    """Yields one usage-reporting summary, optionally failing PTL first."""

    def __init__(self, *, ptl_attempts=0, text="<summary>done</summary>"):
        self.ptl_attempts = ptl_attempts
        self.text = text
        self.attempts = 0

    async def stream(self, messages, system, tools, **kwargs):
        self.attempts += 1
        yield StreamMessageStart(
            model="test-model",
            request_id="req",
            usage={"input_tokens": 300, "cache_read_input_tokens": 40},
        )
        if self.attempts <= self.ptl_attempts:
            yield StreamError(error="prompt is too long: 300000 tokens")
            return
        yield StreamTextDelta(text=self.text)
        yield StreamMessageDelta(stop_reason="end_turn", usage={"output_tokens": 25})


def _history(turns=1):
    """Alternating turns: normalization collapses consecutive same-role ones."""
    messages = []
    for i in range(turns):
        messages.append(UserMessage(content=f"question {i} " * 20))
        messages.append(
            AssistantMessage(content=[TextBlock(text=f"answer {i} " * 20)])
        )
    return messages


@pytest.mark.asyncio
async def test_summarizer_call_is_recorded():
    tracker = RecordingTracker()
    result = await compaction_auto(
        _history(), SummarizerBackend(), settings={}, cost_tracker=tracker,
    )
    assert result is not None
    assert len(tracker.calls) == 1
    usage, model = tracker.calls[0]
    assert usage.input_tokens == 300
    assert usage.cache_read_input_tokens == 40
    assert usage.output_tokens == 25
    assert model == "test-model"


@pytest.mark.asyncio
async def test_discarded_ptl_attempts_are_billed_too():
    # A rejected attempt still consumed input tokens; only recording the
    # successful one under-reports what the provider charged.
    tracker = RecordingTracker()
    backend = SummarizerBackend(ptl_attempts=1)
    result = await compaction_auto(
        _history(turns=5), backend, settings={}, cost_tracker=tracker,
    )
    assert result is not None
    assert backend.attempts == 2
    assert len(tracker.calls) == 2
    assert tracker.calls[0][0].output_tokens == 0  # failed before any output


@pytest.mark.asyncio
async def test_no_tracker_is_still_supported():
    result = await compaction_auto(_history(), SummarizerBackend(), settings={})
    assert result is not None


@pytest.mark.asyncio
async def test_usage_free_backend_records_nothing():
    class SilentBackend:
        async def stream(self, messages, system, tools, **kwargs):
            yield StreamTextDelta(text="<summary>done</summary>")

    tracker = RecordingTracker()
    result = await compaction_auto(
        _history(), SilentBackend(), settings={}, cost_tracker=tracker,
    )
    assert result is not None
    assert tracker.calls == []
