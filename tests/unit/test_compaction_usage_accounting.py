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


class CutOffFirstBackend:
    """First summary runs out of budget mid-reasoning, the retry completes."""

    def __init__(self):
        self.attempts = 0

    async def stream(self, messages, system, tools, **kwargs):
        self.attempts += 1
        yield StreamMessageStart(model="test-model", request_id="req")
        if self.attempts == 1:
            yield StreamTextDelta(text="<analysis> 2. **My first action**: Ran `ls -la &&")
            yield StreamMessageDelta(stop_reason="max_tokens", usage={"output_tokens": 2278})
            return
        yield StreamTextDelta(text="<analysis>ok</analysis><summary>Read all 14 notes.</summary>")
        yield StreamMessageDelta(stop_reason="end_turn", usage={"output_tokens": 40})


@pytest.mark.asyncio
async def test_a_cut_off_summary_is_retried_not_accepted():
    """A reasoning model spent the summarizer budget thinking and left 122
    tokens ending mid-command, with no <summary> block. It was accepted as
    the compaction, so the agent resumed knowing only that it had run `ls`
    and re-read everything (Qwen3.8, 16k window)."""
    backend = CutOffFirstBackend()
    result = await compaction_auto(_history(turns=4), backend, settings={})

    assert backend.attempts == 2, "the cut-off summary must not be accepted"
    summary = " ".join(str(getattr(m, "content", "")) for m in result.summary_messages)
    assert "Read all 14 notes" in summary
    assert "ls -la &&" not in summary


class NoThinkingBackend(SummarizerBackend):
    """A custom-endpoint backend: offers kwargs that switch reasoning off."""

    def __init__(self):
        super().__init__()
        self.kwargs_seen = []

    def no_thinking_kwargs(self):
        return {"chat_template_kwargs": {"enable_thinking": False}}

    async def stream(self, messages, system, tools, **kwargs):
        self.kwargs_seen.append(kwargs)
        async for event in super().stream(messages, system, tools, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_summarizer_call_switches_reasoning_off_when_the_backend_can():
    """A reasoning model thought to the output cap on every summarizer
    attempt - four of ~4.5 min each, all failing - because hidden reasoning
    ate the budget the visible <analysis> block already provides for."""
    backend = NoThinkingBackend()
    await compaction_auto(_history(), backend, settings={})
    assert backend.kwargs_seen[0].get("chat_template_kwargs") == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_summarizer_sends_nothing_extra_to_a_backend_without_the_switch():
    # Hosted providers may reject an unknown parameter.
    backend = SummarizerBackend()
    seen = []
    original = backend.stream

    async def spy(messages, system, tools, **kwargs):
        seen.append(kwargs)
        async for event in original(messages, system, tools, **kwargs):
            yield event

    backend.stream = spy
    await compaction_auto(_history(), backend, settings={})
    assert "chat_template_kwargs" not in seen[0]
