"""Layer C bills like any other call: the summarizer must reach cost tracking."""

import pytest

from alancode.compact.compact_auto import FORMAT_FEEDBACK_PLACEHOLDER, compaction_auto
from alancode.compact.hard_truncate import hard_truncate_messages
from alancode.utils.tokens import count_tokens_for_call
from alancode.tools.text_tool_parser import get_format
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


class ReasonedToTheCapBackend:
    """First attempt spends the whole budget on hidden reasoning: no text."""

    def __init__(self):
        self.inputs = []

    async def stream(self, messages, system, tools, **kwargs):
        self.inputs.append(len(messages))
        yield StreamMessageStart(model="test-model", request_id="req")
        if len(self.inputs) == 1:
            yield StreamMessageDelta(stop_reason="max_tokens", usage={"output_tokens": 8192})
            return
        yield StreamTextDelta(text="<analysis>ok</analysis><summary>Read all 14 notes.</summary>")
        yield StreamMessageDelta(stop_reason="end_turn", usage={"output_tokens": 40})


@pytest.mark.asyncio
async def test_an_empty_cut_off_summary_is_retried_on_a_smaller_input():
    """GLM-5.3's template has no reasoning switch: attempts that reasoned to
    the cap and wrote nothing were re-sent unchanged until compaction gave up."""
    backend = ReasonedToTheCapBackend()
    result = await compaction_auto(_history(turns=4), backend, settings={})

    assert result is not None
    assert len(backend.inputs) == 2
    assert backend.inputs[1] < backend.inputs[0], "the retry must trim the input"


class InputRecordingBackend(SummarizerBackend):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def stream(self, messages, system, tools, **kwargs):
        self.inputs.append(messages)
        async for event in super().stream(messages, system, tools, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_format_feedback_does_not_reach_the_summarizer():
    """GLM-5.3 summarized a session holding three format errors. It cannot
    write its own special tokens as text, so the summary read "Expected
    format: [ToolName]parameter_name]..." in stand-in brackets, and the next
    28 turns called tools that way."""
    feedback = get_format("glm").format_error()
    history = _history(turns=2) + [
        UserMessage(content=feedback),
        AssistantMessage(content=[TextBlock(text="retrying")]),
    ]
    backend = InputRecordingBackend()

    await compaction_auto(history, backend, settings={"tool_call_format": "glm"})

    sent = str(backend.inputs[0])
    assert "arg_key" in feedback, "the feedback must really spell the syntax"
    assert "arg_key" not in sent
    assert FORMAT_FEEDBACK_PLACEHOLDER in sent


def _digit_grid(rows=40):
    """Spaced digits: close to one token per character on a real tokenizer."""
    return "\n".join(" ".join(str((i * j) % 10) for j in range(64)) for i in range(rows))


def test_hard_truncation_measures_digit_grids_as_the_loop_does():
    """A run's compaction record read 116k tokens while the server had just
    counted 182k: chars/3 under-counts grids of digits. A fallback truncating
    to a target on that scale keeps far more than the target."""
    history = [UserMessage(content="task")]
    for i in range(30):
        history.append(AssistantMessage(content=[TextBlock(text=f"step {i}")]))
        history.append(UserMessage(content=_digit_grid()))
    target = 20_000

    retained, dropped = hard_truncate_messages(history, target, model="gpt-4o")

    assert dropped > 0
    assert count_tokens_for_call("gpt-4o", retained) <= target
    by_chars, _ = hard_truncate_messages(history, target)
    assert count_tokens_for_call("gpt-4o", by_chars) > target, (
        "precondition: on chars/3 the same history overshoots the target"
    )


@pytest.mark.asyncio
async def test_compaction_record_carries_the_measure_that_triggered_it():
    result = await compaction_auto(
        _history(turns=4), SummarizerBackend(), settings={}, current_tokens=182_000,
    )
    assert result.boundary_message.compact_metadata.pre_tokens == 182_000
    assert result.pre_compact_token_count == 182_000


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


class TemplateRefusesNoThinking:
    """A llama.cpp chat template that raises on enable_thinking=false."""

    def __init__(self):
        self.calls = []
        self._rejected = False

    def no_thinking_kwargs(self):
        return {} if self._rejected else {"chat_template_kwargs": {"enable_thinking": False}}

    def reject_no_thinking(self):
        self._rejected = True

    async def stream(self, messages, system, tools, **kwargs):
        self.calls.append(kwargs)
        yield StreamMessageStart(model="test-model", request_id="req")
        if "chat_template_kwargs" in kwargs:
            yield StreamError(
                error="HTTP 500: Jinja Exception: Disabling thinking is not supported."
            )
            return
        yield StreamTextDelta(text="<summary>done</summary>")
        yield StreamMessageDelta(stop_reason="end_turn", usage={"output_tokens": 10})


@pytest.mark.asyncio
async def test_a_template_that_refuses_the_switch_falls_back_and_remembers():
    """Qwen3.8-2.4T's template raised on enable_thinking=false, so every
    summarizer call failed: 6 compactions in one session, all failed, and the
    agent ran on hard truncation alone."""
    backend = TemplateRefusesNoThinking()

    first = await compaction_auto(_history(), backend, settings={})
    assert first is not None, "must fall back to summarizing with reasoning"
    assert "chat_template_kwargs" in backend.calls[0]
    assert "chat_template_kwargs" not in backend.calls[1]

    backend.calls.clear()
    second = await compaction_auto(_history(), backend, settings={})
    assert second is not None
    assert backend.calls and all("chat_template_kwargs" not in c for c in backend.calls), (
        "the refusal must be remembered, not paid for on every compaction"
    )
