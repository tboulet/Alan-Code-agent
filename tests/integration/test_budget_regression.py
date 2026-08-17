"""Budget regression matrix - the full A/B/C pipeline across context windows.

Recreates the reproduction of GitHub issue #2 (tool-result flooding on small
context windows crashing the session) and locks in the redesign's guarantees:

- I1 (call legality): no constructed call violates input + max_tokens <= CW,
  asserted at the backend boundary for every call including summarizer and
  escalated ones. The audit estimate is a deliberate UNDERestimate (chars/4)
  so a flagged violation is real, never estimator noise.
- Compaction reachability: Layer C is attempted whenever the flood crosses
  the threshold - on every window size.
- Liveness (I6): sessions survive floods, giant results, failing summarizers
  (circuit breaker -> hard-truncate fallback), and impossible configs - the
  turn ends gracefully and the NEXT turn still works.

Scenario numbering follows perso_dev/testing_001/ADVERSARIAL_TESTING_BRIEF.md.
"""

from typing import Any, AsyncGenerator

import pytest

from alancode.agent import AlanCodeAgent
from alancode.messages.types import AssistantMessage, UserMessage
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    BackendStreamEvent,
    StreamError,
    StreamMessageStop,
    StreamTextDelta,
    ThinkingConfig,
    ToolSchema,
)
from alancode.backends.scripted_backend import (
    ScriptedBackend,
    ScriptedResponse,
    text,
    tool_call,
)
from alancode.tools.base import Tool, ToolResult, ToolUseContext


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class DummyTool(Tool):
    """Read-only tool returning a fixed payload - the flooding instrument."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    @property
    def name(self) -> str:
        return "Dummy"

    @property
    def description(self) -> str:
        return "Test tool returning a fixed payload."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        return ToolResult(data=self._payload)

    def permission_level(self, args: dict) -> str:
        return "read"


class AuditedBackend(LLMBackend):
    """Wraps a ScriptedBackend, asserting invariant I1 at the boundary.

    - Every stream() call is recorded with an input estimate and its
      max_tokens; ``est + max_tokens > CW`` lands in ``violations``.
      The estimate is chars/4 - a deliberate underestimate for the
      word-soup payloads used here, so violations are never false alarms.
    - Summarizer calls (recognized by the compaction system prompt) are
      answered directly with a scripted summary (or a scripted failure),
      WITHOUT consuming the inner script - keeps turn-indexed rules aligned.
    """

    def __init__(
        self,
        inner: ScriptedBackend,
        context_window: int,
        *,
        fail_summarizer: bool = False,
    ) -> None:
        self.inner = inner
        self.cw = context_window
        self.fail_summarizer = fail_summarizer
        self.violations: list[dict] = []
        self.calls: list[dict] = []
        self.summarizer_calls = 0

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(context_window=self.cw, max_output_tokens=8_192)

    @staticmethod
    def _estimate(messages: list[dict], system: list[str]) -> int:
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        chars += sum(len(s) for s in system)
        return chars // 4  # underestimate: flagged violations are real

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
        est = self._estimate(messages, system)
        is_summarizer = bool(system) and "summariz" in system[0].lower()
        kind = "summarizer" if is_summarizer else "main"
        self.calls.append(
            {"est": est, "max_tokens": max_tokens, "kind": kind,
             "messages": messages}
        )
        if max_tokens is not None and est + max_tokens > self.cw:
            self.violations.append(
                {"kind": kind, "est": est, "max_tokens": max_tokens, "cw": self.cw}
            )

        if is_summarizer:
            self.summarizer_calls += 1
            if self.fail_summarizer:
                yield StreamError(
                    error="scripted summarizer failure", error_type="api_error",
                )
                return
            yield StreamTextDelta(
                text="<summary>Compact summary of the prior work.</summary>"
            )
            yield StreamMessageStop()
            return

        async for event in self.inner.stream(
            messages, system, tools, model=model, max_tokens=max_tokens,
            thinking=thinking, stop_sequences=stop_sequences, **kwargs,
        ):
            yield event


def flood_payload(n_chars: int) -> str:
    """Word-soup payload: varied tokens, no long repeated runs (repeated
    chars compress absurdly in BPE tokenizers and would skew estimates)."""
    words = []
    i = 0
    while sum(len(w) + 1 for w in words) < n_chars:
        words.append(f"word{i}")
        i += 1
    return " ".join(words)


class CountingTool(DummyTool):
    """DummyTool that records how many times it actually ran."""

    def __init__(self, payload: str) -> None:
        super().__init__(payload)
        self.executions = 0

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        self.executions += 1
        return await super().call(args, context)


def make_agent(tmp_path, backend, payload_chars=9_000, tool=None, **kwargs):
    return AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        programmatic=True,
        permission_mode="yolo",
        custom_system_prompt="You are a test agent.",
        tools=[tool or DummyTool(flood_payload(payload_chars))],
        **kwargs,
    )


async def run_turn(agent, prompt: str) -> list:
    return [event async for event in agent.query_events_async(prompt)]


def final_text(events) -> str:
    texts = [
        e.text
        for e in events
        if isinstance(e, AssistantMessage) and not e.hide_in_api and e.text.strip()
    ]
    return texts[-1] if texts else ""


def assert_survived(backend, events, exc=None):
    assert exc is None, f"unhandled exception escaped the loop: {exc!r}"
    assert backend.violations == [], f"I1 violated: {backend.violations}"
    assert events, "turn produced no events"


# ---------------------------------------------------------------------------
# Scenario 3 - tool-result flooding (issue #2 reproduction)
# ---------------------------------------------------------------------------


CW_MATRIX = [8_192, 16_384, 32_768, 200_000]


class TestToolResultFlooding:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cw", CW_MATRIX)
    async def test_flood_survives_and_compaction_fires(self, tmp_path, cw):
        """Many medium results, each under the per-result cap: the original
        issue #2 crash. Must never produce an illegal call, must attempt
        Layer C when over the threshold, and the session must keep working."""
        n_calls = 12
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}) for _ in range(n_calls)],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend)

        events = await run_turn(agent, "flood me")
        assert_survived(backend, events)

        # 12 results x ~9k chars ~= 27k+ tokens: crosses T on every small
        # window. On 200k (T ~= 151k tokens) no compaction is expected.
        if cw <= 32_768:
            assert backend.summarizer_calls >= 1, (
                "Layer C was never attempted despite crossing the threshold"
            )
        else:
            assert backend.summarizer_calls == 0

        assert final_text(events) == "All done."

        # Liveness: the next turn still works.
        events2 = await run_turn(agent, "still there?")
        assert backend.violations == []
        assert final_text(events2) == "All done."


# ---------------------------------------------------------------------------
# Scenario 3 - single giant result
# ---------------------------------------------------------------------------


class TestGiantResult:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cw", [16_384, 32_768, 200_000])
    async def test_giant_result_truncated_middle_out(self, tmp_path, cw):
        """One result far above the cap: Layer A truncates it middle-out
        before the next call; the payload the model sees is bounded and
        carries the sentinel."""
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}), tool_call("Dummy", {})],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend, payload_chars=200_000)

        events = await run_turn(agent, "read the big thing")
        assert_survived(backend, events)
        assert final_text(events) == "All done."

        # From the second main call on, the tool result in the payload must
        # be the truncated version.
        later_main = [c for c in backend.calls if c["kind"] == "main"][1:]
        assert later_main, "expected at least two main calls"
        for call in later_main:
            serialized = str(call["messages"])
            assert "ALAN-TRUNCATED" in serialized
            assert "elided" in serialized


# ---------------------------------------------------------------------------
# Scenario 5 - escalation is clamped to the window
# ---------------------------------------------------------------------------


class TestEscalationClamp:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cw", [16_384, 32_768])
    async def test_escalated_retry_fits_small_window(self, tmp_path, cw):
        """A max_tokens truncation triggers the 64k escalation retry; on a
        window smaller than 64k the retried call must be clamped, and it
        must still grant MORE than the default budget."""
        inner = ScriptedBackend.from_responses(
            [
                ScriptedResponse(text="partial thought", stop_reason="max_tokens"),
                text("recovered fully"),
            ],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend)

        events = await run_turn(agent, "write something long")
        assert_survived(backend, events)
        assert final_text(events) == "recovered fully"

        main_calls = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main_calls) == 2
        retry = main_calls[1]
        # Escalated beyond the default budget, but legal for the window.
        assert retry["max_tokens"] > 8_192
        assert retry["max_tokens"] < cw


# ---------------------------------------------------------------------------
# Scenario 5b - escalation overrides an explicit pin below the target
# ---------------------------------------------------------------------------


class TestEscalationPastPin:
    @pytest.mark.asyncio
    async def test_pin_below_target_escalates(self, tmp_path):
        """An explicit max_output_tokens below escalated_max_tokens is a
        starting budget: a truncation escalates past it (window-clamped)."""
        cw = 32_768
        inner = ScriptedBackend.from_responses(
            [
                ScriptedResponse(text="partial thought", stop_reason="max_tokens"),
                text("recovered fully"),
            ],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend, max_output_tokens=3_000)

        events = await run_turn(agent, "write something long")
        assert_survived(backend, events)
        assert final_text(events) == "recovered fully"

        main_calls = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main_calls) == 2
        assert main_calls[0]["max_tokens"] == 3_000
        assert main_calls[1]["max_tokens"] > 3_000
        assert main_calls[1]["max_tokens"] < cw

    @pytest.mark.asyncio
    async def test_pin_at_target_stays_hard_ceiling(self, tmp_path):
        """A pin >= escalated_max_tokens never escalates: recovery goes
        straight to the resume message at the same budget."""
        cw = 200_000
        inner = ScriptedBackend.from_responses(
            [
                ScriptedResponse(text="partial thought", stop_reason="max_tokens"),
                text("recovered fully"),
            ],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend, max_output_tokens=64_000)

        events = await run_turn(agent, "write something long")
        assert_survived(backend, events)
        assert final_text(events) == "recovered fully"

        main_calls = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main_calls) == 2
        assert main_calls[1]["max_tokens"] == 64_000
        assert "Output token limit hit" in str(main_calls[1]["messages"])


# ---------------------------------------------------------------------------
# Scenario 6 - truncation mid-tool-call: never execute, always recover
# ---------------------------------------------------------------------------


def truncated_tool_response() -> ScriptedResponse:
    return ScriptedResponse(
        tool_calls=[{"name": "Dummy", "input": {}, "id": "toolu_truncated"}],
        stop_reason="max_tokens",
    )


class TestTruncatedToolCall:
    @pytest.mark.asyncio
    async def test_not_executed_and_escalation_retries(self, tmp_path):
        """A response cut at max_tokens while a tool call was in flight:
        the call must NOT run (it may be cut mid-argument yet still parse),
        an error tool_result must answer it, and the escalation retry must
        fire as if there were no tool call."""
        cw = 32_768
        inner = ScriptedBackend.from_responses(
            [truncated_tool_response(), text("recovered fully")],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        tool = CountingTool(flood_payload(1_000))
        agent = make_agent(tmp_path, backend, tool=tool)

        events = await run_turn(agent, "write a big file")
        assert_survived(backend, events)
        assert final_text(events) == "recovered fully"

        assert tool.executions == 0
        results = [
            e for e in events
            if isinstance(e, UserMessage) and "NOT executed" in str(e.content)
        ]
        assert len(results) == 1

        main_calls = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main_calls) == 2
        assert main_calls[1]["max_tokens"] > main_calls[0]["max_tokens"]

    @pytest.mark.asyncio
    async def test_not_executed_and_recovery_messages_are_api_valid(self, tmp_path):
        """Same truncation with escalation unavailable (pin == target): the
        retry conversation must pair the dangling tool_use with the error
        tool_result (strict servers 400 otherwise) plus the resume nudge."""
        cw = 200_000
        inner = ScriptedBackend.from_responses(
            [truncated_tool_response(), text("recovered fully")],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        tool = CountingTool(flood_payload(1_000))
        agent = make_agent(tmp_path, backend, tool=tool, max_output_tokens=64_000)

        events = await run_turn(agent, "write a big file")
        assert_survived(backend, events)
        assert final_text(events) == "recovered fully"
        assert tool.executions == 0

        main_calls = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main_calls) == 2
        retry = str(main_calls[1]["messages"])
        assert "toolu_truncated" in retry
        assert "NOT executed" in retry
        assert "Output token limit hit" in retry


# ---------------------------------------------------------------------------
# Scenario 1 - impossible config fails fast and gracefully
# ---------------------------------------------------------------------------


class TestConfigErrorGraceful:
    @pytest.mark.asyncio
    async def test_output_budget_eats_window(self, tmp_path):
        cw = 32_768
        inner = ScriptedBackend.from_responses([], fallback=text("unreachable"))
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend, max_output_tokens=cw)

        events = await run_turn(agent, "hello")
        errors = [
            e for e in events
            if isinstance(e, AssistantMessage) and e.is_api_error_message
        ]
        assert errors, "expected a graceful config-error message"
        assert "configuration" in errors[-1].text.lower()
        # No API call was ever attempted with the impossible config.
        assert all(c["kind"] != "main" for c in backend.calls)


# ---------------------------------------------------------------------------
# Scenario 4 - circuit breaker -> hard-truncate fallback (liveness, I6)
# ---------------------------------------------------------------------------


class TestBreakerFallbackLiveness:
    @pytest.mark.asyncio
    async def test_failing_summarizer_hard_truncates_and_survives(self, tmp_path):
        """Layer C fails every time: after 3 attempts the breaker trips,
        the fallback hard-truncates with a visible notice, and the session
        finishes the turn AND answers the next one."""
        cw = 32_768
        n_calls = 14
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}) for _ in range(n_calls)],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw, fail_summarizer=True)
        agent = make_agent(tmp_path, backend)

        events = await run_turn(agent, "flood me")
        assert_survived(backend, events)

        # Exactly 3 compaction invocations, then the breaker path (no 4th).
        # Each failed invocation makes (1 + max_compact_ptl_retries) = 4
        # backend calls internally (the retry loop also consumes generic
        # stream errors), so 3 invocations = 12 summarizer calls.
        assert backend.summarizer_calls == 12

        notices = [
            e for e in events
            if isinstance(e, UserMessage)
            and isinstance(e.content, str)
            and "hard-truncated" in e.content
        ]
        assert notices, "expected the visible hard-truncation notice"

        assert final_text(events) == "All done."

        events2 = await run_turn(agent, "still there?")
        assert backend.violations == []
        assert final_text(events2) == "All done."
