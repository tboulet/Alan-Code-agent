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
        summary_chars: int = 0,
        model_max_output_tokens: int = 8_192,
    ) -> None:
        self.inner = inner
        self.cw = context_window
        self.fail_summarizer = fail_summarizer
        # >0 makes the summarizer SUCCEED with an oversized summary, modelling
        # compaction that works and still leaves the payload unsendable.
        self.summary_chars = summary_chars
        self.model_max_output_tokens = model_max_output_tokens
        self.violations: list[dict] = []
        self.calls: list[dict] = []
        self.summarizer_calls = 0

    def get_model_info(self, model: str | None = None) -> ModelInfo:
        return ModelInfo(
            context_window=self.cw,
            max_output_tokens=self.model_max_output_tokens,
        )

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
            body = (
                flood_payload(self.summary_chars) if self.summary_chars
                else "Compact summary of the prior work."
            )
            yield StreamTextDelta(text=f"<summary>{body}</summary>")
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
    # programmatic=True is the default here, but Phase 7 branches on it (a
    # programmatic caller ends the turn on a text truncation), so tests of the
    # interactive recovery ladder override it.
    kwargs.setdefault("programmatic", True)
    return AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
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
        n_calls = 16
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}) for _ in range(n_calls)],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend)

        events = await run_turn(agent, "flood me")
        assert_survived(backend, events)

        # Counted by a real tokenizer (~3.9 chars/token here, not chars/3),
        # 16 capped results cross T on every small window. On 200k
        # (T ~= 151k tokens) no compaction is expected.
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


# On a pure-text truncation an interactive caller recovers first, so a turn
# only reaches the escalated call once the recovery attempts are used up.
RECOVERY_LIMIT = 3
TRUNCATIONS_TO_ESCALATE = RECOVERY_LIMIT + 1


def truncations(n, final="recovered fully"):
    """``n`` length-truncated responses, then a clean finish."""
    return [
        ScriptedResponse(text="partial thought", stop_reason="max_tokens")
        for _ in range(n)
    ] + [text(final)]


# ---------------------------------------------------------------------------
# Scenario 5pre - what a length truncation does depends on WHAT was cut
# ---------------------------------------------------------------------------


class TestLengthTruncation:
    """One remedy for a length truncation: tell the model and let it continue.

    No escalation - the budget is fixed - and no early return for a
    programmatic caller, which previously got neither.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("programmatic", [True, False])
    async def test_truncation_notifies_and_continues(self, tmp_path, programmatic):
        inner = ScriptedBackend.from_responses(
            truncations(1), fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=200_000)
        agent = make_agent(tmp_path, backend, programmatic=programmatic)

        await run_turn(agent, "write something long")
        main = [c for c in backend.calls if c["kind"] == "main"]

        assert len(main) == 2, "the model must get another generation"
        # Same budget: there is no escalated retry any more.
        assert main[1]["max_tokens"] == main[0]["max_tokens"]
        replay = str(main[1]["messages"])
        assert "partial thought" in replay, "the cut attempt stays in history"
        assert "output token budget" in replay, "the model must be told why"

    @pytest.mark.asyncio
    async def test_programmatic_caller_cut_mid_tool_call_is_told_and_continues(
        self, tmp_path,
    ):
        """The programmatic flag must not suppress the remedy: it used to
        return before both, so a harness-driven run got nothing at all."""
        inner = ScriptedBackend.from_responses(
            [truncated_tool_response(), text("recovered fully")],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=200_000)
        tool = CountingTool(flood_payload(100))
        agent = make_agent(tmp_path, backend, tool=tool)   # programmatic=True

        events = await run_turn(agent, "write a big file")

        assert tool.executions == 0, "a call cut mid-argument must not run"
        main = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main) == 2, "the model must be offered another generation"
        assert main[1]["max_tokens"] == main[0]["max_tokens"], "no escalation"
        assert "output token budget" in str(main[1]["messages"])

    @pytest.mark.asyncio
    async def test_consecutive_truncations_are_bounded(self, tmp_path):
        # Unbounded, a model can thrash against the budget for hours.
        inner = ScriptedBackend.from_responses(
            truncations(TRUNCATIONS_TO_ESCALATE + 5), fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=200_000)
        agent = make_agent(tmp_path, backend, programmatic=False)

        await run_turn(agent, "write something long")
        main = [c for c in backend.calls if c["kind"] == "main"]
        assert len(main) == RECOVERY_LIMIT + 1



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
        # Retried at the SAME budget - the budget is fixed, nothing escalates.
        assert main_calls[1]["max_tokens"] == main_calls[0]["max_tokens"]

    @pytest.mark.asyncio
    async def test_not_executed_and_recovery_messages_are_api_valid(self, tmp_path):
        """The retry conversation must pair the dangling tool_use with the
        error tool_result (strict servers 400 otherwise) plus the notice."""
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
        assert "output token budget" in retry


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
    async def test_successful_but_insufficient_summary_still_survives(self, tmp_path):
        """Compaction SUCCEEDS and the payload is still over the blocking limit.

        Found on a live JZ benchmark run: 25 successful summaries alongside 25
        'Conversation too long' hard stops. The liveness fallback used to be
        gated on compaction having FAILED, so a summary that worked and was
        merely too big fell through to the blocking error and killed the turn.
        A session must never die of context, whatever the reason it is over.
        """
        cw = 32_768
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}) for _ in range(14)],
            fallback=text("All done."),
        )
        # The summarizer returns a summary far too large to fit under the
        # blocking limit: compaction reports success, the payload stays illegal.
        backend = AuditedBackend(inner, context_window=cw, summary_chars=90_000)
        agent = make_agent(tmp_path, backend)

        events = await run_turn(agent, "flood me")

        assert backend.summarizer_calls >= 1, "compaction never ran"
        assert final_text(events) != "", "the turn produced no answer at all"
        assert "Conversation too long" not in final_text(events), (
            "a successful-but-oversized summary still killed the turn: the "
            "liveness fallback is gated on compaction FAILING"
        )
        assert_survived(backend, events)

    async def test_failing_summarizer_hard_truncates_and_survives(self, tmp_path):
        """Layer C fails every time: after 3 attempts the breaker trips,
        the fallback hard-truncates with a visible notice, and the session
        finishes the turn AND answers the next one."""
        cw = 32_768
        # Enough results to trip the breaker once but not to re-cross T after
        # the fallback resets it, which would start a second cycle.
        n_calls = 18
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


# ---------------------------------------------------------------------------
# Layer B announces itself
# ---------------------------------------------------------------------------


class _FloodingBash(DummyTool):
    """Layer B only clears results of known tools, so the flood wears Bash's name."""

    @property
    def name(self) -> str:
        return "Bash"


@pytest.mark.asyncio
async def test_layer_b_clearing_yields_a_clear_boundary(tmp_path):
    """Clearing old tool results used to happen silently: the boundary type
    existed but nothing created it, so a harness saw the prompt shrink with
    no event to attribute it to."""
    from alancode.messages.types import SystemMessage, SystemMessageSubtype

    inner = ScriptedBackend.from_responses(
        [tool_call("Bash", {"command": "x"}) for _ in range(14)],
        fallback=text("All done."),
    )
    backend = AuditedBackend(inner, context_window=16_384)
    agent = make_agent(tmp_path, backend, tool=_FloodingBash(flood_payload(9_000)))
    # Isolate Layer B from Layer C, which would otherwise keep the history
    # below B's target. A settings key, not a constructor kwarg.
    assert agent.update_session_setting("compaction_auto_enabled", False) is None

    events = await run_turn(agent, "flood")
    clears = [
        e for e in events
        if isinstance(e, SystemMessage)
        and e.subtype == SystemMessageSubtype.COMPACT_CLEAR_BOUNDARY
    ]
    assert clears, "Layer B fired without announcing it"
    meta = clears[0].compact_clear_metadata
    assert meta.tokens_saved > 0
    assert meta.pre_tokens > meta.tokens_saved


@pytest.mark.asyncio
async def test_clearing_never_pre_empts_summarising(tmp_path):
    """Layer B's target sits above Layer C's threshold so that B cannot fire
    first - which holds only if both measure on one scale. B used a chars/3
    estimate while C used a tokenizer count; chars/3 reads well above a real
    tokenizer, so B crossed its target while C never reached its threshold.
    Measured on Qwen3.8/llama.cpp: 35 clearings, 0 summarizer requests."""
    from alancode.messages.types import SystemMessage, SystemMessageSubtype

    test_output = (
        '  File "solution.py", line 12, in solve\n'
        "    return bfs(grid, start)\n"
        "AssertionError: expected 4 got 3\n"
    ) * 25
    inner = ScriptedBackend.from_responses(
        [tool_call("Bash", {"command": "python run_test.py"}) for _ in range(40)],
        fallback=text("All done."),
    )
    backend = AuditedBackend(inner, context_window=16_384)
    # A name litellm tokenizes with a real tokenizer; for an unrecognised one
    # both layers fall back to chars/3 and agree, hiding the bug.
    agent = make_agent(
        tmp_path, backend, tool=_FloodingBash(test_output[:2000]),
        model="gpt-4o", max_output_tokens=4000,
    )

    events = await run_turn(agent, "fix the tests")
    kinds = [
        e.subtype for e in events
        if isinstance(e, SystemMessage) and e.subtype in (
            SystemMessageSubtype.COMPACT_BOUNDARY,
            SystemMessageSubtype.COMPACT_CLEAR_BOUNDARY,
        )
    ]
    assert kinds, "precondition: the flood must engage context management"
    assert kinds[0] == SystemMessageSubtype.COMPACT_BOUNDARY, (
        f"Layer B cleared before Layer C ever summarised: {[k.value for k in kinds[:4]]}"
    )


def test_token_counter_accepts_alancode_tool_schemas():
    """litellm.token_counter reads OpenAI-shaped tools and raised on
    alancode's schema, so every in-loop count fell back to chars/3."""
    from alancode.tools.builtin.bash import BashTool
    from alancode.utils.tokens import count_tokens_for_call, estimate_message_tokens
    from alancode.messages.factory import create_user_message

    msgs = [create_user_message("def solve(grid):\n    return grid\n" * 150)]
    counted = count_tokens_for_call("gpt-4o", msgs, tools=[BashTool().to_schema()])
    assert counted < estimate_message_tokens(msgs), "still on the chars/3 fallback"
