"""Liveness findings: F01 (emergency compaction dead), F02 (dead session
on small CW), F07 (PTL matcher misses non-Anthropic phrasings).

Each test asserts the design-intent behaviour and fails on current code.
"""

from typing import Any

from alancode.providers.base import StreamError
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call

from harness import (
    AuditedProvider,
    FloodTool,
    error_messages,
    final_text,
    make_agent,
    run_turn,
    summarizer_payloads,
    user_notices,
    word_soup,
)


class TestF01EmergencyCompactionDead:
    async def test_main_ptl_triggers_emergency_compaction(self, tmp_path):
        """A provider prompt-too-long on the main call must trigger ONE
        emergency compaction and a retry (design: query loop Phase 7 +
        decision plan D6). Currently the PTL exception path returns before
        the emergency branch: summarizer_calls stays 0 and the turn ends
        with a raw error."""
        provider = AuditedProvider(
            ScriptedProvider.from_responses(
                [text("after ptl")], fallback=text("All done."),
            ),
            context_window=32_768,
            main_ptl_first_n=1,
        )
        agent = make_agent(tmp_path, provider)

        events = await run_turn(agent, "hello")
        assert provider.summarizer_calls >= 1, (
            "prompt-too-long on the main call never reached emergency "
            "compaction (dead code path in loop Phase 7)"
        )
        assert final_text(events) == "after ptl"


class TestF02DeadSessionSmallCW:
    async def test_breaker_eventually_saves_a_16k_session(self, tmp_path):
        """I6 (liveness): a session must never die of context. On CW=16384
        with a persistently failing summarizer, the (T, U) band is too
        narrow to accumulate 3 in-turn failures before the blocking check
        ends the turn - and the failure counter resets with each new
        LoopState - so the breaker fallback NEVER fires and every turn
        forever ends with 'Conversation too long'."""
        cw = 16_384
        responses = [tool_call("Dummy", {}) for _ in range(18)]
        inner = ScriptedProvider.from_responses(
            responses, fallback=text("All done."),
        )
        provider = AuditedProvider(inner, context_window=cw, summarizer_mode="error")
        agent = make_agent(tmp_path, provider, payload_chars=12_000)

        saved = False
        for turn in range(4):
            events = await run_turn(agent, f"turn {turn}")
            if user_notices(events, "hard-truncated") or final_text(events) == "All done.":
                saved = True
                break

        assert saved, (
            "4 turns in a row ended with the blocking error and the "
            "liveness fallback never fired: the session is permanently "
            "dead of context (I6 violated)"
        )


class TestF07PtlMatcherMissesProviderPhrasings:
    async def test_vllm_phrased_ptl_triggers_truncation_retry(self, tmp_path):
        """compact_auto detects PTL with a private 'prompt'+'too long'
        substring check instead of api.errors.is_prompt_too_long. A
        vLLM-style 'maximum context length' rejection is treated as a
        generic error: the retry re-sends the identical payload, so a
        deterministic PTL can never be resolved by the middle-truncation
        backstop. Expected: the second attempt's payload is smaller."""
        cw = 32_768

        class VllmPtlOnce(AuditedProvider):
            async def stream(self, messages, system, tools, **kw: Any):
                is_summ = bool(system) and "summariz" in system[0].lower()
                if is_summ and self.summarizer_calls == 0:
                    self.summarizer_calls += 1
                    self.journal.append({
                        "kind": "summarizer", "est": 0,
                        "max_tokens": kw.get("max_tokens"), "cw": self.cw,
                        "illegal": False, "messages": messages,
                        "system": system, "payload_str": str(messages),
                    })
                    yield StreamError(
                        error=(
                            "This model's maximum context length is 32768 "
                            "tokens. However, you requested more tokens."
                        ),
                        error_type="api_error",
                    )
                    return
                async for e in super().stream(messages, system, tools, **kw):
                    yield e

        inner = ScriptedProvider.from_responses(
            [tool_call("Dummy", {}) for _ in range(14)],
            fallback=text("All done."),
        )
        provider = VllmPtlOnce(inner, context_window=cw)
        tool = FloodTool(lambda i: f"R-{i} " + word_soup(8_000))
        agent = make_agent(tmp_path, provider, tools=[tool])

        await run_turn(agent, "flood")
        attempts = summarizer_payloads(provider)
        assert len(attempts) >= 2
        assert len(attempts[1]) < len(attempts[0]), (
            "the PTL retry re-sent an identical payload: the provider's "
            "phrasing was not recognized as prompt-too-long, so no "
            "middle-truncation was applied between attempts"
        )
