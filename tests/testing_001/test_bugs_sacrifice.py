"""Sacrifice-order findings: F03 (breaker fallback destroys all recent
work), F04 (fallback boundary drops the kept head on the next turn),
F08 (Layer A re-truncates its own output, corrupting the sentinel).
"""

import re

from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call

from harness import (
    AuditedProvider,
    FloodTool,
    main_payloads,
    make_agent,
    run_turn,
    user_notices,
    word_soup,
)


def breaker_setup(tmp_path, cw=32_768, n=18):
    inner = ScriptedProvider.from_responses(
        [tool_call("Dummy", {}) for _ in range(n)],
        fallback=text("All done."),
    )
    provider = AuditedProvider(inner, context_window=cw, summarizer_mode="error")
    tool = FloodTool(lambda i: f"RESULT-{i}-MARKER " + word_soup(8_000))
    agent = make_agent(tmp_path, provider, tools=[tool])
    return provider, agent, n


class TestF03BreakerFallbackEatsTail:
    async def test_recent_work_survives_the_fallback(self, tmp_path):
        """_hard_truncate_fallback's docstring: 'the TAIL (recent work)
        [is one of] the most valuable parts of the history'. But its seam
        cleanup pops tail messages until one is plain user text - and in a
        tool loop there is none - so the ENTIRE tail is dropped even when
        the target had room for most of it. Expected: recent results
        survive in the first post-fallback payload."""
        provider, agent, n = breaker_setup(tmp_path)

        events = await run_turn(agent, "TASK-MARKER do the flooding")
        assert user_notices(events, "hard-truncated"), "breaker did not fire"

        post = [
            p for p in main_payloads(provider) if "hard-truncated" in p
        ]
        assert post, "no main call after the fallback"
        survivors = [
            i for i in range(n) if f"RESULT-{i}-MARKER" in post[0]
        ]
        assert survivors, (
            "the fallback dropped ALL recent work despite a target "
            "(0.8*T) that had room for most of the tail - maximal "
            "information loss per token saved"
        )


class TestF04BreakerBoundaryAmnesia:
    async def test_fallback_head_survives_into_next_turn(self, tmp_path):
        """The fallback carefully preserves the head (the task statement)
        for the current call - but emits its boundary marker AFTER the
        kept messages in the permanent record, so the next turn's
        boundary cut drops everything the fallback chose to keep. A
        successful Layer C re-emits the summary after the boundary;
        the fallback re-emits nothing. Expected: the task anchor is
        still in the next turn's payload."""
        provider, agent, n = breaker_setup(tmp_path)

        ev1 = await run_turn(agent, "TASK-MARKER do the flooding")
        assert user_notices(ev1, "hard-truncated"), "breaker did not fire"
        n_calls_before = provider.main_calls

        await run_turn(agent, "so, where were we?")
        next_turn_payloads = main_payloads(provider)[n_calls_before:]
        assert next_turn_payloads
        assert "TASK-MARKER" in next_turn_payloads[0], (
            "the head kept by the fallback (the user's task statement) "
            "was silently dropped at the next turn's boundary cut: only "
            "the notice survived"
        )


class TestF08LayerARetruncation:
    async def test_sentinel_keeps_true_original_size(self, tmp_path):
        """Layer A's output (cap + sentinel + 2 newlines) is longer than
        the cap, so every later iteration re-truncates it: sentinels stack
        up and the newest one reports the already-truncated length as the
        original total. Expected: one sentinel, reporting the true
        original size, on every call."""
        cw = 200_000
        payload = "HEAD " + word_soup(500_000) + " TAIL"
        inner = ScriptedProvider.from_responses(
            [tool_call("Dummy", {}) for _ in range(3)],
            fallback=text("All done."),
        )
        provider = AuditedProvider(inner, context_window=cw)
        agent = make_agent(tmp_path, provider, tools=[FloodTool(payload)])

        await run_turn(agent, "go")

        rx = re.compile(r"of ([\d,]+) chars")
        last = main_payloads(provider)[-1]
        totals = [int(t.replace(",", "")) for t in rx.findall(last)]
        assert totals, "no truncation sentinel found"
        assert all(total == len(payload) for total in totals), (
            f"sentinels report original sizes {totals}, real original "
            f"{len(payload)}: Layer A re-truncated an earlier result"
        )
