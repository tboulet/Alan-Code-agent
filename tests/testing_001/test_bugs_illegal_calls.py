"""Illegal-call findings: F05 (CJK content breaks I1), F06 (giant paste
forces an illegal summarizer call), F09 (abort between stream and tool
execution leaves a dangling tool_use in the next turn's payload).
"""

import pytest

from alancode.messages.types import AssistantMessage
from alancode.backends.scripted_backend import (
    ScriptedBackend,
    multi_tool_call,
    text,
    tool_call,
)

from .harness import (
    AuditedBackend,
    FloodTool,
    cjk_soup,
    make_agent,
    run_turn,
    word_soup,
)


@pytest.mark.xfail(
    strict=True,
    reason="Accepted limitation: fallback token estimates are inaccurate for CJK text",
)
class TestF05CjkIllegalCalls:
    async def test_cjk_flood_never_produces_illegal_call(self, tmp_path):
        """I1: input + max_tokens <= CW for every call. The chars/3
        estimator counts a CJK char as 1/3 token while real tokenizers
        count >= 1 token per char, so a CJK flood sails under T and U
        while the real payload exceeds the window ~3x over. The audit
        estimate here counts CJK chars at 1 token each - still an
        underestimate - so every flagged violation is real."""
        cw = 16_384
        inner = ScriptedBackend.from_responses(
            [tool_call("Dummy", {}) for _ in range(10)],
            fallback=text("All done."),
        )
        backend = AuditedBackend(inner, context_window=cw)
        agent = make_agent(tmp_path, backend, tools=[FloodTool(cjk_soup(9_000))])

        await run_turn(agent, "flood me with CJK")
        assert backend.violations == [], (
            f"{len(backend.violations)} illegal call(s) with CJK content "
            f"(first: est={backend.violations[0]['est']} + "
            f"max_tokens={backend.violations[0]['max_tokens']} > {cw}); "
            "on a validating backend every one is a rejected request, and "
            "the margin m=1%*CW cannot absorb a 3x estimator error"
        )


class TestF06GiantPasteIllegalSummarizer:
    async def test_summarizer_call_stays_legal_on_giant_paste(self, tmp_path):
        """A single ~70k-char user message on CW=16384: Layer A/B do not
        touch user messages, C fires at the T crossing (before the
        blocking check), and compact_auto's max(256, clamp(...)) floor
        sends a summarizer call whose input alone exceeds the window
        (I3 violated). Expected: the summarizer input is bounded before
        the call, or the call is not made."""
        cw = 16_384
        backend = AuditedBackend(
            ScriptedBackend.from_responses([], fallback=text("All done.")),
            context_window=cw,
        )
        agent = make_agent(tmp_path, backend)

        await run_turn(agent, "TASK " + word_soup(70_000))
        bad = [v for v in backend.violations if v["kind"] == "summarizer"]
        assert bad == [], (
            f"illegal summarizer call: est={bad[0]['est']} + "
            f"max_tokens={bad[0]['max_tokens']} > {cw}" if bad else ""
        )


class TestF09AbortDanglingToolUse:
    async def test_next_turn_payload_has_no_dangling_tool_use(self, tmp_path):
        """Abort landing between the stream (which produced tool_use
        blocks) and tool execution: the loop returns without synthetic
        tool results, so the permanent record holds an assistant tool_use
        with no matching result. Every real API rejects the next turn's
        payload with a 400. Expected: no dangling tool_use ids reach the
        backend."""
        backend = AuditedBackend(
            ScriptedBackend.from_responses(
                [tool_call("Dummy", {})], fallback=text("All done."),
            ),
            context_window=200_000,
        )
        agent = make_agent(tmp_path, backend)

        async def listener(event):
            if isinstance(event, AssistantMessage) and event.has_tool_use:
                agent._abort_event.set()

        agent.add_event_listener(listener)
        await run_turn(agent, "go")
        agent.remove_event_listener(listener)
        n_before = backend.main_calls

        await run_turn(agent, "next turn")
        next_calls = [
            c for c in backend.journal if c["kind"] == "main"
        ][n_before:]
        assert next_calls
        msgs = next_calls[0]["messages"]
        call_ids = set()
        result_ids = set()
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                call_ids.add(tc["id"])
            if m.get("role") == "tool":
                result_ids.add(m.get("tool_call_id"))
        dangling = call_ids - result_ids
        assert dangling == set(), (
            f"dangling tool_use ids in the next turn's payload: {dangling} "
            "- a real backend rejects this request with a 400"
        )

    async def test_abort_closes_every_pending_call_without_executing(self, tmp_path):
        backend = AuditedBackend(
            ScriptedBackend.from_responses(
                [multi_tool_call(("Dummy", {}), ("Dummy", {}))],
                fallback=text("All done."),
            ),
            context_window=200_000,
        )
        tool = FloodTool("must not run")
        agent = make_agent(tmp_path, backend, tools=[tool])

        async def listener(event):
            if isinstance(event, AssistantMessage) and event.has_tool_use:
                agent._abort_event.set()

        agent.add_event_listener(listener)
        await run_turn(agent, "go")
        agent.remove_event_listener(listener)
        await run_turn(agent, "next turn")

        assert tool.calls == 0
        messages = [
            call for call in backend.journal if call["kind"] == "main"
        ][-1]["messages"]
        call_ids = {
            call["id"]
            for message in messages
            for call in message.get("tool_calls") or []
        }
        result_ids = {
            message.get("tool_call_id")
            for message in messages
            if message.get("role") == "tool"
        }
        assert call_ids == result_ids
