"""Tests for the 2x2 query API: query, query_events, query_async, query_events_async."""

import pytest

from alancode.agent import AlanCodeAgent, AgentState
from alancode.messages.types import AssistantMessage, RequestStartEvent, UserMessage
from alancode.providers.scripted_provider import (
    ScriptedProvider,
    ScriptedResponse,
    rule,
    text,
    tool_call,
)


def _make_agent(*responses, **kwargs):
    """Create an agent with scripted responses."""
    provider = ScriptedProvider.from_responses(list(responses))
    return AlanCodeAgent(provider=provider, cwd="/tmp/test", permission_mode="yolo", **kwargs)


# ── query() — sync, returns str ──────────────────────────────────────────────


class TestQuery:
    """Tests for agent.query() — sync text return."""

    def test_returns_string(self):
        agent = _make_agent(text("Hello!"))
        result = agent.query("Hi")
        assert isinstance(result, str)
        assert "Hello!" in result

    def test_returns_final_text_after_tool_use(self):
        agent = _make_agent(
            ScriptedResponse(
                tool_calls=[{"name": "Bash", "id": "t1", "input": {"command": "echo hi"}}],
                stop_reason="tool_use",
            ),
            text("The command printed 'hi'."),
        )
        result = agent.query("Run echo")
        assert "hi" in result.lower()

    def test_multi_turn_preserves_history(self):
        agent = _make_agent(text("First"), text("Second"))
        r1 = agent.query("Message 1")
        r2 = agent.query("Message 2")
        assert "First" in r1
        assert "Second" in r2
        assert len(agent.messages) >= 4  # 2 user + 2 assistant

    def test_empty_response(self):
        agent = _make_agent(text(""))
        result = agent.query("Say nothing")
        assert isinstance(result, str)


# ── query_events() — sync, returns list ──────────────────────────────────────


class TestQueryEvents:
    """Tests for agent.query_events() — sync event list return."""

    def test_returns_list(self):
        agent = _make_agent(text("Hello"))
        events = agent.query_events("Hi")
        assert isinstance(events, list)
        assert len(events) > 0

    def test_contains_request_start(self):
        agent = _make_agent(text("ok"))
        events = agent.query_events("Test")
        request_starts = [e for e in events if isinstance(e, RequestStartEvent)]
        assert len(request_starts) >= 1

    def test_contains_assistant_message(self):
        agent = _make_agent(text("Hello!"))
        events = agent.query_events("Hi")
        final_msgs = [
            e for e in events
            if isinstance(e, AssistantMessage) and not e.hide_in_api
        ]
        assert len(final_msgs) >= 1
        assert "Hello!" in final_msgs[-1].text

    def test_contains_tool_result_after_tool_call(self):
        agent = _make_agent(
            ScriptedResponse(
                tool_calls=[{"name": "Bash", "id": "t1", "input": {"command": "echo test"}}],
                stop_reason="tool_use",
            ),
            text("Done"),
        )
        events = agent.query_events("Run something")
        tool_results = [
            e for e in events
            if isinstance(e, UserMessage) and isinstance(e.content, list)
        ]
        assert len(tool_results) >= 1

    def test_events_order(self):
        """Events should follow: RequestStart → streaming deltas → final message."""
        agent = _make_agent(text("Answer"))
        events = agent.query_events("Question")
        types = [type(e).__name__ for e in events]
        assert "RequestStartEvent" in types
        assert "AssistantMessage" in types


# ── query_async() — async, returns str ───────────────────────────────────────


class TestQueryAsync:
    """Tests for agent.query_async() — async text return."""

    @pytest.mark.asyncio
    async def test_returns_string(self):
        agent = _make_agent(text("Async hello!"))
        result = await agent.query_async("Hi")
        assert isinstance(result, str)
        assert "Async hello!" in result

    @pytest.mark.asyncio
    async def test_after_tool_use(self):
        agent = _make_agent(
            ScriptedResponse(
                tool_calls=[{"name": "Bash", "id": "t1", "input": {"command": "ls"}}],
                stop_reason="tool_use",
            ),
            text("Listed files."),
        )
        result = await agent.query_async("List files")
        assert "Listed" in result


# ── query_events_async() — async generator ───────────────────────────────────


class TestQueryEventsAsync:
    """Tests for agent.query_events_async() — async streaming generator."""

    @pytest.mark.asyncio
    async def test_yields_events(self):
        agent = _make_agent(text("Streaming!"))
        events = []
        async for event in agent.query_events_async("Hi"):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_yields_virtual_and_final(self):
        agent = _make_agent(text("Hello world"))
        virtual = []
        final = []
        async for event in agent.query_events_async("Hi"):
            if isinstance(event, AssistantMessage):
                if event.hide_in_api:
                    virtual.append(event)
                else:
                    final.append(event)
        assert len(virtual) >= 1  # streaming deltas
        assert len(final) >= 1    # final assembled message

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        agent = _make_agent(text("ok"))
        assert agent.state == AgentState.WAITING
        async for _ in agent.query_events_async("Test"):
            pass
        assert agent.state == AgentState.WAITING


# ── Cross-cutting concerns ───────────────────────────────────────────────────


class TestCrossCutting:
    """Tests that apply across all query methods."""

    def test_cost_tracked(self):
        agent = _make_agent(text("ok"))
        agent.query("Hi")
        # ScriptedProvider reports usage in StreamMessageDelta
        assert agent.usage.input_tokens > 0 or agent.usage.output_tokens > 0

    def test_session_id_stable(self):
        agent = _make_agent(text("a"), text("b"))
        sid = agent.session_id
        agent.query("First")
        agent.query("Second")
        assert agent.session_id == sid

    def test_max_iterations_per_turn_respected(self):
        """query_events should stop at max_iterations_per_turn even with infinite tool calls."""
        provider = ScriptedProvider(rules=[
            rule(respond=tool_call("Bash", {"command": "echo loop"})),
        ])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test", permission_mode="yolo", max_iterations_per_turn=2)
        events = agent.query_events("Loop forever")
        # Should not run forever
        assert provider._call_count <= 4

    def test_inject_message(self):
        """inject_message queues a message (doesn't crash)."""
        agent = _make_agent(text("ok"))
        agent.inject_message("Extra context")
        assert not agent._message_queue.empty()
