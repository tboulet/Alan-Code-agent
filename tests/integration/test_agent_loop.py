"""Integration tests for the full agentic loop with ScriptedProvider."""

import pytest

from alancode.agent import AlanCodeAgent

from alancode.providers.scripted_provider import ScriptedProvider, ScriptedResponse, text, tool_call
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    RequestStartEvent,
    SystemMessage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Text-only responses
# ---------------------------------------------------------------------------


class TestTextOnlyResponse:
    """Model outputs text only -- verify clean completion."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        provider = ScriptedProvider.from_responses([ScriptedResponse(text="Hello, I can help!")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        events = []
        async for event in agent.query_events_async("Hello"):
            events.append(event)

        # Should have at least one non-virtual AssistantMessage with the text
        assistant_msgs = [
            e
            for e in events
            if isinstance(e, AssistantMessage) and not e.hide_in_api
        ]
        assert len(assistant_msgs) >= 1
        assert "Hello, I can help!" in assistant_msgs[-1].text

    @pytest.mark.asyncio
    async def test_provider_called_once_for_text(self):
        provider = ScriptedProvider.from_responses([ScriptedResponse(text="Done")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        async for _ in agent.query_events_async("Hi"):
            pass

        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_events_include_request_start(self):
        provider = ScriptedProvider.from_responses([ScriptedResponse(text="ok")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        events = []
        async for event in agent.query_events_async("Test"):
            events.append(event)

        # The loop yields a RequestStartEvent before each API call
        request_starts = [e for e in events if isinstance(e, RequestStartEvent)]
        assert len(request_starts) >= 1


# ---------------------------------------------------------------------------
# Tool use responses
# ---------------------------------------------------------------------------


class TestToolUseResponse:
    """Model calls a tool -- verify execution and result injection."""

    @pytest.mark.asyncio
    async def test_tool_call_and_result(self):
        # First response: model calls a tool
        # Second response: model produces final text after seeing tool result
        provider = ScriptedProvider.from_responses(
            [
                ScriptedResponse(
                    tool_calls=[
                        {
                            "name": "Read",
                            "id": "tu_1",
                            "input": {"file_path": "/tmp/test/test.txt"},
                        }
                    ],
                    stop_reason="tool_use",
                ),
                ScriptedResponse(text="I read the file. Here's what I found..."),
            ]
        )
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        events = []
        async for event in agent.query_events_async("Read test.txt"):
            events.append(event)

        # Verify provider was called twice (once for tool call, once for final)
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_tool_result_appears_in_events(self):
        provider = ScriptedProvider.from_responses(
            [
                ScriptedResponse(
                    tool_calls=[
                        {
                            "name": "Bash",
                            "id": "tu_1",
                            "input": {"command": "echo hello"},
                        }
                    ],
                    stop_reason="tool_use",
                ),
                ScriptedResponse(text="Command output was: hello"),
            ]
        )
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        events = []
        async for event in agent.query_events_async("Run echo hello"):
            events.append(event)

        # There should be a UserMessage containing the tool result
        tool_result_msgs = [
            e
            for e in events
            if isinstance(e, UserMessage)
            and isinstance(e.content, list)
            and any(hasattr(b, "tool_use_id") for b in e.content)
        ]
        assert len(tool_result_msgs) >= 1


# ---------------------------------------------------------------------------
# Multi-turn conversation
# ---------------------------------------------------------------------------


class TestMultiTurn:
    """Multi-turn conversation -- verify state persistence."""

    @pytest.mark.asyncio
    async def test_second_turn_sees_history(self):
        provider = ScriptedProvider.from_responses(
            [
                ScriptedResponse(text="First response"),
                ScriptedResponse(text="Second response, I remember the first"),
            ]
        )
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        async for _ in agent.query_events_async("First message"):
            pass
        async for _ in agent.query_events_async("Second message"):
            pass

        # Second call should have seen previous messages in the conversation
        assert len(provider.call_log) == 2
        second_call_messages = provider.call_log[1]["messages"]
        # Should include: first user message, first assistant response,
        # second user message -> at least 3 messages
        assert len(second_call_messages) >= 3

    @pytest.mark.asyncio
    async def test_agent_state_returns_to_waiting(self):
        from alancode.agent import AgentState

        provider = ScriptedProvider.from_responses([ScriptedResponse(text="done")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        assert agent.state == AgentState.WAITING
        async for _ in agent.query_events_async("Hello"):
            pass
        assert agent.state == AgentState.WAITING


# ---------------------------------------------------------------------------
# Max turns limit
# ---------------------------------------------------------------------------


class TestMaxTurns:
    """Max turns limit -- verify the loop stops."""

    @pytest.mark.asyncio
    async def test_stops_at_max_iterations_per_turn(self):
        # Create tool calls that would loop forever
        provider = ScriptedProvider.from_responses(
            [
                ScriptedResponse(
                    tool_calls=[
                        {
                            "name": "Bash",
                            "id": f"tu_{i}",
                            "input": {"command": "echo hi"},
                        }
                    ],
                    stop_reason="tool_use",
                )
                for i in range(20)
            ]
        )
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test", max_iterations_per_turn=3)

        events = []
        async for event in agent.query_events_async("Loop forever"):
            events.append(event)

        # Should have stopped before using all 20 responses.
        # max_iterations_per_turn=3 means 3 tool-execution turns, plus the initial call.
        assert provider._call_count <= 5

        # Should have a max_iterations_per_turn_reached attachment message
        attachment_msgs = [
            e
            for e in events
            if isinstance(e, AttachmentMessage)
            and e.attachment.type == "max_iterations_per_turn_reached"
        ]
        assert len(attachment_msgs) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Provider error -- verify graceful handling."""

    @pytest.mark.asyncio
    async def test_stream_error_yields_error_message(self):
        provider = ScriptedProvider.from_responses(
            [ScriptedResponse(error="Service unavailable")]
        )
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        events = []
        async for event in agent.query_events_async("This will fail"):
            events.append(event)

        # Should yield an error assistant message
        error_msgs = [
            e
            for e in events
            if isinstance(e, AssistantMessage) and e.is_api_error_message
        ]
        assert len(error_msgs) >= 1


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Verify the agent's message list grows correctly."""

    @pytest.mark.asyncio
    async def test_messages_list_grows(self):
        provider = ScriptedProvider.from_responses([ScriptedResponse(text="response")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")

        assert len(agent.messages) == 0

        async for _ in agent.query_events_async("Hello"):
            pass

        # Should have at least: user message + assistant message
        assert len(agent.messages) >= 2

    @pytest.mark.asyncio
    async def test_session_id_is_set(self):
        provider = ScriptedProvider.from_responses([ScriptedResponse(text="ok")])
        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test")
        assert agent.session_id is not None
        assert len(agent.session_id) > 0
