"""Tests for the GUI module: SessionUI, CLIUI, GUIUI, serialization, server."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from alancode.gui.base import SessionUI
from alancode.gui.protocol import OutputEvent, InputRequest
from alancode.gui.serialization import (
    agent_event_to_output,
    cost_summary_event,
    local_output_event,
)
from alancode.gui.server import _cwd_url_segment, _find_available_port
from alancode.messages.types import (
    AssistantMessage,
    RequestStartEvent,
    SystemMessage,
    SystemMessageSubtype,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from alancode.messages.factory import (
    create_assistant_message,
    create_tool_result_message,
    create_user_message,
)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:

    def test_assistant_delta(self):
        msg = AssistantMessage(
            content=[TextBlock(text="hello")],
            hide_in_api=True,
        )
        event = agent_event_to_output(msg)
        assert event.type == "assistant_delta"
        assert event.data["content"][0]["type"] == "text"
        assert event.original is msg

    def test_assistant_final(self):
        msg = create_assistant_message("hello world")
        event = agent_event_to_output(msg)
        assert event.type == "assistant_message"

    def test_user_message(self):
        msg = create_user_message("hi")
        event = agent_event_to_output(msg)
        assert event.type == "user_message"
        assert event.data["content"] == "hi"

    def test_tool_result(self):
        msg = create_tool_result_message("tu_1", "result text")
        event = agent_event_to_output(msg)
        assert event.type == "user_message"

    def test_system_message(self):
        msg = SystemMessage(content="info", subtype=SystemMessageSubtype.INFORMATIONAL)
        event = agent_event_to_output(msg)
        assert event.type == "system_message"

    def test_request_start(self):
        event = agent_event_to_output(RequestStartEvent())
        assert event.type == "request_start"

    def test_assistant_with_tool_use(self):
        msg = AssistantMessage(content=[
            TextBlock(text="Let me check"),
            ToolUseBlock(id="tu_1", name="Bash", input={"command": "ls"}),
        ])
        event = agent_event_to_output(msg)
        assert event.type == "assistant_message"
        assert len(event.data["content"]) == 2

    def test_assistant_with_thinking(self):
        msg = AssistantMessage(content=[
            ThinkingBlock(thinking="reasoning..."),
            TextBlock(text="answer"),
        ])
        event = agent_event_to_output(msg)
        assert event.data["content"][0]["type"] == "thinking"

    def test_cost_summary_event(self):
        event = cost_summary_event(100, 50, 10, 5, 0.001, False)
        assert event.type == "cost_summary"
        assert event.data["input_tokens"] == 100

    def test_local_output_event(self):
        event = local_output_event("Hello", style="green")
        assert event.type == "local_output"
        assert event.data["text"] == "Hello"

    def test_original_preserved(self):
        msg = create_user_message("test")
        event = agent_event_to_output(msg)
        assert event.original is msg


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestProtocol:

    def test_output_event_has_timestamp(self):
        e = OutputEvent(type="test", data={})
        assert e.timestamp
        assert "T" in e.timestamp

    def test_input_request_defaults(self):
        r = InputRequest()
        assert r.type == "prompt"
        assert r.question == "> "
        assert r.options == []


# ---------------------------------------------------------------------------
# Server utility tests
# ---------------------------------------------------------------------------


class TestServerUtils:

    def test_cwd_url_segment(self):
        assert _cwd_url_segment("/home/user/projects/my-project") == "my-project"

    def test_cwd_url_segment_empty(self):
        assert _cwd_url_segment("") == "alan"

    def test_find_available_port(self):
        port = _find_available_port()
        assert isinstance(port, int)
        assert port >= 8420


# ---------------------------------------------------------------------------
# SessionUI ABC test
# ---------------------------------------------------------------------------


class TestSessionUIABC:

    def test_cannot_instantiate_abc(self):
        """SessionUI is abstract — can't be instantiated directly."""
        with pytest.raises(TypeError):
            SessionUI()

    def test_default_lifecycle_methods(self):
        """Lifecycle methods have default no-op implementations."""

        class MinimalUI(SessionUI):
            async def get_input(self, prompt=""):
                return ""
            async def ask_user(self, q, opts):
                return ""
            async def on_agent_event(self, event):
                pass
            async def on_cost(self, usage, cost, unknown):
                pass
            @property
            def console(self):
                return MagicMock()

        ui = MinimalUI()
        # These should not raise
        ui.on_agent_start()
        ui.on_agent_done()
        ui.reset_stream_state()


# ---------------------------------------------------------------------------
# Session name tests
# ---------------------------------------------------------------------------


class TestSessionName:

    def test_session_name_default(self, tmp_path):
        from alancode.session.state import SessionState
        s = SessionState(session_id="test", cwd=str(tmp_path))
        assert s.session_name == ""

    def test_session_name_set_and_persist(self, tmp_path):
        from alancode.session.state import SessionState
        s = SessionState(session_id="test", cwd=str(tmp_path))
        s.session_name = "my-refactor"
        s2 = SessionState(session_id="test", cwd=str(tmp_path))
        assert s2.session_name == "my-refactor"
