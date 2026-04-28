"""Tests for system reminders (date, git status, queued messages)."""

import pytest
import os
import tempfile
import subprocess

from alancode.query.loop import _build_turn_reminders, _drain_message_queue
from alancode.tools.base import ToolUseContext
from alancode.messages.types import UserMessage


class TestTurnReminders:
    """Tests for _build_turn_reminders() — date+time, once per turn."""

    def test_contains_date_and_time(self):
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        reminders = _build_turn_reminders(ctx)
        assert len(reminders) == 1
        text = reminders[0].content
        assert isinstance(text, str)
        assert "currentDateTime" in text
        assert "<system-reminder>" in text
        # Should contain date in YYYY-MM-DD HH:MM format
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text)

    def test_hidden_in_ui(self):
        ctx = ToolUseContext(cwd="/tmp", messages=[])
        reminders = _build_turn_reminders(ctx)
        assert reminders[0].hide_in_ui is True


class TestGitStatusNotInSystemPrompt:
    """Git status is intentionally excluded from the system prompt — its
    output changes after every file edit and would invalidate the prompt
    cache on every turn for coding sessions. The agent runs ``git status``
    via Bash when it actually needs the current state.
    """

    def test_environment_section_excludes_git_status(self):
        from alancode.prompt.system_prompt import get_environment_section
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir, capture_output=True,
            )
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("hello")

            env = get_environment_section(model="test", cwd=tmpdir)
            assert "gitStatus:" not in env
            assert "snapshot in time" not in env


class TestDrainMessageQueue:
    """Tests for _drain_message_queue()."""

    def test_empty_queue(self):
        assert _drain_message_queue(None) == []
        assert _drain_message_queue([]) == []

    def test_single_message(self):
        queue = ["Hello from user"]
        messages = _drain_message_queue(queue)
        assert len(messages) == 1
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "Hello from user"
        assert queue == []

    def test_multiple_messages(self):
        queue = ["First", "Second", "Third"]
        messages = _drain_message_queue(queue)
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[2].content == "Third"
        assert queue == []

    def test_queue_is_consumed(self):
        queue = ["Only once"]
        first = _drain_message_queue(queue)
        second = _drain_message_queue(queue)
        assert len(first) == 1
        assert len(second) == 0


class TestInjectMessageIntegration:
    """Integration: inject_message → queued → consumed in loop."""

    @pytest.mark.asyncio
    async def test_injected_message_reaches_model(self):
        from alancode.agent import AlanCodeAgent
        from alancode.providers.scripted_provider import (
            ScriptedProvider, rule, text, tool_call,
        )

        provider = ScriptedProvider(rules=[
            rule(turn=0, respond=tool_call("Bash", {"command": "echo hi"})),
            rule(turn=1, respond=text("I see your injected message.")),
        ])

        agent = AlanCodeAgent(provider=provider, cwd="/tmp/test", permission_mode="yolo")
        agent.inject_message("Extra context from the user")

        events = agent.query_events("Do something")
        assert provider._call_count == 2

        second_call_msgs = provider.call_log[1]["messages"]
        all_content = str(second_call_msgs)
        assert "Extra context from the user" in all_content


class TestSystemPromptSessionTime:
    """Test that system prompt contains fixed session start time."""

    def test_environment_section_has_session_time(self):
        from alancode.prompt.system_prompt import get_environment_section
        env = get_environment_section(model="test", cwd="/tmp")
        assert "Session started:" in env
        # Should contain YYYY-MM-DD HH:MM format
        import re
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", env)
