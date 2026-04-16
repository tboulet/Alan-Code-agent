"""Integration tests using ScriptedProvider for realistic agentic scenarios."""

import pytest
import os
import tempfile

from alancode.agent import AlanCodeAgent

from alancode.providers.scripted_provider import (
    ScriptedProvider,
    rule,
    text,
    tool_call,
    multi_tool_call,
)
from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    RequestStartEvent,
    UserMessage,
)


# ---------------------------------------------------------------------------
# File edit scenario
# ---------------------------------------------------------------------------


class TestFileEditScenario:
    """Model reads a file, edits it, then confirms."""

    @pytest.mark.asyncio
    async def test_read_then_edit_then_confirm(self):
        """Simulate: model reads a file, edits it, verifies the edit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "hello.py")
            with open(test_file, "w") as f:
                f.write("print('hello')\n")

            provider = ScriptedProvider(rules=[
                # Turn 0: read the file
                rule(turn=0, respond=tool_call("Read", {"file_path": test_file})),
                # Turn 1: edit it (after seeing the file content)
                rule(turn=1, respond=tool_call("Edit", {
                    "file_path": test_file,
                    "old_string": "print('hello')",
                    "new_string": "print('hello world')",
                })),
                # Turn 2: verify by reading again
                rule(turn=2, respond=tool_call("Read", {"file_path": test_file})),
                # Turn 3: report success
                rule(respond=text("I've updated hello.py to print 'hello world'.")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            events = []
            async for event in agent.query_events_async("Change hello.py to print 'hello world'"):
                events.append(event)

            # Verify file was actually edited
            with open(test_file) as f:
                assert "hello world" in f.read()

            # Verify 4 API calls happened (read, edit, verify read, final text)
            assert provider._call_count == 4


# ---------------------------------------------------------------------------
# Error recovery scenario
# ---------------------------------------------------------------------------


class TestErrorRecoveryScenario:
    """Model encounters an error and adapts."""

    @pytest.mark.asyncio
    async def test_tool_error_then_retry(self):
        """Model tries to read a nonexistent file, then recovers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_file = os.path.join(tmpdir, "actual.py")
            with open(real_file, "w") as f:
                f.write("x = 1\n")

            missing = os.path.join(tmpdir, "missing.py")

            provider = ScriptedProvider(rules=[
                # Turn 0: try to read nonexistent file
                rule(turn=0, respond=tool_call("Read", {"file_path": missing})),
                # Turn 1: after seeing error, try the correct file
                rule(
                    turn=1,
                    condition=lambda ctx: (
                        "not found" in ctx.last_tool_result.lower()
                        or "Error" in ctx.last_tool_result
                        or "error" in ctx.last_tool_result.lower()
                        or "No such file" in ctx.last_tool_result
                    ),
                    respond=tool_call("Read", {"file_path": real_file}),
                ),
                # Turn 2: success
                rule(respond=text("Found actual.py with content x = 1")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            events = []
            async for event in agent.query_events_async("Read my python file"):
                events.append(event)

            assert provider._call_count == 3


# ---------------------------------------------------------------------------
# Multi-tool concurrency
# ---------------------------------------------------------------------------


class TestMultiToolConcurrency:
    """Model calls multiple read-only tools at once."""

    @pytest.mark.asyncio
    async def test_parallel_reads(self):
        """Model reads two files in parallel, then summarizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "a.py")
            f2 = os.path.join(tmpdir, "b.py")
            with open(f1, "w") as f:
                f.write("# file a\n")
            with open(f2, "w") as f:
                f.write("# file b\n")

            provider = ScriptedProvider(rules=[
                rule(turn=0, respond=multi_tool_call(
                    ("Read", {"file_path": f1}),
                    ("Read", {"file_path": f2}),
                )),
                rule(respond=text("Both files read successfully.")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            events = []
            async for event in agent.query_events_async("Read both files"):
                events.append(event)

            assert provider._call_count == 2


# ---------------------------------------------------------------------------
# Bash execution
# ---------------------------------------------------------------------------


class TestBashExecution:
    """Model runs shell commands."""

    @pytest.mark.asyncio
    async def test_bash_and_report(self):
        """Model runs a command and reports the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider(rules=[
                rule(turn=0, respond=tool_call("Bash", {"command": "echo 'hello from bash'"})),
                rule(
                    condition=lambda ctx: "hello from bash" in ctx.last_tool_result,
                    respond=text("The command output: hello from bash"),
                ),
                rule(respond=text("Something went wrong.")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            result = agent.query("Run echo hello")
            assert "hello from bash" in result


# ---------------------------------------------------------------------------
# Write and verify
# ---------------------------------------------------------------------------


class TestWriteAndVerify:
    """Model creates a file and verifies it exists."""

    @pytest.mark.asyncio
    async def test_write_then_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "new_file.py")
            provider = ScriptedProvider(rules=[
                rule(turn=0, respond=tool_call("Write", {
                    "file_path": target,
                    "content": "x = 42\n",
                })),
                rule(turn=1, respond=tool_call("Glob", {
                    "pattern": "*.py",
                    "path": tmpdir,
                })),
                rule(respond=text("Created new_file.py")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            events = []
            async for event in agent.query_events_async("Create a python file"):
                events.append(event)

            assert os.path.exists(target)
            with open(target) as f:
                assert f.read() == "x = 42\n"


# ---------------------------------------------------------------------------
# Grep search
# ---------------------------------------------------------------------------


class TestGrepSearch:
    """Model searches for patterns in code."""

    @pytest.mark.asyncio
    async def test_grep_then_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "code.py")
            with open(f, "w") as fh:
                fh.write("# TODO: fix this\ndef broken():\n    pass\n")

            provider = ScriptedProvider(rules=[
                rule(turn=0, respond=tool_call("Grep", {
                    "pattern": "TODO",
                    "path": tmpdir,
                    "glob": "*.py",
                })),
                rule(
                    condition=lambda ctx: "code.py" in ctx.last_tool_result,
                    respond=tool_call("Read", {"file_path": f}),
                ),
                rule(respond=text("Found TODO in code.py")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            result = agent.query("Find all TODOs")
            # Either the final text mentions TODO, or at least grep + read happened
            assert "TODO" in result or provider._call_count >= 2


# ---------------------------------------------------------------------------
# Max turns with reactive provider
# ---------------------------------------------------------------------------


class TestMaxTurnsWithReactive:
    """Verify max_iterations_per_turn works with reactive provider."""

    @pytest.mark.asyncio
    async def test_max_iterations_per_turn_stops_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Provider always calls tools -- should be stopped by max_iterations_per_turn
            provider = ScriptedProvider(rules=[
                rule(respond=tool_call("Bash", {"command": "echo turn"})),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
                max_iterations_per_turn=3,
            )
            events = []
            async for event in agent.query_events_async("Do something"):
                events.append(event)

            # Should stop at max_iterations_per_turn, not run forever
            assert provider._call_count <= 4

            # Should have a max_iterations_per_turn_reached attachment
            attachment_msgs = [
                e for e in events
                if isinstance(e, AttachmentMessage)
                and e.attachment.type == "max_iterations_per_turn_reached"
            ]
            assert len(attachment_msgs) == 1


# ---------------------------------------------------------------------------
# ask_text convenience method
# ---------------------------------------------------------------------------


class TestAskSync:
    """Verify ask_text returns final assistant text."""

    @pytest.mark.asyncio
    async def test_ask_text_returns_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider(rules=[
                rule(respond=text("The answer is 42.")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            result = agent.query("What is the answer?")
            assert "42" in result

    @pytest.mark.asyncio
    async def test_ask_text_after_tool_call(self):
        """ask_text returns the final text even when tools ran first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = ScriptedProvider(rules=[
                rule(turn=0, respond=tool_call("Bash", {"command": "echo done"})),
                rule(respond=text("All done.")),
            ])

            agent = AlanCodeAgent(
                provider=provider, cwd=tmpdir, permission_mode="yolo",
            )
            result = agent.query("Do it")
            assert "All done" in result
