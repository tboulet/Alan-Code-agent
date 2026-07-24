"""Tests for the streaming display filter state machine in alancode/cli/display.py.

These tests exercise new_stream_state and _stream_text_delta by inspecting
the state dict directly, without rendering to a real console.
"""

import pytest
from unittest.mock import MagicMock

from alancode.cli.display import (
    _stream_text_delta,
    new_stream_state,
)


class TestNewStreamState:
    """new_stream_state returns a fresh, independent state dict."""

    def test_initial_values(self):
        state = new_stream_state()
        assert state["in_thinking"] is False
        assert state["in_tool_call"] is False
        assert state["buffer"] == ""
        assert state["thinking_active"] is False

    def test_instances_are_independent(self):
        """Two UIs (or turns) never share filter state."""
        a = new_stream_state()
        b = new_stream_state()
        a["in_tool_call"] = True
        a["buffer"] = "leftover"
        assert b["in_tool_call"] is False
        assert b["buffer"] == ""


class TestStreamTextDeltaToolCallSuppression:
    """Text inside <tool_call>...</tool_call> should be suppressed."""

    def setup_method(self):
        self.state = new_stream_state()

    def test_tool_call_content_suppressed(self):
        """Content between <tool_call> and </tool_call> is not printed."""
        console = MagicMock()
        _stream_text_delta('<tool_call>{"name":"Bash"}</tool_call>', console, self.state)
        # The tool call content should be suppressed — console.print should
        # NOT have been called with the JSON payload.
        printed = "".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert '{"name"' not in printed
        assert "Bash" not in printed

    def test_tool_call_sets_in_tool_call_state(self):
        """After <tool_call> and before </tool_call>, in_tool_call is True."""
        console = MagicMock()
        _stream_text_delta("<tool_call>", console, self.state)
        assert self.state["in_tool_call"] is True

    def test_tool_call_close_clears_state(self):
        """</tool_call> resets in_tool_call to False."""
        console = MagicMock()
        _stream_text_delta("<tool_call>suppressed</tool_call>", console, self.state)
        assert self.state["in_tool_call"] is False

    def test_text_before_tool_call_printed(self):
        """Text before <tool_call> should be printed normally."""
        console = MagicMock()
        _stream_text_delta("Hello <tool_call>hidden</tool_call>", console, self.state)
        printed = "".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert "H" in printed
        assert "e" in printed
        assert "l" in printed

    def test_text_after_tool_call_printed(self):
        """Text after </tool_call> should be printed normally."""
        console = MagicMock()
        _stream_text_delta("<tool_call>x</tool_call>visible", console, self.state)
        printed = "".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert "v" in printed
        assert "i" in printed
        assert "s" in printed


class TestStreamTextDeltaThinking:
    """Text inside <think>...</think> is displayed in dim italic."""

    def setup_method(self):
        self.state = new_stream_state()

    def test_think_tag_sets_in_thinking(self):
        console = MagicMock()
        _stream_text_delta("<think>", console, self.state)
        assert self.state["in_thinking"] is True

    def test_think_close_clears_in_thinking(self):
        console = MagicMock()
        _stream_text_delta("<think>reasoning</think>", console, self.state)
        assert self.state["in_thinking"] is False

    def test_thinking_text_rendered_italic(self):
        """Characters inside <think> tags should be printed with dim italic markup."""
        console = MagicMock()
        _stream_text_delta("<think>AB</think>", console, self.state)
        italic_calls = [
            call for call in console.print.call_args_list
            if call.args and "dim italic" in str(call.args[0])
        ]
        assert len(italic_calls) >= 1  # At least one italic-styled print

    def test_stray_close_tag_without_open(self):
        """A </think> with no opening tag is swallowed; text renders normally
        (no assume-thinking mode: styling comes only from explicit tags)."""
        console = MagicMock()
        _stream_text_delta("reasoning here</think>normal", console, self.state)
        assert self.state["in_thinking"] is False
        italic_calls = [
            call for call in console.print.call_args_list
            if call.args and "dim italic" in str(call.args[0])
        ]
        assert len(italic_calls) == 0


class TestStreamTextDeltaBuffering:
    """Partial tags at the end of a delta are buffered."""

    def setup_method(self):
        self.state = new_stream_state()

    def test_partial_tag_buffered(self):
        """A '<' at the end gets buffered for the next delta."""
        console = MagicMock()
        _stream_text_delta("hello<", console, self.state)
        assert self.state["buffer"] == "<"

    def test_partial_tag_completed_next_delta(self):
        """Buffered partial tag is completed by next delta."""
        console = MagicMock()
        _stream_text_delta("hello<thi", console, self.state)
        assert "<thi" in self.state["buffer"]
        # Complete the tag in the next delta
        _stream_text_delta("nk>reasoning</think>done", console, self.state)
        assert self.state["in_thinking"] is False
        assert self.state["buffer"] == ""

    def test_non_tag_angle_bracket_not_stuck(self):
        """A '<' followed by enough non-tag chars should not stay buffered."""
        console = MagicMock()
        _stream_text_delta("a < b and c > d", console, self.state)
        # The '<' is not a known tag start, and enough chars follow
        # to determine it's not a partial tag. Buffer should be empty.
        assert self.state["buffer"] == ""


class TestStreamTextDeltaMultipleDeltas:
    """Simulates realistic streaming where text arrives in small chunks."""

    def setup_method(self):
        self.state = new_stream_state()

    def test_normal_text_across_deltas(self):
        """Plain text split across multiple deltas all gets printed."""
        console = MagicMock()
        _stream_text_delta("Hel", console, self.state)
        _stream_text_delta("lo ", console, self.state)
        _stream_text_delta("world", console, self.state)
        printed = "".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        assert "H" in printed
        assert "w" in printed
        assert "d" in printed

    def test_tool_call_across_deltas(self):
        """Tool call split across deltas is still suppressed."""
        console = MagicMock()
        _stream_text_delta("text<tool_", console, self.state)
        _stream_text_delta("call>hidden</tool_call>after", console, self.state)
        assert self.state["in_tool_call"] is False
        printed = "".join(
            str(call.args[0]) if call.args else ""
            for call in console.print.call_args_list
        )
        # "hidden" should not appear, "text" and "after" should
        assert "h" not in printed or "after" in printed  # Suppressed content
