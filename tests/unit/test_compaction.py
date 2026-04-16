"""Test compaction layers."""

import pytest

from alancode.compact.compact_truncate import (
    REPLACEMENT_MESSAGE,
    compaction_truncate_tool_results,
)
from alancode.compact.compact_clear import (
    CLEARED_MESSAGE,
    compaction_clear_tool_results,
)
from alancode.messages.types import (
    AssistantMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from alancode.messages.factory import (
    create_assistant_message,
    create_tool_result_message,
    create_user_message,
)
from alancode.settings import SETTINGS_DEFAULTS
from alancode.utils.tokens import (
    calculate_token_warning_state,
    estimate_message_tokens,
    get_auto_compact_threshold,
    rough_token_count,
)


# ---------------------------------------------------------------------------
# Tool result budget tests
# ---------------------------------------------------------------------------


class TestToolResultBudget:
    def test_small_results_unchanged(self):
        msg = create_tool_result_message("tu_1", "short result")
        messages = [msg]
        result = compaction_truncate_tool_results(messages)
        assert len(result) == 1
        block = result[0].content[0]
        assert block.content == "short result"

    def test_large_results_truncated(self):
        large_content = "x" * (SETTINGS_DEFAULTS["tool_result_max_chars"] + 1000)
        msg = create_tool_result_message("tu_1", large_content)
        messages = [msg]
        result = compaction_truncate_tool_results(messages)
        assert len(result) == 1
        block = result[0].content[0]
        # Content should be replaced with a truncation message
        assert "truncated" in block.content.lower()
        assert str(len(large_content)) in block.content

    def test_custom_max_chars(self):
        content = "x" * 200
        msg = create_tool_result_message("tu_1", content)
        result = compaction_truncate_tool_results([msg], max_chars=100)
        block = result[0].content[0]
        assert "truncated" in block.content.lower()

    def test_does_not_mutate_input(self):
        large_content = "x" * (SETTINGS_DEFAULTS["tool_result_max_chars"] + 100)
        msg = create_tool_result_message("tu_1", large_content)
        original_content = msg.content[0].content
        compaction_truncate_tool_results([msg])
        # Original should be unchanged
        assert msg.content[0].content == original_content

    def test_plain_user_messages_pass_through(self):
        msg = create_user_message("just text")
        result = compaction_truncate_tool_results([msg])
        assert len(result) == 1
        assert result[0].content == "just text"

    def test_mixed_messages(self):
        user_msg = create_user_message("question")
        assistant_msg = create_assistant_message("answer")
        small_tool = create_tool_result_message("tu_1", "small")
        large_tool = create_tool_result_message("tu_2", "y" * (SETTINGS_DEFAULTS["tool_result_max_chars"] + 1))

        result = compaction_truncate_tool_results([user_msg, assistant_msg, small_tool, large_tool])
        assert len(result) == 4
        # Small tool result unchanged
        assert result[2].content[0].content == "small"
        # Large tool result truncated
        assert "truncated" in result[3].content[0].content.lower()


# ---------------------------------------------------------------------------
# Micro-compact tests
# ---------------------------------------------------------------------------


class TestMicroCompact:
    def _make_tool_exchange(self, tool_name, tool_id, result_text):
        """Helper: create an assistant message with a ToolUseBlock and a user
        message with the corresponding ToolResultBlock."""
        assistant = AssistantMessage(
            content=[
                ToolUseBlock(id=tool_id, name=tool_name, input={"command": "test"}),
            ]
        )
        user = UserMessage(
            content=[
                ToolResultBlock(tool_use_id=tool_id, content=result_text),
            ]
        )
        return assistant, user

    def test_clears_old_tool_results(self):
        messages = []
        # Create 15 tool exchanges (Bash is compactable)
        for i in range(15):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}" * 100)
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(messages, keep_recent=5)
        assert tokens_saved > 0

        # Check that old results (first 10) are cleared
        cleared_count = 0
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.content == CLEARED_MESSAGE:
                        cleared_count += 1
        assert cleared_count == 10  # 15 - 5 recent = 10 cleared

    def test_preserves_recent_results(self):
        messages = []
        for i in range(5):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}")
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(messages, keep_recent=10)
        # All 5 are within keep_recent=10, so nothing cleared
        assert tokens_saved == 0
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        assert block.content != CLEARED_MESSAGE

    def test_returns_tokens_saved(self):
        messages = []
        for i in range(12):
            a, u = self._make_tool_exchange("Read", f"tu_{i}", "long content " * 200)
            messages.extend([a, u])

        _, tokens_saved = compaction_clear_tool_results(messages, keep_recent=2)
        assert tokens_saved > 0

    def test_non_compactable_tools_preserved(self):
        """Tools not in the COMPACTABLE_TOOLS set should never be cleared."""
        messages = []
        for i in range(15):
            a, u = self._make_tool_exchange("CustomTool", f"tu_{i}", f"output_{i}")
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(messages, keep_recent=2)
        assert tokens_saved == 0  # CustomTool is not compactable

    def test_empty_messages(self):
        new_msgs, tokens_saved = compaction_clear_tool_results([])
        assert new_msgs == []
        assert tokens_saved == 0


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_rough_token_count(self):
        # Flat chars/3 fallback (no live calibration anymore).
        assert rough_token_count("hello world") >= 1
        # 1000 chars at the fallback ratio (3.0) -> 333 tokens
        count = rough_token_count("a" * 1000)
        assert count == 333

    def test_rough_token_count_empty(self):
        # Empty string should give at least 1
        assert rough_token_count("") == 1

    def test_estimate_message_tokens(self):
        messages = [
            create_user_message("Hello, how are you?"),
            create_assistant_message("I'm fine, thanks!"),
        ]
        total = estimate_message_tokens(messages)
        # Should be > 0 with message overhead + content
        assert total > 0

    def test_auto_compact_threshold(self):
        threshold = get_auto_compact_threshold(200_000)
        # Should be context_window - max(max_output, 20K) - buffer
        expected = 200_000 - SETTINGS_DEFAULTS["compact_max_output_tokens"] - SETTINGS_DEFAULTS["auto_compact_buffer_tokens"]
        assert threshold == expected

    def test_warning_state_normal(self):
        state = calculate_token_warning_state(10_000, 200_000)
        assert state["is_above_warning"] is False
        assert state["is_above_error"] is False
        assert state["is_at_blocking_limit"] is False
        assert state["percent_left"] > 0.9

    def test_warning_state_above_warning(self):
        # Usable = 200K - 20K = 180K
        # Remaining = 180K - usage
        # Warning threshold is 20K remaining
        usage = 180_000 - SETTINGS_DEFAULTS["warning_threshold_buffer_tokens"] + 1
        state = calculate_token_warning_state(usage, 200_000)
        assert state["is_above_warning"] is True

    def test_warning_state_blocking(self):
        # Usable = 200K - 20K = 180K
        usage = 180_000 - 1000  # Only 1000 tokens remaining
        state = calculate_token_warning_state(usage, 200_000)
        assert state["is_at_blocking_limit"] is True
