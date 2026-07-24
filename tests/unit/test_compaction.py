"""Test compaction layers."""

import pytest

from alancode.compact.compact_truncate import (
    TRUNCATION_SENTINEL,
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
    estimate_message_tokens,
    rough_token_count,
)


# ---------------------------------------------------------------------------
# Tool result budget tests
# ---------------------------------------------------------------------------


class TestToolResultBudget:
    def test_small_results_unchanged(self):
        msg = create_tool_result_message("tu_1", "short result")
        messages = [msg]
        result, count = compaction_truncate_tool_results(messages)
        assert len(result) == 1
        assert count == 0
        block = result[0].content[0]
        assert block.content == "short result"

    def test_large_results_truncated_middle_out(self):
        cap = 20_000  # legacy default cap (setting is "auto" now)
        large_content = "H" * cap + "x" * 1000 + "T" * cap
        msg = create_tool_result_message("tu_1", large_content)
        result, count = compaction_truncate_tool_results([msg])
        assert count == 1
        block = result[0].content[0]
        # Head and tail preserved, sentinel in the middle stating the elision
        assert block.content.startswith("H")
        assert block.content.endswith("T")
        assert "elided" in block.content
        assert f"{len(large_content):,}" in block.content

    def test_custom_max_chars_head_tail_split(self):
        content = "A" * 100 + "B" * 100
        msg = create_tool_result_message("tu_1", content)
        result, count = compaction_truncate_tool_results([msg], max_chars=100)
        assert count == 1
        block = result[0].content[0]
        # 60/40 split of the 100-char budget
        assert block.content.startswith("A" * 60)
        assert block.content.endswith("B" * 40)
        assert "elided" in block.content

    def test_does_not_mutate_input(self):
        large_content = "x" * (20_000 + 100)
        msg = create_tool_result_message("tu_1", large_content)
        original_content = msg.content[0].content
        compaction_truncate_tool_results([msg])
        # Original should be unchanged
        assert msg.content[0].content == original_content

    def test_plain_user_messages_pass_through(self):
        msg = create_user_message("just text")
        result, count = compaction_truncate_tool_results([msg])
        assert len(result) == 1
        assert count == 0
        assert result[0].content == "just text"

    def test_mixed_messages(self):
        user_msg = create_user_message("question")
        assistant_msg = create_assistant_message("answer")
        small_tool = create_tool_result_message("tu_1", "small")
        large_tool = create_tool_result_message("tu_2", "y" * (20_000 + 1))

        result, count = compaction_truncate_tool_results(
            [user_msg, assistant_msg, small_tool, large_tool]
        )
        assert len(result) == 4
        assert count == 1
        # Small tool result unchanged
        assert result[2].content[0].content == "small"
        # Large tool result truncated
        assert "elided" in result[3].content[0].content


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

    def test_clears_down_to_target(self):
        messages = []
        # Create 15 tool exchanges (Bash is compactable)
        for i in range(15):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}" * 100)
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=1
        )
        assert tokens_saved > 0
        # With a target of 1, everything compactable gets cleared (no floor)
        cleared_count = 0
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.content == CLEARED_MESSAGE:
                        cleared_count += 1
        assert cleared_count == 15

    def test_gate_at_or_below_target(self):
        messages = []
        for i in range(5):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}")
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=999_999
        )
        # Estimate is far below the target: layer inactive
        assert tokens_saved == 0
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        assert block.content != CLEARED_MESSAGE

    def test_stops_at_target_oldest_first(self):
        """Clearing proceeds oldest-first and stops once the target is reached."""
        messages = []
        for i in range(12):
            a, u = self._make_tool_exchange("Read", f"tu_{i}", "long content " * 200)
            messages.extend([a, u])

        total = estimate_message_tokens(messages)
        per_result = rough_token_count("long content " * 200)
        # Target reachable by clearing roughly three results
        target = total - int(2.5 * per_result)
        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=target
        )
        assert tokens_saved > 0

        cleared = [
            block.tool_use_id
            for msg in new_msgs
            if isinstance(msg, UserMessage) and isinstance(msg.content, list)
            for block in msg.content
            if isinstance(block, ToolResultBlock) and block.content == CLEARED_MESSAGE
        ]
        # Stopped early (nowhere near all 12), oldest first
        assert 1 <= len(cleared) <= 4
        assert cleared == [f"tu_{i}" for i in range(len(cleared))]

    def test_non_compactable_tools_preserved(self):
        """Tools not in the COMPACTABLE_TOOLS set should never be cleared."""
        messages = []
        for i in range(15):
            a, u = self._make_tool_exchange("CustomTool", f"tu_{i}", f"output_{i}")
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=1
        )
        assert tokens_saved == 0  # CustomTool is not compactable

    def test_empty_messages(self):
        new_msgs, tokens_saved = compaction_clear_tool_results(
            [], clear_target_tokens=1
        )
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

