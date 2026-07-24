"""Tests for the compaction system upgrade.

Tests cover:
- compaction_truncate_tool_results: threshold gating and truncation behavior
- compaction_clear_tool_results: threshold gating and clear behavior
- truncate_middle_for_ptl: preserves start/end, cuts middle
- format_compact_summary: extracts <summary> tags, strips <analysis>
- get_compact_prompt: contains all 9 sections
- Circuit breaker logic
"""

import pytest

from alancode.compact.compact_truncate import (
    TRUNCATION_SENTINEL,
    compaction_truncate_tool_results,
)
from alancode.compact.compact_clear import (
    CLEARED_MESSAGE,
    compaction_clear_tool_results,
)
from alancode.compact.compact_auto import truncate_middle_for_ptl
from alancode.compact.prompt import (
    format_compact_summary,
    get_compact_prompt,
    get_post_compact_message,
    get_post_compact_notification,
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
from alancode.utils.tokens import estimate_message_tokens, rough_token_count


# ---------------------------------------------------------------------------
# compaction_truncate (Layer 1) tests
# ---------------------------------------------------------------------------


class TestCompactionTruncate:
    def test_always_truncates_oversized(self):
        """Layer A is always on: any result over max_chars is truncated."""
        msg = create_tool_result_message("tu_1", "x" * 100_000)
        result, count = compaction_truncate_tool_results([msg], max_chars=50_000)
        assert len(result) == 1
        assert count == 1
        block = result[0].content[0]
        assert "ALAN-TRUNCATED" in block.content

    def test_all_oversized_processed(self):
        """Every oversized result is truncated, not just the first."""
        msg1 = create_tool_result_message("tu_1", "FIRST" * 20_000)
        msg2 = create_tool_result_message("tu_2", "SECOND" * 20_000)
        result, count = compaction_truncate_tool_results([msg1, msg2], max_chars=50_000)
        assert count == 2
        assert "ALAN-TRUNCATED" in result[0].content[0].content
        assert "ALAN-TRUNCATED" in result[1].content[0].content

    def test_does_not_mutate_input(self):
        """Original messages are not modified."""
        msg = create_tool_result_message("tu_1", "x" * 100_000)
        original_content = msg.content[0].content
        compaction_truncate_tool_results([msg], max_chars=50_000)
        assert msg.content[0].content == original_content


# ---------------------------------------------------------------------------
# compaction_clear (Layer 3) tests
# ---------------------------------------------------------------------------


class TestCompactionClear:
    def _make_tool_exchange(self, tool_name, tool_id, result_text):
        """Create an assistant message with ToolUseBlock + user message with ToolResultBlock."""
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

    def test_at_or_below_target_noop(self):
        """When estimated tokens <= clear target, no clearing occurs."""
        messages = []
        for i in range(15):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}" * 100)
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=999_999,
        )
        assert tokens_saved == 0

    def test_above_target_clears_no_floor(self):
        """Above the target, clearing has NO keep-recent floor: with a tiny
        target even the newest compactable results are cleared."""
        messages = []
        for i in range(15):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", f"output_{i}" * 100)
            messages.extend([a, u])

        new_msgs, tokens_saved = compaction_clear_tool_results(
            messages, clear_target_tokens=1,
        )
        assert tokens_saved > 0

        cleared_count = 0
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.content == CLEARED_MESSAGE:
                        cleared_count += 1
        assert cleared_count == 15  # all of them - no floor

    def test_oldest_first_stops_at_target(self):
        """Clearing is oldest-first and stops once the estimate reaches
        the target - newest results survive when the target allows."""
        messages = []
        for i in range(6):
            a, u = self._make_tool_exchange("Bash", f"tu_{i}", "content " * 300)
            messages.extend([a, u])

        total = estimate_message_tokens(messages)
        per_result = rough_token_count("content " * 300)
        target = total - int(2.5 * per_result)  # ~3 clears needed
        new_msgs, _ = compaction_clear_tool_results(
            messages, clear_target_tokens=target,
        )

        cleared = []
        preserved = []
        for msg in new_msgs:
            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        if block.content == CLEARED_MESSAGE:
                            cleared.append(block.tool_use_id)
                        else:
                            preserved.append(block.tool_use_id)

        assert 1 <= len(cleared) < 6
        # Cleared are exactly the oldest prefix
        assert cleared == [f"tu_{i}" for i in range(len(cleared))]
        # The newest survived
        assert "tu_5" in preserved


# ---------------------------------------------------------------------------
# truncate_middle_for_ptl tests
# ---------------------------------------------------------------------------


class TestTruncateMiddleForPtl:
    def test_too_few_messages_returns_none(self):
        """With <= 4 messages, returns None (nothing to cut)."""
        msgs = [create_user_message(f"msg {i}") for i in range(4)]
        result = truncate_middle_for_ptl(msgs)
        assert result is None

    def test_preserves_start_and_end(self):
        """Start and end messages are preserved."""
        msgs = [create_user_message(f"msg_{i}") for i in range(20)]
        result = truncate_middle_for_ptl(msgs)
        assert result is not None
        assert len(result) < len(msgs)
        # First message preserved
        assert result[0].content == "msg_0"
        # Last message preserved
        assert result[-1].content == "msg_19"

    def test_cuts_about_20_percent(self):
        """About 20% of messages are removed."""
        msgs = [create_user_message(f"msg_{i}") for i in range(20)]
        result = truncate_middle_for_ptl(msgs)
        assert result is not None
        # 20 messages, ~20% = 4 removed, so ~16 remaining
        assert len(result) == 16

    def test_middle_is_cut(self):
        """The middle portion is what gets removed."""
        msgs = [create_user_message(f"msg_{i}") for i in range(10)]
        result = truncate_middle_for_ptl(msgs)
        assert result is not None
        # With 10 messages, cut_size = 2, cut_start = 4, cut_end = 6
        # So messages 4 and 5 are removed
        contents = [m.content for m in result]
        assert "msg_0" in contents
        assert "msg_9" in contents
        assert "msg_4" not in contents
        assert "msg_5" not in contents


# ---------------------------------------------------------------------------
# format_compact_summary tests
# ---------------------------------------------------------------------------


class TestFormatCompactSummary:
    def test_extracts_summary_tags(self):
        """Extracts content from <summary> tags."""
        raw = "<analysis>thinking...</analysis>\n<summary>The actual summary.</summary>"
        result = format_compact_summary(raw)
        assert "The actual summary." in result
        assert "thinking..." not in result
        assert result.startswith("Summary:")

    def test_strips_analysis_block(self):
        """<analysis> block is removed from output."""
        raw = "<analysis>Long analysis here</analysis>\n<summary>Just the summary</summary>"
        result = format_compact_summary(raw)
        assert "Long analysis here" not in result
        assert "Just the summary" in result

    def test_fallback_no_summary_tags(self):
        """Without <summary> tags, uses the full response."""
        raw = "No tags here, just plain text summary."
        result = format_compact_summary(raw)
        assert "No tags here" in result

    def test_analysis_stripped_even_without_summary_tags(self):
        """<analysis> is stripped even when <summary> tags are missing."""
        raw = "<analysis>My thinking</analysis>\nThe rest of the output."
        result = format_compact_summary(raw)
        assert "My thinking" not in result
        assert "The rest of the output." in result

    def test_multiline_summary(self):
        """Multi-line summary content is preserved."""
        raw = (
            "<analysis>step 1\nstep 2</analysis>\n"
            "<summary>1. Section One\n   Details\n\n2. Section Two\n   More details</summary>"
        )
        result = format_compact_summary(raw)
        assert "Section One" in result
        assert "Section Two" in result


# ---------------------------------------------------------------------------
# get_compact_prompt tests
# ---------------------------------------------------------------------------


class TestGetCompactPrompt:
    def test_contains_all_9_sections(self):
        """The prompt contains all 9 summary sections."""
        prompt = get_compact_prompt()
        assert "1. Primary Request and Intent" in prompt
        assert "2. Key Technical Concepts" in prompt
        assert "3. Files and Code Sections" in prompt
        assert "4. Errors and fixes" in prompt
        assert "5. Problem Solving" in prompt
        assert "6. All user messages" in prompt
        assert "7. Pending Tasks" in prompt
        assert "8. Current Work" in prompt
        assert "9. Optional Next Step" in prompt

    def test_contains_no_tools_preamble(self):
        """The prompt starts with the no-tools preamble."""
        prompt = get_compact_prompt()
        assert "CRITICAL: Respond with TEXT ONLY" in prompt
        assert "Do NOT call any tools" in prompt

    def test_contains_no_tools_trailer(self):
        """The prompt ends with the no-tools reminder."""
        prompt = get_compact_prompt()
        assert "REMINDER: Do NOT call any tools" in prompt

    def test_contains_analysis_summary_structure(self):
        """The prompt describes the analysis + summary structure."""
        prompt = get_compact_prompt()
        assert "<analysis>" in prompt
        assert "<summary>" in prompt

    def test_custom_instructions_included(self):
        """Custom instructions are appended when provided."""
        prompt = get_compact_prompt("Focus on test results")
        assert "Focus on test results" in prompt
        assert "Additional Instructions:" in prompt

    def test_no_custom_instructions(self):
        """Without custom instructions, 'Additional Instructions' is absent."""
        prompt = get_compact_prompt()
        assert "Additional Instructions:" not in prompt

    def test_empty_custom_instructions_ignored(self):
        """Empty string custom instructions are ignored."""
        prompt = get_compact_prompt("  ")
        assert "Additional Instructions:" not in prompt


# ---------------------------------------------------------------------------
# get_post_compact_message / notification tests
# ---------------------------------------------------------------------------


class TestPostCompactMessage:
    def test_contains_summary(self):
        raw = "<summary>My summary content</summary>"
        msg = get_post_compact_message(raw)
        assert "My summary content" in msg

    def test_contains_continuation_instruction(self):
        msg = get_post_compact_message("summary text")
        assert "Continue the conversation" in msg
        assert "Resume directly" in msg

    def test_includes_transcript_path(self):
        msg = get_post_compact_message("summary", transcript_path="/path/to/transcript.jsonl")
        assert "/path/to/transcript.jsonl" in msg

    def test_no_transcript_path(self):
        msg = get_post_compact_message("summary")
        assert "transcript" not in msg.lower()


class TestPostCompactNotification:
    def test_memory_on(self):
        notification = get_post_compact_notification(memory_mode="on")
        assert "memory" in notification.lower()

    def test_memory_off(self):
        notification = get_post_compact_notification(memory_mode="off")
        assert "memory" not in notification.lower()

    def test_system_reminder_tags(self):
        notification = get_post_compact_notification()
        assert "<system-reminder>" in notification
        assert "</system-reminder>" in notification


# ---------------------------------------------------------------------------
# Circuit breaker (integration-level unit test)
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_max_consecutive_compact_failures_default(self):
        """The default for max_consecutive_compact_failures is 3."""
        from alancode.settings import SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["max_consecutive_compact_failures"] == 3

    def test_compaction_threshold_percent_default(self):
        """The default for compaction_threshold_percent is 80."""
        from alancode.settings import SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["compaction_threshold_percent"] == "auto"  # auto = 80

    def test_max_compact_ptl_retries_default(self):
        """The default for max_compact_ptl_retries is 3."""
        from alancode.settings import SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["max_compact_ptl_retries"] == 3
