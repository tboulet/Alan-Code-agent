"""Test message types, factory functions, and normalization."""

import pytest
from uuid import uuid4

from alancode.messages.types import (
    AssistantMessage,
    AttachmentMessage,
    Attachment,
    CompactMetadata,
    ProgressMessage,
    SystemMessage,
    SystemMessageSubtype,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
    is_compact_boundary,
    get_messages_after_compact_boundary,
    get_last_assistant_message,
)
from alancode.messages.factory import (
    INTERRUPT_MESSAGE,
    INTERRUPT_MESSAGE_FOR_TOOL_USE,
    SYNTHETIC_MODEL,
    create_assistant_error_message,
    create_assistant_message,
    create_compact_boundary_message,
    create_tool_result_message,
    create_user_interruption_message,
    create_user_message,
    create_system_message,
    create_attachment_message,
)
from alancode.messages.normalization import (
    normalize_messages_for_api,
    merge_user_messages,
    get_text_content,
)


# ---------------------------------------------------------------------------
# Message factory tests
# ---------------------------------------------------------------------------


class TestMessageFactory:
    def test_create_user_message(self):
        msg = create_user_message("Hello")
        assert isinstance(msg, UserMessage)
        assert msg.content == "Hello"
        assert msg.type == "user"
        assert msg.hide_in_ui is False
        assert msg.hide_in_api is False

    def test_create_user_message_meta(self):
        msg = create_user_message("system context", hide_in_ui=True)
        assert msg.hide_in_ui is True
        assert msg.content == "system context"

    def test_create_user_message_virtual(self):
        msg = create_user_message("display only", hide_in_api=True)
        assert msg.hide_in_api is True

    def test_create_assistant_message_from_string(self):
        msg = create_assistant_message("I can help!")
        assert isinstance(msg, AssistantMessage)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "I can help!"
        assert msg.text == "I can help!"

    def test_create_assistant_message_from_blocks(self):
        blocks = [TextBlock(text="Part 1"), TextBlock(text="Part 2")]
        msg = create_assistant_message(blocks)
        assert len(msg.content) == 2
        assert msg.text == "Part 1Part 2"

    def test_create_assistant_error_message(self):
        msg = create_assistant_error_message(
            "Something went wrong",
            api_error="invalid_request",
            error_details="bad input",
        )
        assert msg.is_api_error_message is True
        assert msg.model == SYNTHETIC_MODEL
        assert msg.api_error == "invalid_request"
        assert msg.error_details == "bad input"
        assert msg.text == "Something went wrong"

    def test_create_user_interruption_message(self):
        msg = create_user_interruption_message()
        assert isinstance(msg, UserMessage)
        assert msg.content == INTERRUPT_MESSAGE

    def test_create_user_interruption_message_tool_use(self):
        msg = create_user_interruption_message(tool_use=True)
        assert msg.content == INTERRUPT_MESSAGE_FOR_TOOL_USE

    def test_create_tool_result_message(self):
        msg = create_tool_result_message(
            tool_use_id="tu_123",
            content="result data",
        )
        assert isinstance(msg, UserMessage)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        block = msg.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.tool_use_id == "tu_123"
        assert block.content == "result data"
        assert block.is_error is False

    def test_create_tool_result_message_error(self):
        msg = create_tool_result_message(
            tool_use_id="tu_456",
            content="error occurred",
            is_error=True,
        )
        block = msg.content[0]
        assert block.is_error is True

    def test_create_compact_boundary_message(self):
        msg = create_compact_boundary_message(
            trigger="auto",
            pre_tokens=50000,
            messages_summarized=20,
        )
        assert isinstance(msg, SystemMessage)
        assert msg.subtype == SystemMessageSubtype.COMPACT_BOUNDARY
        assert msg.compact_metadata is not None
        assert msg.compact_metadata.pre_tokens == 50000
        assert msg.compact_metadata.messages_summarized == 20

    def test_create_system_message(self):
        msg = create_system_message("info text", level="warning")
        assert isinstance(msg, SystemMessage)
        assert msg.subtype == SystemMessageSubtype.INFORMATIONAL
        assert msg.level == "warning"
        assert msg.content == "info text"

    def test_create_attachment_message(self):
        msg = create_attachment_message(
            "edited_text_file",
            content="file contents here",
            metadata={"path": "/tmp/foo.py"},
        )
        assert isinstance(msg, AttachmentMessage)
        assert msg.attachment.type == "edited_text_file"
        assert msg.attachment.content == "file contents here"


# ---------------------------------------------------------------------------
# Message type helpers
# ---------------------------------------------------------------------------


class TestMessageTypeHelpers:
    def test_assistant_message_text_property(self):
        msg = AssistantMessage(content=[TextBlock(text="A"), TextBlock(text="B")])
        assert msg.text == "AB"

    def test_assistant_message_tool_use_blocks(self):
        tu = ToolUseBlock(id="t1", name="Bash", input={"command": "ls"})
        msg = AssistantMessage(content=[TextBlock(text="ok"), tu])
        assert msg.tool_use_blocks == [tu]
        assert msg.has_tool_use is True

    def test_assistant_message_no_tool_use(self):
        msg = AssistantMessage(content=[TextBlock(text="hello")])
        assert msg.has_tool_use is False
        assert msg.tool_use_blocks == []

    def test_usage_accumulate(self):
        u1 = Usage(input_tokens=100, output_tokens=50)
        u2 = Usage(input_tokens=200, output_tokens=100, cache_read_input_tokens=10)
        u1.accumulate(u2)
        assert u1.input_tokens == 300
        assert u1.output_tokens == 150
        assert u1.cache_read_input_tokens == 10

    def test_usage_total_input(self):
        u = Usage(input_tokens=100, cache_creation_input_tokens=20, cache_read_input_tokens=30)
        assert u.total_input == 150


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_filters_virtual_user_messages(self):
        messages = [
            create_user_message("real"),
            create_user_message("ghost", hide_in_api=True),
            create_assistant_message("response"),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)

    def test_filters_virtual_assistant_messages(self):
        messages = [
            create_user_message("hi"),
            create_assistant_message("real response"),
            AssistantMessage(content=[TextBlock(text="streaming delta")], hide_in_api=True),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 2

    def test_filters_progress_messages(self):
        messages = [
            create_user_message("hi"),
            ProgressMessage(tool_use_id="t1", data={"status": "running"}),
            create_assistant_message("done"),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 2

    def test_filters_system_messages(self):
        messages = [
            create_user_message("hi"),
            create_system_message("info"),
            create_assistant_message("response"),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 2

    def test_merges_consecutive_user_messages(self):
        messages = [
            create_user_message("first"),
            create_user_message("second"),
            create_assistant_message("response"),
        ]
        result = normalize_messages_for_api(messages)
        # Two user messages should be merged into one
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        text = get_text_content(result[0])
        assert "first" in text
        assert "second" in text

    def test_preserves_alternating_messages(self):
        messages = [
            create_user_message("q1"),
            create_assistant_message("a1"),
            create_user_message("q2"),
            create_assistant_message("a2"),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 4
        assert isinstance(result[0], UserMessage)
        assert isinstance(result[1], AssistantMessage)
        assert isinstance(result[2], UserMessage)
        assert isinstance(result[3], AssistantMessage)

    def test_converts_local_command_to_user(self):
        messages = [
            SystemMessage(
                content="/compact output",
                subtype=SystemMessageSubtype.LOCAL_COMMAND,
            ),
            create_assistant_message("compacted"),
        ]
        result = normalize_messages_for_api(messages)
        assert len(result) == 2
        assert isinstance(result[0], UserMessage)
        assert result[0].hide_in_ui is True

    def test_converts_attachment_to_user(self):
        messages = [
            create_user_message("hi"),
            create_attachment_message("edited_text_file", content="file data"),
            create_assistant_message("response"),
        ]
        result = normalize_messages_for_api(messages)
        # Attachment becomes user message; may merge with the first user message
        # The result should be alternating user/assistant
        user_msgs = [m for m in result if isinstance(m, UserMessage)]
        assert len(user_msgs) >= 1


# ---------------------------------------------------------------------------
# Compact boundary helpers
# ---------------------------------------------------------------------------


class TestCompactBoundary:
    def test_is_compact_boundary(self):
        boundary = create_compact_boundary_message("auto", pre_tokens=5000)
        assert is_compact_boundary(boundary) is True

    def test_is_not_compact_boundary(self):
        msg = create_user_message("hi")
        assert is_compact_boundary(msg) is False
        info = create_system_message("info")
        assert is_compact_boundary(info) is False

    def test_get_messages_after_boundary(self):
        msgs = [
            create_user_message("old"),
            create_assistant_message("old response"),
            create_compact_boundary_message("auto", pre_tokens=5000),
            create_user_message("new"),
            create_assistant_message("new response"),
        ]
        after = get_messages_after_compact_boundary(msgs)
        assert len(after) == 3  # boundary + 2 new messages
        assert is_compact_boundary(after[0])

    def test_get_messages_no_boundary(self):
        msgs = [
            create_user_message("q1"),
            create_assistant_message("a1"),
        ]
        after = get_messages_after_compact_boundary(msgs)
        assert len(after) == 2  # returns all messages

    def test_get_last_assistant_message(self):
        msgs = [
            create_user_message("q"),
            create_assistant_message("a1"),
            create_user_message("q2"),
            create_assistant_message("a2"),
        ]
        last = get_last_assistant_message(msgs)
        assert last is not None
        assert last.text == "a2"

    def test_get_last_assistant_message_none(self):
        msgs = [create_user_message("q")]
        assert get_last_assistant_message(msgs) is None
