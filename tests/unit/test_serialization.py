"""Test message serialization (alancode/messages/serialization.py)."""

import pytest

from alancode.messages.serialization import block_to_anthropic_dict as block_to_dict, message_to_anthropic_dict as message_to_api_dict, messages_to_openai_dicts
from alancode.messages.types import (
    AssistantMessage,
    ImageBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserMessage,
)
from alancode.messages.factory import create_user_message, create_assistant_message


# ---------------------------------------------------------------------------
# block_to_dict
# ---------------------------------------------------------------------------


class TestBlockToDict:

    def test_text_block(self):
        block = TextBlock(text="hello world")
        d = block_to_dict(block)
        assert d == {"type": "text", "text": "hello world"}

    def test_text_block_empty(self):
        block = TextBlock(text="")
        d = block_to_dict(block)
        assert d == {"type": "text", "text": ""}

    def test_tool_use_block(self):
        block = ToolUseBlock(
            id="toolu_123",
            name="Bash",
            input={"command": "ls -la"},
        )
        d = block_to_dict(block)
        assert d == {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "Bash",
            "input": {"command": "ls -la"},
        }

    def test_tool_use_block_empty_input(self):
        block = ToolUseBlock(id="toolu_456", name="ListFiles", input={})
        d = block_to_dict(block)
        assert d["type"] == "tool_use"
        assert d["input"] == {}

    def test_tool_result_block_string_content(self):
        block = ToolResultBlock(
            tool_use_id="toolu_123",
            content="file1.txt\nfile2.txt",
            is_error=False,
        )
        d = block_to_dict(block)
        assert d == {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "file1.txt\nfile2.txt",
            "is_error": False,
        }

    def test_tool_result_block_error(self):
        block = ToolResultBlock(
            tool_use_id="toolu_789",
            content="Command failed",
            is_error=True,
        )
        d = block_to_dict(block)
        assert d["is_error"] is True

    def test_tool_result_block_list_content(self):
        """When content is a list of TextBlocks, they should be recursively serialized."""
        inner_blocks = [TextBlock(text="line 1"), TextBlock(text="line 2")]
        block = ToolResultBlock(
            tool_use_id="toolu_abc",
            content=inner_blocks,
            is_error=False,
        )
        d = block_to_dict(block)
        assert d["type"] == "tool_result"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        assert d["content"][0] == {"type": "text", "text": "line 1"}
        assert d["content"][1] == {"type": "text", "text": "line 2"}

    def test_thinking_block_with_signature(self):
        block = ThinkingBlock(thinking="Let me think...", signature="sig_abc")
        d = block_to_dict(block)
        assert d == {
            "type": "thinking",
            "thinking": "Let me think...",
            "signature": "sig_abc",
        }

    def test_thinking_block_without_signature(self):
        block = ThinkingBlock(thinking="reasoning here", signature="")
        d = block_to_dict(block)
        assert d == {"type": "thinking", "thinking": "reasoning here"}
        assert "signature" not in d

    def test_redacted_thinking_block(self):
        block = RedactedThinkingBlock(data="base64encodeddata")
        d = block_to_dict(block)
        assert d == {"type": "redacted_thinking", "data": "base64encodeddata"}

    def test_image_block(self):
        source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBOR...",
        }
        block = ImageBlock(source=source)
        d = block_to_dict(block)
        assert d == {"type": "image", "source": source}

    def test_unknown_block_type_fallback(self):
        """A block type not handled by block_to_dict returns {"type": "unknown"}."""
        # Use a plain object that doesn't match any known type
        class FakeBlock:
            pass

        d = block_to_dict(FakeBlock())
        assert d == {"type": "unknown"}


# ---------------------------------------------------------------------------
# message_to_api_dict — UserMessage
# ---------------------------------------------------------------------------


class TestMessageToApiDictUser:

    def test_user_message_string_content(self):
        msg = create_user_message("Hello, assistant!")
        d = message_to_api_dict(msg)
        assert d == {"role": "user", "content": "Hello, assistant!"}

    def test_user_message_list_content(self):
        blocks = [
            TextBlock(text="Here's my question"),
            ToolResultBlock(tool_use_id="toolu_1", content="result data"),
        ]
        msg = create_user_message(blocks)
        d = message_to_api_dict(msg)
        assert d["role"] == "user"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 2
        assert d["content"][0]["type"] == "text"
        assert d["content"][1]["type"] == "tool_result"

    def test_user_message_empty_string(self):
        msg = create_user_message("")
        d = message_to_api_dict(msg)
        assert d == {"role": "user", "content": ""}


# ---------------------------------------------------------------------------
# message_to_api_dict — AssistantMessage
# ---------------------------------------------------------------------------


class TestMessageToApiDictAssistant:

    def test_assistant_message_text_only(self):
        msg = create_assistant_message("Here is my answer.")
        d = message_to_api_dict(msg)
        assert d["role"] == "assistant"
        assert isinstance(d["content"], list)
        assert len(d["content"]) == 1
        assert d["content"][0] == {"type": "text", "text": "Here is my answer."}

    def test_assistant_message_with_tool_use(self):
        blocks = [
            TextBlock(text="Let me check..."),
            ToolUseBlock(id="toolu_x", name="Bash", input={"command": "ls"}),
        ]
        msg = create_assistant_message(blocks)
        d = message_to_api_dict(msg)
        assert d["role"] == "assistant"
        assert len(d["content"]) == 2
        assert d["content"][0]["type"] == "text"
        assert d["content"][1]["type"] == "tool_use"
        assert d["content"][1]["name"] == "Bash"

    def test_assistant_message_with_thinking(self):
        blocks = [
            ThinkingBlock(thinking="I need to consider...", signature="sig_1"),
            TextBlock(text="My answer is..."),
        ]
        msg = create_assistant_message(blocks)
        d = message_to_api_dict(msg)
        assert d["role"] == "assistant"
        assert d["content"][0]["type"] == "thinking"
        assert d["content"][1]["type"] == "text"

    def test_assistant_message_empty_content_list(self):
        msg = AssistantMessage(content=[])
        d = message_to_api_dict(msg)
        assert d == {"role": "assistant", "content": []}


# ---------------------------------------------------------------------------
# messages_to_openai_dicts — OpenAI format
# ---------------------------------------------------------------------------


class TestOpenAIFormat:

    def test_user_string_message(self):
        msg = create_user_message("Hello")
        result = messages_to_openai_dicts([msg])
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_text_only(self):
        msg = create_assistant_message("My answer")
        result = messages_to_openai_dicts([msg])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "My answer"
        assert "tool_calls" not in result[0]

    def test_assistant_with_tool_calls(self):
        blocks = [
            TextBlock(text="Let me check."),
            ToolUseBlock(id="call_1", name="Bash", input={"command": "ls"}),
        ]
        msg = AssistantMessage(content=blocks)
        result = messages_to_openai_dicts([msg])
        assert len(result) == 1
        d = result[0]
        assert d["role"] == "assistant"
        assert d["content"] == "Let me check."
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["type"] == "function"
        assert d["tool_calls"][0]["function"]["name"] == "Bash"

    def test_user_with_tool_results(self):
        """UserMessage with tool_result blocks splits into role=tool messages."""
        blocks = [
            ToolResultBlock(tool_use_id="call_1", content="file1.txt\nfile2.txt", is_error=False),
        ]
        msg = UserMessage(content=blocks)
        result = messages_to_openai_dicts([msg])
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert "file1.txt" in result[0]["content"]

    def test_user_with_tool_results_and_text(self):
        """Tool results + text produce multiple messages."""
        blocks = [
            ToolResultBlock(tool_use_id="call_1", content="output", is_error=False),
            TextBlock(text="Additional context"),
        ]
        msg = UserMessage(content=blocks)
        result = messages_to_openai_dicts([msg])
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Additional context"

    def test_multiple_tool_results(self):
        blocks = [
            ToolResultBlock(tool_use_id="call_1", content="out1", is_error=False),
            ToolResultBlock(tool_use_id="call_2", content="out2", is_error=False),
        ]
        msg = UserMessage(content=blocks)
        result = messages_to_openai_dicts([msg])
        assert len(result) == 2
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_2"

    def test_full_conversation(self):
        """Complete conversation: user → assistant+tool_use → tool_result → assistant."""
        msgs = [
            create_user_message("List files"),
            AssistantMessage(content=[
                TextBlock(text="I'll list them."),
                ToolUseBlock(id="call_1", name="Bash", input={"command": "ls"}),
            ]),
            UserMessage(content=[
                ToolResultBlock(tool_use_id="call_1", content="file1\nfile2", is_error=False),
            ]),
            create_assistant_message("Here are the files: file1, file2"),
        ]
        result = messages_to_openai_dicts(msgs)
        assert result[0] == {"role": "user", "content": "List files"}
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_calls"][0]["function"]["name"] == "Bash"
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_1"
        assert result[3]["role"] == "assistant"
        assert result[3]["content"] == "Here are the files: file1, file2"

    def test_assistant_tool_only_no_text(self):
        """Assistant with only tool calls, no text."""
        blocks = [ToolUseBlock(id="call_1", name="Read", input={"file_path": "x.py"})]
        msg = AssistantMessage(content=blocks)
        result = messages_to_openai_dicts([msg])
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
