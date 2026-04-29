"""Tests for text-based tool call parser."""

import pytest

from alancode.tools.text_tool_parser import (
    extract_tool_calls_from_text,
    get_tool_format_system_prompt,
    get_format,
    FORMATS,
)


class TestGLMFormat:
    """GLM-4 text tool call format."""

    def test_single_tool_call(self):
        text = (
            "I'll list the files.</think>"
            "<tool_call>Bash<arg_key>command</arg_key>"
            "<arg_value>ls -la /tmp</arg_value></tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la /tmp"}
        assert "</think>" not in result.cleaned_text
        assert "<tool_call>" not in result.cleaned_text
        assert result.error is None

    def test_multiple_args(self):
        text = (
            "<tool_call>Edit"
            "<arg_key>file_path</arg_key><arg_value>/tmp/test.py</arg_value>"
            "<arg_key>old_string</arg_key><arg_value>def foo():</arg_value>"
            "<arg_key>new_string</arg_key><arg_value>def bar():</arg_value>"
            "</tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Edit"
        assert result.tool_calls[0].input == {
            "file_path": "/tmp/test.py",
            "old_string": "def foo():",
            "new_string": "def bar():",
        }

    def test_missing_closing_tag_not_executed(self):
        """Missing </tool_call> must NOT parse as a complete call.

        During streaming, partial content arrives without the closing
        tag yet — if we accepted these, mid-stream fragments would
        execute tools with truncated arguments. The parser must wait
        for the closing tag.

        Malformed detection also requires both opening AND closing tags;
        a bare <tool_call> in prose (e.g. when the model quotes the tag
        in an apology) must not trigger a retry — that caused a
        self-perpetuating loop where the error message itself, containing
        <tool_call>, kept getting echoed back.
        """
        text = (
            "<tool_call>Bash<arg_key>command</arg_key>"
            "<arg_value>ls /tmp</arg_value>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 0
        assert result.error is None

    def test_no_tool_call(self):
        text = "Just a regular response with no tool calls."
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 0
        assert result.cleaned_text == text
        assert result.error is None

    def test_real_glm_output(self):
        """Actual GLM-4.7-FP8 output sample."""
        text = (
            "The user wants me to list the files in /tmp using the bash tool. "
            "This is a straightforward request.</think>"
            "<tool_call>Bash<arg_key>command</arg_key>"
            "<arg_value>ls -la /tmp</arg_value></tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.error is None

    def test_malformed_glm_output(self):
        """GLM outputs a wrong XML variant with both <tool_call> and </tool_call>."""
        text = (
            "I'll check the files.</think>"
            "<tool_call>Bash command='ls -la'</tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 0
        assert result.error is not None
        assert "format" in result.error.lower()
        assert "<tool_call>ToolName" in result.error  # Shows expected format

    def test_malformed_no_arg_tags(self):
        """GLM outputs tool_call but without arg_key/arg_value tags."""
        text = '<tool_call>Bash {"command": "ls"}</tool_call>'
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 0
        assert result.error is not None


class TestHermesFormat:
    """Hermes/Qwen 2.5 text tool call format."""

    def test_single_tool_call(self):
        text = (
            '<tool_call>\n'
            '{"name": "Bash", "arguments": {"command": "ls /tmp"}}\n'
            '</tool_call>'
        )
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls /tmp"}
        assert result.error is None

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n'
            '{"name": "Read", "arguments": {"file_path": "/tmp/a.txt"}}\n'
            '</tool_call>\n'
            '<tool_call>\n'
            '{"name": "Read", "arguments": {"file_path": "/tmp/b.txt"}}\n'
            '</tool_call>'
        )
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 2

    def test_no_tool_call(self):
        text = "Here is the answer to your question."
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 0
        assert result.error is None

    def test_malformed_json(self):
        text = "<tool_call>\nnot valid json at all\n</tool_call>"
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 0
        assert result.error is not None
        assert "not valid" in result.error


class TestAlanFormat:
    """Alan's custom text tool call format."""

    def test_single_tool_call(self):
        text = (
            '<tool_use>\n'
            '{"name": "Bash", "input": {"command": "ls"}}\n'
            '</tool_use>'
        )
        result = extract_tool_calls_from_text(text, format="alan")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.error is None

    def test_multiple_tool_calls(self):
        text = (
            'Let me check both files.\n'
            '<tool_use>{"name": "Read", "input": {"file_path": "a.py"}}</tool_use>\n'
            '<tool_use>{"name": "Read", "input": {"file_path": "b.py"}}</tool_use>'
        )
        result = extract_tool_calls_from_text(text, format="alan")
        assert len(result.tool_calls) == 2
        assert "Let me check" in result.cleaned_text


class TestThinkingStrip:
    """The </think> tag should be stripped from cleaned text."""

    def test_thinking_stripped(self):
        text = "Some reasoning here</think>The actual response"
        result = extract_tool_calls_from_text(text, format="hermes")
        assert result.cleaned_text == "The actual response"

    def test_no_thinking(self):
        text = "Just a normal response"
        result = extract_tool_calls_from_text(text, format="hermes")
        assert result.cleaned_text == "Just a normal response"


class TestMalformedDetection:
    """Test that malformed tool calls produce actionable error messages."""

    def test_malformed_error_contains_expected_format(self):
        """Error message should show the correct format."""
        text = "<tool_call>some garbage here</tool_call>"
        result = extract_tool_calls_from_text(text, format="hermes")
        assert result.error is not None
        assert "Expected format" in result.error

    def test_no_error_when_no_tool_attempt(self):
        """Normal text with no tool tags should not trigger error."""
        text = "Just a regular answer about tool usage in general."
        result = extract_tool_calls_from_text(text, format="glm")
        assert result.error is None

    def test_glm_error_shows_arg_key_format(self):
        """GLM error should show the arg_key/arg_value format."""
        text = "<tool_call>Bash(command='ls')</tool_call>"
        result = extract_tool_calls_from_text(text, format="glm")
        assert result.error is not None
        assert "<arg_key>" in result.error

    def test_bare_tool_call_tag_in_prose_not_flagged(self):
        """A lone <tool_call> mentioned in prose must not trigger an error.

        Regression: when the model apologized and quoted the tag literally
        (e.g. "I will use <tool_call> tags correctly"), the loose detector
        fired, and the resulting error message — which itself contains
        <tool_call> — got quoted again next turn, causing a retry loop.
        """
        text = "Sorry, I should have used <tool_call> tags. I'll retry."
        for fmt in ("hermes", "glm", "alan"):
            tag = "<tool_use>" if fmt == "alan" else "<tool_call>"
            sample = f"Sorry, I should have used {tag} tags. I'll retry."
            result = extract_tool_calls_from_text(sample, format=fmt)
            assert result.error is None, f"{fmt} flagged a bare {tag} mention"

    def test_error_message_does_not_echo_model_output(self):
        """The error message must not include the model's own text.

        Echoing it back confused the model about where its message ended
        and the tool feedback began.
        """
        garbage = "this is the model's bogus tool call attempt"
        text = f"<tool_call>{garbage}</tool_call>"
        result = extract_tool_calls_from_text(text, format="hermes")
        assert result.error is not None
        assert garbage not in result.error
        assert "Example:" in result.error


class TestSystemPrompt:
    """get_tool_format_system_prompt generates format instructions."""

    def test_hermes_prompt(self):
        schemas = [{"type": "function", "function": {"name": "Bash", "description": "Run command", "parameters": {}}}]
        prompt = get_tool_format_system_prompt("hermes", schemas)
        assert "<tool_call>" in prompt
        assert "Bash" in prompt

    def test_glm_prompt(self):
        schemas = [{"type": "function", "function": {"name": "Bash", "description": "Run command", "parameters": {}}}]
        prompt = get_tool_format_system_prompt("glm", schemas)
        assert "Bash" in prompt
        assert "<arg_key>" in prompt

    def test_alan_prompt(self):
        schemas = [{"type": "function", "function": {"name": "Bash", "description": "Run command", "parameters": {}}}]
        prompt = get_tool_format_system_prompt("alan", schemas)
        assert "<tool_use>" in prompt

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_tool_format_system_prompt("unknown", [])


class TestFormatRegistry:
    """Test the format class registry."""

    def test_all_formats_registered(self):
        assert "hermes" in FORMATS
        assert "glm" in FORMATS
        assert "alan" in FORMATS

    def test_get_format_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_format("unknown")

    def test_extract_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            extract_tool_calls_from_text("text", format="unknown")
