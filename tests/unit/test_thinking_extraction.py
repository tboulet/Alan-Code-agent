"""Tests for thinking extraction in text_tool_parser.py."""

import pytest

from alancode.tools.text_tool_parser import (
    _extract_thinking,
    extract_tool_calls_from_text,
    ParseResult,
)


class TestExtractThinking:
    """Direct tests on the _extract_thinking helper."""

    def test_with_think_tags(self):
        text = "<think>I need to reason about this</think>The answer is 42."
        thinking, remaining = _extract_thinking(text)
        assert thinking == "I need to reason about this"
        assert remaining == "The answer is 42."

    def test_with_think_tags_multiline(self):
        text = "<think>\nLine one\nLine two\n</think>\nOutput here"
        thinking, remaining = _extract_thinking(text)
        assert thinking == "Line one\nLine two"
        assert remaining == "Output here"

    def test_only_closing_think_tag_glm_style(self):
        """GLM models often output reasoning text before </think> without opening tag."""
        text = "I should list the files first</think>Let me check."
        thinking, remaining = _extract_thinking(text)
        assert thinking == "I should list the files first"
        assert remaining == "Let me check."

    def test_only_closing_think_tag_empty_thinking(self):
        """</think> at the very start means empty thinking content."""
        text = "</think>The response"
        thinking, remaining = _extract_thinking(text)
        # Empty string before </think> -> stripped is "" -> returns None
        assert thinking is None
        assert remaining == "The response"

    def test_no_think_tags(self):
        text = "Just a normal response with no thinking."
        thinking, remaining = _extract_thinking(text)
        assert thinking is None
        assert remaining == "Just a normal response with no thinking."

    def test_empty_think_tags(self):
        text = "<think></think>Response"
        thinking, remaining = _extract_thinking(text)
        # Empty content inside tags -> stripped is "" -> returns None
        assert thinking is None
        assert remaining == "Response"

    def test_whitespace_only_think_tags(self):
        text = "<think>   \n  </think>Response"
        thinking, remaining = _extract_thinking(text)
        # Whitespace-only -> stripped is "" -> returns None
        assert thinking is None
        assert remaining == "Response"

    def test_think_tags_in_middle(self):
        text = "Before <think>reasoning</think> After"
        thinking, remaining = _extract_thinking(text)
        assert thinking == "reasoning"
        assert remaining == "Before  After"

    def test_empty_input(self):
        thinking, remaining = _extract_thinking("")
        assert thinking is None
        assert remaining == ""


class TestParseResultThinkingField:
    """ParseResult.thinking is populated by extract_tool_calls_from_text."""

    def test_thinking_field_set(self):
        result = ParseResult(
            tool_calls=[],
            cleaned_text="response",
            thinking="some reasoning",
        )
        assert result.thinking == "some reasoning"

    def test_thinking_field_none_by_default(self):
        result = ParseResult(tool_calls=[], cleaned_text="response")
        assert result.thinking is None


class TestExtractToolCallsThinking:
    """extract_tool_calls_from_text returns thinking alongside tool calls."""

    def test_thinking_with_hermes_tool_call(self):
        text = (
            "<think>I need to run a command</think>"
            '<tool_call>\n'
            '{"name": "Bash", "arguments": {"command": "ls"}}\n'
            '</tool_call>'
        )
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 1
        assert result.thinking == "I need to run a command"
        assert "<think>" not in result.cleaned_text
        assert "</think>" not in result.cleaned_text

    def test_thinking_with_glm_tool_call(self):
        """GLM style: reasoning before </think>, then tool call."""
        text = (
            "I should list the directory.</think>"
            "<tool_call>Bash<arg_key>command</arg_key>"
            "<arg_value>ls -la</arg_value></tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 1
        assert result.thinking == "I should list the directory."
        assert "</think>" not in result.cleaned_text

    def test_thinking_without_tool_calls(self):
        """Model thinks but doesn't call a tool."""
        text = (
            "<think>Let me consider this carefully.</think>"
            "The answer is straightforward: 42."
        )
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 0
        assert result.thinking == "Let me consider this carefully."
        assert result.cleaned_text == "The answer is straightforward: 42."
        assert result.error is None

    def test_no_thinking_no_tool_calls(self):
        """Plain text: no thinking, no tool calls."""
        text = "Here is my answer."
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 0
        assert result.thinking is None
        assert result.cleaned_text == "Here is my answer."
        assert result.error is None

    def test_thinking_with_alan_tool_call(self):
        text = (
            "<think>Need to read the file</think>"
            '<tool_use>{"name": "Read", "input": {"file_path": "/tmp/x.py"}}</tool_use>'
        )
        result = extract_tool_calls_from_text(text, format="alan")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Read"
        assert result.thinking == "Need to read the file"

    def test_glm_thinking_only_closing_tag_no_tool(self):
        """GLM outputs reasoning then </think> but no tool call."""
        text = "This is my reasoning about the problem.</think>Here is the answer."
        result = extract_tool_calls_from_text(text, format="glm")
        assert len(result.tool_calls) == 0
        assert result.thinking == "This is my reasoning about the problem."
        assert result.cleaned_text == "Here is the answer."

    def test_thinking_with_malformed_tool_call(self):
        """Thinking is still extracted even when tool call is malformed."""
        text = (
            "<think>I should run a command</think>"
            "<tool_call>some broken stuff</tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="hermes")
        assert len(result.tool_calls) == 0
        assert result.error is not None
        assert result.thinking == "I should run a command"
