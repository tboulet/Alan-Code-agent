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


class TestHermesXMLFormat:
    """Hermes-XML (Qwen3-Coder-Next style) text tool call format."""

    def test_single_tool_call(self):
        text = (
            '<tool_call>\n'
            '<function=Bash>\n'
            '<parameter=command>ls /tmp</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
        result = extract_tool_calls_from_text(text, format="hermes_xml")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls /tmp"}
        assert result.error is None

    def test_missing_closing_tool_call_tag_parses(self):
        # Qwen2.5-72B stops generation right after </function>; the call
        # must still parse without the trailing </tool_call>.
        text = (
            'I will inspect the directory.\n'
            '<tool_call>\n'
            '<function=Bash>\n'
            '<parameter=command>ls /tmp</parameter>\n'
            '<parameter=purpose>list files</parameter>\n'
            '</function>'
        )
        result = extract_tool_calls_from_text(text, format="hermes_xml")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input["command"] == "ls /tmp"
        assert result.error is None

    def test_multiline_raw_body_preserved(self):
        body = 'cat > hello.py <<EOF\nprint("hi")\nEOF\npython3 hello.py'
        text = (
            '<tool_call>\n'
            '<function=Bash>\n'
            f'<parameter=command>\n{body}\n</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
        result = extract_tool_calls_from_text(text, format="hermes_xml")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input["command"] == body


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


class TestBashBlockFormat:
    """SWE-agent style fenced ```bash block format."""

    def test_single_block(self):
        text = "I will list the files first.\n\n```bash\nls -la /tmp\n```"
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la /tmp"}
        assert "```bash" not in result.cleaned_text
        assert "I will list the files first." in result.cleaned_text
        assert result.error is None

    def test_multiline_heredoc_preserved(self):
        body = "cat > hello.py <<'EOF'\nprint(\"hi\")\nEOF\npython3 hello.py"
        text = f"Writing the file now.\n```bash\n{body}\n```"
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input["command"] == body

    def test_multiple_blocks_first_only(self):
        text = (
            "First I check, then I run.\n"
            "```bash\necho one\n```\n"
            "and then:\n"
            "```bash\necho two\n```"
        )
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {"command": "echo one"}
        assert result.error is None

    def test_no_block_is_normal_text(self):
        text = "Let me think about the layout of the grid first."
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert result.tool_calls == []
        assert result.error is None
        assert result.cleaned_text == text

    def test_unclosed_block_not_executed(self):
        """A block cut before its closing fence (mid-stream chunk or a
        length-truncated output) must NOT parse as a call - the loop's
        truncation recovery handles the truncated case."""
        text = "Writing:\n```bash\ncat > f <<'EOF'\nif ("
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert result.tool_calls == []
        assert result.error is None

    def test_other_language_fences_ignored(self):
        text = "```python\nprint('hi')\n```\nand\n```sh\nls\n```"
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert result.tool_calls == []
        assert result.error is None

    def test_inline_fence_mention_not_matched(self):
        text = "Use a ```bash block to act, closing it with ``` as usual."
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert result.tool_calls == []
        assert result.error is None

    def test_parse_thinking_disabled(self):
        assert get_format("bash_block").parse_thinking is False
        assert get_format("hermes").parse_thinking is True

    def test_inline_opening_fence_glued_to_prose(self):
        # GLM-5.2 emits the opening fence glued to its prose line.
        text = "Let me look at the files first.```bash\ncat solution.py\n```"
        result = extract_tool_calls_from_text(text, format="bash_block")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {"command": "cat solution.py"}

    def test_crlf_line_endings_parse(self):
        # GLM-5.2 intermittently emits CRLF; both fences must still match.
        text = (
            "Label.<tool_call>Bash Output:\r\n"
            "```bash\r\n"
            "cat > f.py <<EOF\r\n"
            "print(1)\r\n"
            "EOF\r\n"
            "python f.py\r\n"
            "```"
        )
        for fmt in ("bash_block", "auto"):
            result = extract_tool_calls_from_text(text, format=fmt)
            assert len(result.tool_calls) == 1, fmt
            assert result.tool_calls[0].name == "Bash"


class TestKimiFormat:
    """Kimi K2-family special-token format."""

    def test_single_call(self):
        text = (
            "I'll list the files.\n"
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.Bash:0"
            '<|tool_call_argument_begin|>{"command": "ls -la"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result = extract_tool_calls_from_text(text, format="kimi")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la"}
        assert "<|tool_call_begin|>" not in result.cleaned_text
        assert result.error is None

    def test_tool_id_shapes(self):
        for raw_id in ("functions.Bash:0", "Bash:2", "Bash"):
            text = (
                f"<|tool_call_begin|>{raw_id}"
                '<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>'
            )
            result = extract_tool_calls_from_text(text, format="kimi")
            assert len(result.tool_calls) == 1, raw_id
            assert result.tool_calls[0].name == "Bash", raw_id

    def test_invalid_json_args_flagged(self):
        text = (
            "<|tool_call_begin|>functions.Bash:0"
            "<|tool_call_argument_begin|>{command: ls}<|tool_call_end|>"
        )
        result = extract_tool_calls_from_text(text, format="kimi")
        assert result.tool_calls == []
        assert result.error is not None
        assert "JSON" in result.error

    def test_prose_without_tokens_clean(self):
        result = extract_tool_calls_from_text("Just reasoning.", format="kimi")
        assert result.tool_calls == []
        assert result.error is None

    def test_system_prompt(self):
        prompt = get_tool_format_system_prompt("kimi", [])
        assert "<|tool_call_begin|>" in prompt

    def test_auto_detects_kimi(self):
        text = (
            "<|tool_call_begin|>functions.Bash:0"
            '<|tool_call_argument_begin|>{"command": "ls"}<|tool_call_end|>'
        )
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"

    def test_opaque_function_id_kept_verbatim(self):
        """K2.7 emits opaque ids (text_de60e4f6) - the parser keeps them;
        the loop's single-tool remap resolves them to the real tool."""
        text = (
            "Probing the env first."
            "<|tool_calls_section_begin|><|tool_call_begin|>text_de60e4f6"
            '<|tool_call_argument_begin|>{"command": "python explore.py"}'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )
        result = extract_tool_calls_from_text(text, format="kimi")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "text_de60e4f6"
        assert result.tool_calls[0].input == {"command": "python explore.py"}


class TestKimiK3Format:
    """Kimi K3 structured-token format."""

    def test_live_sample(self):
        # The verbatim K3 turn shape from the Adastra serve.
        text = (
            '<|open|>tools<|sep|><|open|>call tool="bash" index="1"<|sep|>'
            '<|open|>argument key="command" type="string"<|sep|>'
            "cat framework/make_env.py && ls code_library"
            "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
            "<|close|>message<|sep|><|end_of_msg|>"
        )
        result = extract_tool_calls_from_text(text, format="kimi_k3")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].input == {
            "command": "cat framework/make_env.py && ls code_library",
        }
        assert result.error is None

    def test_multiline_heredoc_value(self):
        command = "cat > f.py <<'EOF'\nprint(1)\nEOF\npython f.py"
        text = (
            '<|open|>call tool="bash" index="1"<|sep|>'
            f'<|open|>argument key="command" type="string"<|sep|>{command}'
            "<|close|>argument<|sep|><|close|>call"
        )
        result = extract_tool_calls_from_text(text, format="kimi_k3")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input["command"] == command

    def test_malformed_flagged(self):
        text = '<|open|>call tool="bash"<|sep|>no argument structure'
        result = extract_tool_calls_from_text(text, format="kimi_k3")
        assert result.tool_calls == []
        assert result.error is not None

    def test_prose_clean(self):
        result = extract_tool_calls_from_text("Just prose.", format="kimi_k3")
        assert result.tool_calls == []
        assert result.error is None

    def test_repair_closes_call(self):
        fmt = get_format("kimi_k3")
        text = (
            '<|open|>call tool="bash" index="1"<|sep|>'
            '<|open|>argument key="command" type="string"<|sep|>ls'
            "<|close|>argument<|sep|>"
        )
        result = extract_tool_calls_from_text(
            fmt.repair_stop_truncation(text), format="kimi_k3",
        )
        assert len(result.tool_calls) == 1

    def test_system_prompt(self):
        prompt = get_tool_format_system_prompt("kimi_k3", [])
        assert '<|open|>call tool=' in prompt

    def test_auto_detects_kimi_k3(self):
        text = (
            '<|open|>call tool="bash" index="1"<|sep|>'
            '<|open|>argument key="command" type="string"<|sep|>ls'
            "<|close|>argument<|sep|><|close|>call"
        )
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "bash"


class TestMiniMaxFormat:
    """MiniMax envelope around plain invoke/parameter markup."""

    def _sample(self, command):
        return (
            "<minimax:tool_call>\n"
            '<invoke name="Bash">\n'
            f'<parameter name="command">{command}</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )

    def test_single_invoke(self):
        text = "Checking the env.\n" + self._sample("ls -la code_library")
        result = extract_tool_calls_from_text(text, format="minimax")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la code_library"}
        assert "minimax:tool_call" not in result.cleaned_text
        assert result.error is None

    def test_multiline_heredoc_value(self):
        command = "cat > f.py <<'EOF'\nprint(1)\nEOF\npython f.py"
        result = extract_tool_calls_from_text(
            self._sample(command), format="minimax",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input["command"] == command

    def test_envelope_optional(self):
        text = (
            '<invoke name="Bash">\n'
            '<parameter name="command">ls</parameter>\n'
            "</invoke>"
        )
        result = extract_tool_calls_from_text(text, format="minimax")
        assert len(result.tool_calls) == 1

    def test_malformed_flagged(self):
        text = "<minimax:tool_call>\n<invoke name=Bash>broken"
        result = extract_tool_calls_from_text(text, format="minimax")
        assert result.tool_calls == []
        assert result.error is not None

    def test_prose_clean(self):
        result = extract_tool_calls_from_text("Just prose.", format="minimax")
        assert result.tool_calls == []
        assert result.error is None

    def test_system_prompt(self):
        prompt = get_tool_format_system_prompt("minimax", [])
        assert "<minimax:tool_call>" in prompt

    def test_auto_detects_minimax(self):
        result = extract_tool_calls_from_text(
            self._sample("ls"), format="auto",
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"


DS = "｜"


class TestDeepSeekFormat:
    """DeepSeek DSML markup (fullwidth-bar delimited)."""

    def _sample(self, command):
        return (
            f"<{DS}DSML{DS}tool_calls>\n"
            f'<{DS}DSML{DS}invoke name="Bash">\n'
            f'<{DS}DSML{DS}parameter name="command" string="true">{command}'
            f"</{DS}DSML{DS}parameter>\n"
            f"</{DS}DSML{DS}invoke>\n"
            f"</{DS}DSML{DS}tool_calls>"
        )

    def test_single_invoke(self):
        text = self._sample("cat code_library/example_controller.py")
        result = extract_tool_calls_from_text(text, format="deepseek")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {
            "command": "cat code_library/example_controller.py",
        }
        assert "DSML" not in result.cleaned_text
        assert result.error is None

    def test_multiline_heredoc_value(self):
        command = "cd /workspace && python << 'EOF'\nprint('hi')\nEOF"
        text = "Now the controller:\n\n" + self._sample(command)
        result = extract_tool_calls_from_text(text, format="deepseek")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input["command"] == command
        assert "Now the controller:" in result.cleaned_text

    def test_prose_without_markup_clean(self):
        result = extract_tool_calls_from_text("Just reasoning.", format="deepseek")
        assert result.tool_calls == []
        assert result.error is None

    def test_malformed_markup_flagged(self):
        text = f"<{DS}DSML{DS}invoke name=Bash>broken"
        result = extract_tool_calls_from_text(text, format="deepseek")
        assert result.tool_calls == []
        assert result.error is not None

    def test_system_prompt(self):
        prompt = get_tool_format_system_prompt("deepseek", [])
        assert f"<{DS}DSML{DS}invoke" in prompt

    def test_auto_detects_deepseek(self):
        text = self._sample("ls")
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"


class TestStopRepair:
    """Stop sequences strip the closing marker; repair restores it."""

    def test_bash_block_repair_closes_open_fence(self):
        fmt = get_format("bash_block")
        repaired = fmt.repair_stop_truncation("Check.\n```bash\nls -la")
        result = extract_tool_calls_from_text(repaired, format="bash_block")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {"command": "ls -la"}

    def test_bash_block_repair_noop_when_complete(self):
        fmt = get_format("bash_block")
        text = "```bash\nls\n```"
        assert fmt.repair_stop_truncation(text) == text

    def test_bash_block_repair_noop_without_fence(self):
        fmt = get_format("bash_block")
        assert fmt.repair_stop_truncation("no fence here") == "no fence here"

    def test_glm_repair_closes_tag(self):
        fmt = get_format("glm")
        text = "<tool_call>Bash<arg_key>command</arg_key><arg_value>ls</arg_value>"
        result = extract_tool_calls_from_text(
            fmt.repair_stop_truncation(text), format="glm",
        )
        assert len(result.tool_calls) == 1

    def test_kimi_repair_closes_token(self):
        fmt = get_format("kimi")
        text = (
            "<|tool_call_begin|>functions.Bash:0"
            '<|tool_call_argument_begin|>{"command": "ls"}'
        )
        result = extract_tool_calls_from_text(
            fmt.repair_stop_truncation(text), format="kimi",
        )
        assert len(result.tool_calls) == 1

    def test_auto_repair_chains_and_is_idempotent(self):
        fmt = get_format("auto")
        text = "Check.\n```bash\nls -la"
        repaired = fmt.repair_stop_truncation(fmt.repair_stop_truncation(text))
        result = extract_tool_calls_from_text(repaired, format="auto")
        assert len(result.tool_calls) == 1

    def test_auto_repair_never_corrupts_a_parsing_text(self):
        """The GLM turn-4 bug: a stray <tool_call> label before a valid
        fence made the hermes balance-repair glue </tool_call> onto the
        closing fence line, breaking the parse in the repair-then-parse
        pipeline. Repair must leave an already-parsing text untouched."""
        fmt = get_format("auto")
        text = (
            "Understanding the game mechanics.<tool_call>Bash Output:\n"
            "```bash\n"
            "cat > code_library/explore.py <<EOF\n"
            "print(1)\n"
            "EOF\n"
            "python code_library/explore.py\n"
            "```"
        )
        repaired = fmt.repair_stop_truncation(text)
        assert repaired == text
        result = extract_tool_calls_from_text(repaired, format="auto")
        assert len(result.tool_calls) == 1
        assert result.error is None

    def test_all_text_formats_declare_stops_except_meta_json(self):
        for name in ("bash_block", "hermes", "hermes_xml", "glm", "alan", "kimi", "auto"):
            assert get_format(name).stop_sequences, name
        assert get_format("meta_json").stop_sequences == ()

    def test_auto_stops_exclude_ambiguous_tag_closers(self):
        """Stray <tool_call>-label chatter before a real call must not let
        a tag stop cut the turn early under auto."""
        auto_stops = get_format("auto").stop_sequences
        assert "</tool_call>" not in auto_stops
        assert "</tool_use>" not in auto_stops


class TestAutoFormat:
    """Auto-detecting format: accept any registered markup."""

    def test_bash_block_detected(self):
        text = "Listing first.\n```bash\nls -la\n```"
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la"}

    def test_hermes_xml_detected(self):
        # Qwen3-Coder defecting to its trained markup under a bash_block prompt.
        text = (
            "<tool_call>\n<function=Bash>\n"
            "<parameter=command>ls -la</parameter>\n"
            "</function>\n</tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"
        assert result.tool_calls[0].input == {"command": "ls -la"}

    def test_glm_detected(self):
        text = (
            "<tool_call>Bash<arg_key>command</arg_key>"
            "<arg_value>ls -la</arg_value></tool_call>"
        )
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].input == {"command": "ls -la"}

    def test_hermes_json_detected(self):
        text = '<tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call>'
        result = extract_tool_calls_from_text(text, format="auto")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "Bash"

    def test_plain_text_no_calls_no_error(self):
        result = extract_tool_calls_from_text("Just thinking aloud.", format="auto")
        assert result.tool_calls == []
        assert result.error is None

    def test_malformed_gets_generic_error(self):
        result = extract_tool_calls_from_text(
            "<tool_call>total garbage</tool_call>", format="auto",
        )
        assert result.tool_calls == []
        assert result.error is not None
        assert "Preferred format" in result.error

    def test_thinking_source_excludes_bash_block(self):
        fence = "Draft:\n```bash\nrm -rf x\n```"
        result = extract_tool_calls_from_text(fence, format="auto", source="thinking")
        assert result.tool_calls == []
        # Structured markup still parses from thinking.
        xml = (
            "<tool_call>\n<function=Bash>\n"
            "<parameter=command>ls</parameter>\n</function>\n</tool_call>"
        )
        result = extract_tool_calls_from_text(xml, format="auto", source="thinking")
        assert len(result.tool_calls) == 1

    def test_system_prompt_teaches_bash_block(self):
        prompt = get_tool_format_system_prompt("auto", [])
        assert "```bash" in prompt


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

    def test_bash_block_prompt(self):
        prompt = get_tool_format_system_prompt("bash_block", [])
        assert "```bash" in prompt
        assert "ONE" in prompt

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_tool_format_system_prompt("unknown", [])


class TestFormatRegistry:
    """Test the format class registry."""

    def test_all_formats_registered(self):
        assert "hermes" in FORMATS
        assert "glm" in FORMATS
        assert "alan" in FORMATS
        assert "bash_block" in FORMATS
        assert "kimi" in FORMATS
        assert "auto" in FORMATS

    def test_get_format_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_format("unknown")

    def test_extract_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            extract_tool_calls_from_text("text", format="unknown")
