"""End-to-end bash_block extraction through the query loop."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.messages.types import (
    AssistantMessage,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
)
from alancode.tools.base import Tool, ToolResult


class RecordingBashTool(Tool):
    """Stand-in for the Bash tool that records commands instead of running them."""

    def __init__(self):
        self.commands = []

    @property
    def name(self):
        return "Bash"

    @property
    def description(self):
        return "Run a shell command"

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    async def call(self, args, context):
        self.commands.append(args.get("command", ""))
        return ToolResult(data="ok")

    def permission_level(self, args):
        return "read"


class TextTurnsBackend(LLMBackend):
    """Streams scripted (thinking, text[, stop_reason]) turns, one per call."""

    def __init__(self, turns):
        self.turns = turns
        self.calls = 0
        self.stream_kwargs = []

    async def stream(self, messages, system, tools, **kwargs):
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        thinking, text = turn[0], turn[1]
        stop_reason = turn[2] if len(turn) > 2 else "end_turn"
        self.calls += 1
        self.stream_kwargs.append(kwargs)
        yield StreamMessageStart(model="bash-block-test")
        if thinking:
            yield StreamThinkingDelta(thinking=thinking)
        if text:
            yield StreamTextDelta(text=text)
        yield StreamMessageDelta(stop_reason=stop_reason)
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


def make_agent(tmp_path, backend, tool, tool_call_format="bash_block",
               programmatic=True):
    return AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        tools=[tool],
        tool_call_format=tool_call_format,
        permission_mode="yolo",
        programmatic=programmatic,
    )


@pytest.mark.asyncio
async def test_block_executes_and_markup_removed(tmp_path):
    command = "cat > f.py <<'EOF'\nprint(\"hi\")\nEOF\npython3 f.py"
    backend = TextTurnsBackend([
        (None, f"Writing the test file now.\n\n```bash\n{command}\n```"),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

    events = [event async for event in agent.query_events_async("write it")]

    assert backend.calls == 2
    assert tool.commands == [command]

    assistant_messages = [
        e for e in events
        if isinstance(e, AssistantMessage) and not e.hide_in_api
    ]
    tool_message = assistant_messages[0]
    assert any(isinstance(b, ToolUseBlock) for b in tool_message.content)
    assert "```bash" not in tool_message.text
    assert "Writing the test file now." in tool_message.text

    tool_results = [
        b
        for e in events
        if isinstance(e, UserMessage) and isinstance(e.content, list)
        for b in e.content
        if isinstance(b, ToolResultBlock)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].content == "ok"
    assert assistant_messages[-1].text == "Done."


@pytest.mark.asyncio
async def test_no_fence_stop_is_sent(tmp_path):
    """A fence stop is applied to the whole generation, reasoning included,
    so a thinking model quoting a grid is cut before writing anything
    visible. bash_block therefore sends no stop sequences at all."""
    backend = TextTurnsBackend([
        (None, "Check the files first.\n```bash\nls code_library\n```\n"),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert not backend.stream_kwargs[0].get("stop_sequences")
    assert tool.commands == ["ls code_library"]


@pytest.mark.asyncio
async def test_unclosed_block_is_not_executed(tmp_path):
    """Without a stop to blame, an unclosed fence means the model never
    finished writing the command - running it would execute a fragment."""
    backend = TextTurnsBackend([
        (None, "Check the files first.\n```bash\nrm -rf /important"),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

    [event async for event in agent.query_events_async("go")]

    assert tool.commands == []


@pytest.mark.asyncio
async def test_truncation_trumps_malformed_detection(tmp_path):
    """A max_tokens-truncated response that also pattern-matches a
    malformed structured call must go to the length recovery, not the
    format-error retry."""
    backend = TextTurnsBackend([
        (
            None,
            "hmm <tool_call>not a call</tool_call> then\n"
            "```bash\ncat > f <<'EOF'\ntruncat",
            "max_tokens",
        ),
        (None, "recovered"),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="auto", programmatic=False)

    events = [event async for event in agent.query_events_async("go")]

    assert backend.calls == 2
    assert tool.commands == []
    format_errors = [
        e for e in events
        if isinstance(e, UserMessage) and "Preferred format" in str(e.content)
    ]
    assert format_errors == []


@pytest.mark.asyncio
async def test_max_tokens_cut_fence_not_repaired(tmp_path):
    """A fence cut by the OUTPUT LIMIT (not a stop sequence) must not be
    repaired into an executable call - truncation recovery handles it."""
    backend = TextTurnsBackend([
        (None, "Writing.\n```bash\ncat > f <<'EOF'\nif (", "max_tokens"),
        (None, "recovered"),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, programmatic=False)

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert tool.commands == []
    assert backend.calls == 2  # truncation recovery retried the call


@pytest.mark.asyncio
async def test_auto_accepts_native_markup_defection(tmp_path):
    """Under tool_call_format=auto, a model that ignores the bash_block
    prompt and emits its trained hermes_xml markup still executes."""
    backend = TextTurnsBackend([
        (
            None,
            "I'll check the files.\n"
            "<tool_call>\n<function=Bash>\n"
            "<parameter=command>ls code_library</parameter>\n"
            "</function>\n</tool_call>",
        ),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="auto")

    events = [event async for event in agent.query_events_async("look around")]

    assert backend.calls == 2
    assert tool.commands == ["ls code_library"]
    assistant_messages = [
        e for e in events
        if isinstance(e, AssistantMessage) and not e.hide_in_api
    ]
    assert assistant_messages[-1].text == "Done."


@pytest.mark.asyncio
async def test_auto_thinking_fence_draft_not_executed(tmp_path):
    """auto keeps the bash_block thinking guard: a fence drafted in
    reasoning is never an action, while the visible fence executes."""
    backend = TextTurnsBackend([
        (
            "Draft first:\n```bash\nrm -rf code_library\n```\nNo, safer:",
            "Safer check first.\n```bash\nls code_library\n```",
        ),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="auto")

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert tool.commands == ["ls code_library"]


@pytest.mark.asyncio
async def test_auto_kimi_opaque_id_remapped_to_single_tool(tmp_path):
    """Kimi K2.7 emits opaque function-ids; with one registered tool the
    loop remaps the call instead of dropping it."""
    backend = TextTurnsBackend([
        (
            None,
            "Probing the env."
            "<|tool_calls_section_begin|><|tool_call_begin|>text_de60e4f6"
            '<|tool_call_argument_begin|>{"command": "python explore.py"}'
            "<|tool_call_end|><|tool_calls_section_end|>",
        ),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="auto")

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert tool.commands == ["python explore.py"]


@pytest.mark.asyncio
async def test_auto_deepseek_dsml_executes(tmp_path):
    bar = "｜"
    backend = TextTurnsBackend([
        (
            None,
            f"<{bar}DSML{bar}tool_calls>\n"
            f'<{bar}DSML{bar}invoke name="Bash">\n'
            f'<{bar}DSML{bar}parameter name="command" string="true">'
            f"cat solution.py</{bar}DSML{bar}parameter>\n"
            f"</{bar}DSML{bar}invoke>\n"
            f"</{bar}DSML{bar}tool_calls>",
        ),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="auto")

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert tool.commands == ["cat solution.py"]


@pytest.mark.asyncio
async def test_draft_block_in_thinking_not_executed(tmp_path):
    """A fenced block drafted in reasoning content is not an action: the
    turn must resolve as a normal no-tool answer."""
    backend = TextTurnsBackend([
        (
            "Maybe something like:\n```bash\nrm -rf code_library\n```\n"
            "No - first I should look around.",
            "I need to inspect the environment before acting.",
        ),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

    events = [event async for event in agent.query_events_async("go")]

    assert backend.calls == 1
    assert tool.commands == []
    assert not any(
        isinstance(b, ToolUseBlock)
        for e in events
        if isinstance(e, AssistantMessage)
        for b in e.content
    )


@pytest.mark.asyncio
async def test_kimi_substituted_argument_token_still_executes(tmp_path):
    """Kimi-K2.6 sometimes writes a stray "<think>" where its
    argument-begin token belongs, and echoes back a text_<hex> call id
    instead of the tool name. Neither is a reason to drop the call: the
    JSON is complete, and a single-tool agent already remaps the name.
    Unparsed, the deterministic nudge loops at temperature 0 - measured at
    30 of 33 completions in one bench-03 attempt.
    """
    backend = TextTurnsBackend([
        (None,
         '<|tool_calls_section_begin|><|tool_call_begin|>text_c35d869d'
         '<think>{"command": "echo hi"}<|tool_call_end|>'
         '<|tool_calls_section_end|>'),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, tool_call_format="kimi")

    [event async for event in agent.query_events_async("go")]

    assert tool.commands == ["echo hi"]


@pytest.mark.asyncio
async def test_text_dialect_history_shows_the_dialect_the_model_was_taught(tmp_path):
    """A structured tool_calls entry is re-rendered by the server's chat
    template into that model's NATIVE markup, carrying ids alancode minted.
    A model taught a text dialect then sees a different one in its own
    history and imitates it - measured on bench-04 Kimi-K2.6, 27% and 66%
    of iterations emitting garbled native markup with alancode's text_<hex>
    ids in it, against 0% on every other arm.
    """
    class Capturing(TextTurnsBackend):
        def __init__(self, turns):
            super().__init__(turns)
            self.sent = []

        async def stream(self, messages, system, tools, **kwargs):
            self.sent.append(messages)
            async for event in super().stream(messages, system, tools, **kwargs):
                yield event

    backend = Capturing([
        (None, "Let me look.\n```bash\nls\n```\n"),
        (None, "Done."),
    ])
    agent = make_agent(tmp_path, backend, RecordingBashTool(),
                       tool_call_format="bash_block")
    [event async for event in agent.query_events_async("go")]

    replay = backend.sent[1]
    assistant = [m for m in replay if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert "```bash" in assistant[0]["content"]
    assert "tool_calls" not in assistant[0]
    # No role:tool entry may survive, or it would reference a dropped id.
    assert not [m for m in replay if m["role"] == "tool"]
    assert not any("text_" in str(m.get("content", "")) for m in replay)


@pytest.mark.asyncio
async def test_truncated_block_is_never_completed_by_the_next_generation(tmp_path):
    """The cut attempt stays in history as text. It must never combine with
    what comes next into a call: a fence left open mid-file plus a later
    closing fence would otherwise parse into a command carrying a
    half-written file, and run it.
    """
    backend = TextTurnsBackend([
        # Cut mid-heredoc: the fence is still open.
        (None, "Writing it now.\n```bash\ncat > solution.py <<EOF\nimport numpy as np\ndef solve(", "max_tokens"),
        # The model carries on; on its own this closes nothing it opened.
        (None, "EOF\n```\nThat completes the file."),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, programmatic=False)

    [event async for event in agent.query_events_async("write solution.py")]

    for command in tool.commands:
        assert "def solve(\nEOF" not in command, (
            f"a truncated block was stitched into a command: {command!r}"
        )


@pytest.mark.asyncio
async def test_turn_ending_on_an_unclosed_block_gets_format_feedback(tmp_path):
    """GLM-5.3-Flash ended its first turn on a ```bash block it never
    closed. Nothing parsed and nothing was flagged, so the prose was taken as
    the final answer and the session ended after one turn with 0 calls."""
    backend = TextTurnsBackend([
        (None, "I will read them, one per line.```bash\ncat notes/a.txt\ncat notes/b.txt\n"),
        (None, "Reading them.\n```bash\ncat notes/a.txt\n```\n"),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool, programmatic=True)

    [event async for event in agent.query_events_async("read the notes")]

    assert backend.calls >= 2, "the session ended on the unparsed block"
    assert tool.commands == ["cat notes/a.txt"], "only the well-formed retry runs"
