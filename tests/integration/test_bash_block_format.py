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


def make_agent(tmp_path, backend, tool, tool_call_format="bash_block"):
    return AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        tools=[tool],
        tool_call_format=tool_call_format,
        permission_mode="yolo",
        programmatic=True,
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
async def test_stop_sequences_sent_and_stop_cut_fence_repaired(tmp_path):
    """The format's stop sequences reach the backend, and a completion the
    server cut at the closing-fence stop (stripping it) still executes."""
    backend = TextTurnsBackend([
        (None, "Check the files first.\n```bash\nls code_library"),
        (None, "Done."),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

    events = [event async for event in agent.query_events_async("go")]

    assert events
    assert "\n```\n" in backend.stream_kwargs[0].get("stop_sequences")
    assert tool.commands == ["ls code_library"]


@pytest.mark.asyncio
async def test_max_tokens_cut_fence_not_repaired(tmp_path):
    """A fence cut by the OUTPUT LIMIT (not a stop sequence) must not be
    repaired into an executable call - truncation recovery handles it."""
    backend = TextTurnsBackend([
        (None, "Writing.\n```bash\ncat > f <<'EOF'\nif (", "max_tokens"),
        (None, "recovered"),
    ])
    tool = RecordingBashTool()
    agent = make_agent(tmp_path, backend, tool)

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
