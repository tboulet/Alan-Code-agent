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
    """Streams scripted (thinking, text) pairs, one per call."""

    def __init__(self, turns):
        self.turns = turns
        self.calls = 0

    async def stream(self, messages, system, tools, **kwargs):
        thinking, text = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        yield StreamMessageStart(model="bash-block-test")
        if thinking:
            yield StreamThinkingDelta(thinking=thinking)
        if text:
            yield StreamTextDelta(text=text)
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()

    def get_model_info(self, model=None):
        return ModelInfo(context_window=131_072)


def make_agent(tmp_path, backend, tool):
    return AlanCodeAgent(
        backend=backend,
        cwd=str(tmp_path),
        tools=[tool],
        tool_call_format="bash_block",
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
