"""inserted_example_bash_use opens a new session with a real example call."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
)
from alancode.messages.factory import SYNTHETIC_MODEL
from alancode.messages.types import AssistantMessage
from alancode.query.inserted_example import EXAMPLE_COMMAND, EXAMPLE_PREFACE
from alancode.tools.text_tool_parser import FORMATS

from tests.integration.test_bash_block_format import RecordingBashTool


class FinalAnswerBackend(LLMBackend):
    """Answers every call with plain text; keeps every request it was sent."""

    def __init__(self):
        self.requests = []

    def get_model_info(self, model=None):
        return ModelInfo(context_window=200_000, max_output_tokens=8_192)

    async def stream(self, messages, system, tools, **kwargs):
        self.requests.append(messages)
        yield StreamMessageStart(model="test")
        yield StreamTextDelta(text="Done.")
        yield StreamMessageDelta(stop_reason="end_turn")
        yield StreamMessageStop()


def _agent(tmp_path, backend, tool, tool_call_format="bash_block", **kwargs):
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), programmatic=True,
        permission_mode="yolo", custom_system_prompt="t", tools=[tool],
        tool_call_format=tool_call_format, **kwargs,
    )
    assert agent.update_session_setting("inserted_example_bash_use", True) is None
    return agent


async def _ask(agent, text="go"):
    return [e async for e in agent.query_events_async(text)]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call_format", sorted(FORMATS))
async def test_first_request_carries_the_example_in_the_session_format(tmp_path, tool_call_format):
    backend, tool = FinalAnswerBackend(), RecordingBashTool()
    agent = _agent(tmp_path, backend, tool, tool_call_format)
    await _ask(agent)

    assert tool.commands == [EXAMPLE_COMMAND], "the example must really run"
    first = backend.requests[0]
    assert [m["role"] for m in first] == ["user", "assistant", "user"]
    shown = first[1]["content"]
    assert EXAMPLE_PREFACE in shown
    call = FORMATS[tool_call_format].parse(shown)
    assert [(c.name, c.input) for c in call] == [("Bash", {"command": EXAMPLE_COMMAND})]
    assert "ok" in first[2]["content"]


@pytest.mark.asyncio
async def test_native_tool_calls_get_a_structured_example(tmp_path):
    backend, tool = FinalAnswerBackend(), RecordingBashTool()
    agent = _agent(tmp_path, backend, tool, tool_call_format=None)
    await _ask(agent)

    first = backend.requests[0]
    assert [m["role"] for m in first] == ["user", "assistant", "tool"]
    assert first[1]["tool_calls"][0]["function"]["name"] == "Bash"
    assert first[2]["tool_call_id"] == first[1]["tool_calls"][0]["id"]


@pytest.mark.asyncio
async def test_the_example_is_recorded_as_synthetic_and_only_once(tmp_path):
    backend, tool = FinalAnswerBackend(), RecordingBashTool()
    agent = _agent(tmp_path, backend, tool)
    events = await _ask(agent)
    await _ask(agent, "and now?")

    assert tool.commands == [EXAMPLE_COMMAND]
    example = [e for e in events if isinstance(e, AssistantMessage) and e.has_tool_use]
    assert len(example) == 1 and example[0].model == SYNTHETIC_MODEL
    assert example[0] in agent._messages


@pytest.mark.asyncio
async def test_a_resumed_session_gets_no_new_example(tmp_path):
    backend, tool = FinalAnswerBackend(), RecordingBashTool()
    agent = _agent(tmp_path, backend, tool)
    await _ask(agent)
    await agent.close()

    resumed = _agent(tmp_path, FinalAnswerBackend(), tool, session_id=agent._session_id)
    await _ask(resumed, "continue")
    assert tool.commands == [EXAMPLE_COMMAND]


@pytest.mark.asyncio
async def test_it_is_off_by_default(tmp_path):
    backend, tool = FinalAnswerBackend(), RecordingBashTool()
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), programmatic=True,
        permission_mode="yolo", custom_system_prompt="t", tools=[tool],
        tool_call_format="bash_block",
    )
    await _ask(agent)
    assert tool.commands == []
    assert [m["role"] for m in backend.requests[0]] == ["user"]
