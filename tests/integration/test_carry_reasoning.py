"""Reasoning of replies made since the last tool call stays visible."""

import pytest

from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
)
from alancode.messages.types import CarriedReasoningBlock, ThinkingBlock

from alancode.agent import AlanCodeAgent

from tests.integration.test_bash_block_format import RecordingBashTool
from tests.integration.test_stop_at_first_call import BLOCK, RunawayBackend, _run

CALL = "```bash\necho real\n```"
CARRY = {"carry_reasoning_until_call": True}


class ScriptedBackend(LLMBackend):
    """Plays (thinking, text, stop_reason) turns and records, per request,
    the text of every assistant message it was sent."""

    def __init__(self, turns):
        self.turns = turns
        self.seen = []

    def get_model_info(self, model=None):
        return ModelInfo(context_window=200_000, max_output_tokens=8_192)

    async def stream(self, messages, system, tools, **kwargs):
        self.seen.append("\n".join(
            str(m.get("content") or "")
            for m in messages if m["role"] == "assistant"
        ))
        thinking, text, stop_reason = self.turns[len(self.seen) - 1]
        yield StreamMessageStart(model="scripted")
        if thinking:
            yield StreamThinkingDelta(thinking=thinking)
        if text:
            yield StreamTextDelta(text=text)
        yield StreamMessageDelta(stop_reason=stop_reason)
        yield StreamMessageStop()


def _agent(tmp_path, backend, **kwargs):
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), programmatic=True,
        permission_mode="yolo", custom_system_prompt="t",
        tools=[RecordingBashTool()], tool_call_format="bash_block", **kwargs,
    )
    assert agent.update_session_setting("carry_reasoning_until_call", True) is None
    return agent


def _capped_then_call():
    return ScriptedBackend([
        ("The key is at row 4.", "", "max_tokens"),
        ("So I go up.", CALL, "end_turn"),
        ("", "Done.", "end_turn"),
    ])


@pytest.mark.asyncio
async def test_reasoning_cut_at_the_cap_is_shown_until_the_next_call(tmp_path):
    """MiMo-V2.6-Pro-RL spent three 16000-token answers in a row re-deriving
    the same analysis: each was cut while reasoning and nothing was kept."""
    backend = _capped_then_call()
    tool, replies = await _run(tmp_path, backend, **CARRY)

    assert tool.commands == ["echo real"]
    assert "<think>The key is at row 4.</think>" in backend.seen[1]
    assert "row 4" not in backend.seen[2] and "I go up" not in backend.seen[2]
    stored = [b for r in replies for b in r.content]
    assert any(isinstance(b, ThinkingBlock) for b in stored)
    assert not any(isinstance(b, CarriedReasoningBlock) for b in stored)


@pytest.mark.asyncio
async def test_reasoning_of_an_empty_reply_is_shown_to_its_retry(tmp_path):
    backend = ScriptedBackend([
        ("The door is locked.", "", "end_turn"),
        ("", "Done.", "end_turn"),
    ])
    await _run(tmp_path, backend, **CARRY)
    assert "<think>The door is locked.</think>" in backend.seen[1]


@pytest.mark.asyncio
async def test_it_is_off_by_default(tmp_path):
    backend = _capped_then_call()
    await _run(tmp_path, backend)
    assert "row 4" not in backend.seen[1]


@pytest.mark.asyncio
async def test_persist_thinking_is_not_doubled(tmp_path):
    backend = _capped_then_call()
    await _run(tmp_path, backend, persist_thinking=True, **CARRY)
    assert backend.seen[1].count("row 4") == 1


@pytest.mark.asyncio
async def test_reasoning_of_a_finished_answer_is_shown_to_the_next_query(tmp_path):
    """A harness that answers a reply without a call by a reminder sends it
    as a new query."""
    backend = ScriptedBackend([
        ("The exit is on the left.", "I am done.", "end_turn"),
        ("", CALL, "end_turn"),
        ("", "Done.", "end_turn"),
    ])
    agent = _agent(tmp_path, backend)
    async for _ in agent.query_events_async("go"):
        pass
    async for _ in agent.query_events_async("keep going"):
        pass

    assert "<think>The exit is on the left.</think>" in backend.seen[1]
    assert "I am done." in backend.seen[2]
    assert "exit is on the left" not in backend.seen[2]


@pytest.mark.asyncio
async def test_reasoning_is_shown_again_after_a_resume(tmp_path):
    first = ScriptedBackend([("The exit is on the left.", "I am done.", "end_turn")])
    agent = _agent(tmp_path, first)
    async for _ in agent.query_events_async("go"):
        pass
    await agent.close()

    second = ScriptedBackend([("", "Done.", "end_turn")])
    resumed = _agent(tmp_path, second, session_id=agent._session_id)
    async for _ in resumed.query_events_async("keep going"):
        pass

    assert "<think>The exit is on the left.</think>" in second.seen[0]


@pytest.mark.asyncio
async def test_a_block_in_carried_reasoning_never_runs(tmp_path):
    backend = ScriptedBackend([
        (f"A draft:\n{CALL}\nNo, wrong.", "", "max_tokens"),
        ("", "Done.", "end_turn"),
    ])
    tool, _ = await _run(tmp_path, backend, **CARRY)
    assert "echo real" in backend.seen[1]
    assert tool.commands == []


@pytest.mark.asyncio
async def test_long_reasoning_is_cut_to_its_end(tmp_path):
    backend = ScriptedBackend([
        ("START " + "x" * 50_000 + " END", "", "max_tokens"),
        ("", "Done.", "end_turn"),
    ])
    await _run(tmp_path, backend, **CARRY)
    assert len(backend.seen[1]) < 15_000
    assert "END</think>" in backend.seen[1] and "START" not in backend.seen[1]
    assert "[earlier reasoning omitted]" in backend.seen[1]


@pytest.mark.asyncio
async def test_a_generation_still_ends_at_its_call(tmp_path):
    backend = RunawayBackend([BLOCK], thinking="I will look around.")
    tool, _ = await _run(tmp_path, backend, **CARRY)
    assert backend.tail_deltas_read <= 1
    assert tool.commands == ["echo real"]
