"""A bash_block generation ends once its one call is written."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.base import (
    LLMBackend,
    ModelInfo,
    StreamMessageDelta,
    StreamMessageStart,
    StreamMessageStop,
    StreamTextDelta,
    StreamThinkingDelta,
)
from alancode.messages.types import AssistantMessage

from tests.integration.test_bash_block_format import RecordingBashTool

RUNAWAY_DELTAS = 200
BLOCK = "```bash\necho real\n```"


class RunawayBackend(LLMBackend):
    """First call: optional reasoning, the given text pieces, then a long
    runaway tail. Later calls: a plain final answer."""

    def __init__(self, pieces, *, thinking=None):
        self.pieces = pieces
        self.thinking = thinking
        self.calls = 0
        self.tail_deltas_read = 0
        self.first_stream_closed_early = False

    def get_model_info(self, model=None):
        return ModelInfo(context_window=200_000, max_output_tokens=8_192)

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="runaway-test")
        if self.calls > 1:
            yield StreamTextDelta(text="Done.")
            yield StreamMessageDelta(stop_reason="end_turn")
            yield StreamMessageStop()
            return
        finished = False
        try:
            if self.thinking:
                yield StreamThinkingDelta(thinking=self.thinking)
            for piece in self.pieces:
                yield StreamTextDelta(text=piece)
            for i in range(RUNAWAY_DELTAS):
                self.tail_deltas_read += 1
                yield StreamTextDelta(text=f"\nI imagine the result {i}.")
            yield StreamMessageDelta(stop_reason="max_tokens")
            yield StreamMessageStop()
            finished = True
        finally:
            self.first_stream_closed_early = not finished


async def _run(tmp_path, backend, tool_call_format="bash_block", **settings):
    tool = RecordingBashTool()
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), programmatic=True,
        permission_mode="yolo", custom_system_prompt="t", tools=[tool],
        tool_call_format=tool_call_format,
    )
    for name, value in settings.items():
        assert agent.update_session_setting(name, value) is None
    events = [e async for e in agent.query_events_async("go")]
    replies = [e for e in events if isinstance(e, AssistantMessage) and not e.hide_in_api]
    return tool, replies


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call_format", ["bash_block", "auto"])
async def test_generation_ends_at_the_call_when_reasoning_is_over(tmp_path, tool_call_format):
    """DeepSeek-V4-Pro wrote its call in 15 tokens, then imagined the rest of
    the session to the 16000-token cap: 42 minutes, and the cut reply ran
    nothing."""
    backend = RunawayBackend([BLOCK], thinking="I will look around.")
    tool, replies = await _run(tmp_path, backend, tool_call_format)

    assert backend.tail_deltas_read <= 1, "the runaway tail must not be read"
    assert backend.first_stream_closed_early, "the backend stream must be closed"
    assert tool.commands == ["echo real"]
    assert replies[0].stop_reason == "end_turn"
    assert replies[-1].text == "Done."


@pytest.mark.asyncio
async def test_a_block_in_unclosed_inline_reasoning_does_not_end_it(tmp_path):
    """No reasoning channel and no </think> yet: the block may be a draft."""
    backend = RunawayBackend(["```bash\necho draft\n```", " hmm</think>", BLOCK])
    tool, _ = await _run(tmp_path, backend)

    assert backend.tail_deltas_read <= 1
    assert tool.commands == ["echo real"]


@pytest.mark.asyncio
async def test_without_any_sign_of_reasoning_the_reply_is_read_to_the_end(tmp_path):
    backend = RunawayBackend([BLOCK])
    await _run(tmp_path, backend)
    assert backend.tail_deltas_read == RUNAWAY_DELTAS


@pytest.mark.asyncio
async def test_an_unclosed_block_does_not_end_it(tmp_path):
    backend = RunawayBackend(["```bash\necho real\n"], thinking="thinking")
    await _run(tmp_path, backend)
    assert backend.tail_deltas_read == RUNAWAY_DELTAS


@pytest.mark.asyncio
async def test_the_setting_turns_it_off(tmp_path):
    backend = RunawayBackend([BLOCK], thinking="thinking")
    await _run(tmp_path, backend, stop_at_first_call=False)
    assert backend.tail_deltas_read == RUNAWAY_DELTAS


@pytest.mark.asyncio
async def test_formats_that_may_carry_several_calls_are_read_to_the_end(tmp_path):
    call = "<tool_call>\n<function=Bash>\n<parameter=command>echo real</parameter>\n</function>\n</tool_call>"
    backend = RunawayBackend([call], thinking="thinking")
    await _run(tmp_path, backend, tool_call_format="hermes_xml")
    assert backend.tail_deltas_read == RUNAWAY_DELTAS
