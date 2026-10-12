"""A reply ending inside its reasoning on a bash block runs that block."""

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
from alancode.messages.types import ThinkingBlock

from tests.integration.test_stop_at_first_call import _run

BLOCK = "```bash\necho real\n```"
DRAFT = "```bash\necho draft\n```"


class ReasoningBackend(LLMBackend):
    """First call: the given reasoning and visible text. Later calls: a
    plain final answer."""

    def __init__(self, thinking, text="", stop_reason="end_turn"):
        self.thinking = thinking
        self.text = text
        self.stop_reason = stop_reason
        self.calls = 0

    def get_model_info(self, model=None):
        return ModelInfo(context_window=200_000, max_output_tokens=8_192)

    async def stream(self, messages, system, tools, **kwargs):
        self.calls += 1
        yield StreamMessageStart(model="reasoning-test")
        if self.calls > 1:
            yield StreamTextDelta(text="Done.")
            yield StreamMessageDelta(stop_reason="end_turn")
            yield StreamMessageStop()
            return
        yield StreamThinkingDelta(thinking=self.thinking)
        if self.text:
            yield StreamTextDelta(text=self.text)
        yield StreamMessageDelta(stop_reason=self.stop_reason)
        yield StreamMessageStop()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call_format", ["bash_block", "auto"])
async def test_block_closing_the_reasoning_runs_when_nothing_is_visible(tmp_path, tool_call_format):
    """DeepSeek-V4-Pro ended 381 of 3310 replies this way; the retry that
    followed ran the drafted command again 10 times."""
    backend = ReasoningBackend(f"First a draft.\n{DRAFT}\nBetter:\n{BLOCK}\n")
    tool, replies = await _run(tmp_path, backend, tool_call_format)

    assert tool.commands == ["echo real"]
    assert backend.calls == 2
    kept = "".join(b.thinking for b in replies[0].content if isinstance(b, ThinkingBlock))
    assert "echo draft" in kept and "echo real" not in kept


@pytest.mark.asyncio
async def test_reasoning_that_goes_on_after_the_block_is_a_draft(tmp_path):
    backend = ReasoningBackend(f"{BLOCK}\nNo, that would reset the level.")
    tool, _ = await _run(tmp_path, backend)
    assert tool.commands == []


@pytest.mark.asyncio
async def test_a_visible_answer_keeps_the_reasoning_block_a_draft(tmp_path):
    backend = ReasoningBackend(BLOCK, text="The level is solved.")
    tool, replies = await _run(tmp_path, backend)
    assert tool.commands == []
    assert replies[-1].text == "The level is solved."


@pytest.mark.asyncio
async def test_a_reply_cut_by_the_output_limit_runs_nothing(tmp_path):
    backend = ReasoningBackend(BLOCK, stop_reason="max_tokens")
    tool, _ = await _run(tmp_path, backend)
    assert tool.commands == []


@pytest.mark.asyncio
async def test_the_setting_turns_it_off(tmp_path):
    backend = ReasoningBackend(BLOCK)
    tool, _ = await _run(tmp_path, backend, run_reasoning_closing_call=False)
    assert tool.commands == []
