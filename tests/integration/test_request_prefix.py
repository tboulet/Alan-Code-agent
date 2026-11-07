"""Each request extends the previous one, so a server can reuse its cache."""

import copy

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.scripted_backend import ScriptedBackend as Script, text, tool_call

from tests.integration.test_bash_block_format import RecordingBashTool
from tests.integration.test_budget_regression import AuditedBackend, make_agent, run_turn
from tests.integration.test_carry_reasoning import CALL, ScriptedBackend
from tests.summary_request import is_summary_request


class RequestRecorder(ScriptedBackend):
    def __init__(self, turns):
        super().__init__(turns)
        self.requests = []

    async def stream(self, messages, system, tools, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        async for event in super().stream(messages, system, tools, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_a_new_turn_extends_the_last_request_after_a_cut_reply(tmp_path):
    """A MiMo run on llama.cpp: the first new turn after 137 calls dropped the
    notices of 14 cut replies, and the server reprocessed 145k tokens."""
    backend = RequestRecorder([
        ("", "Part one", "max_tokens"),
        ("", CALL, "end_turn"),
        ("", "Done.", "end_turn"),
        ("", "Done again.", "end_turn"),
    ])
    agent = AlanCodeAgent(
        backend=backend, cwd=str(tmp_path), programmatic=True,
        permission_mode="yolo", custom_system_prompt="t",
        tools=[RecordingBashTool()], tool_call_format="bash_block",
    )
    async for _ in agent.query_events_async("go"):
        pass
    async for _ in agent.query_events_async("keep going"):
        pass

    assert len(backend.requests) == 4
    for previous, current in zip(backend.requests, backend.requests[1:]):
        assert current[:len(previous)] == previous


class SentRecorder(AuditedBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent = []

    async def stream(self, messages, system, tools, **kwargs):
        self.sent.append(copy.deepcopy((messages, system, tools)))
        async for event in super().stream(messages, system, tools, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_the_summary_request_extends_the_last_request(tmp_path):
    """With its own system prompt and no tools, each MiMo compaction made
    llama.cpp re-read 190k tokens (65 min on CPU) for a 1k-token instruction."""
    backend = SentRecorder(
        Script.from_responses([tool_call("Dummy", {}) for _ in range(16)],
                              fallback=text("All done.")),
        context_window=16_384,
    )
    agent = make_agent(tmp_path, backend)
    await run_turn(agent, "flood me")

    first = next(i for i, (m, _, _) in enumerate(backend.sent) if is_summary_request(m))
    assert first > 0
    summary, last = backend.sent[first], backend.sent[first - 1]
    assert summary[1] == last[1] and summary[2] == last[2] and summary[2]
    assert summary[0][:len(last[0])] == last[0]
