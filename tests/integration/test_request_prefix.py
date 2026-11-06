"""Each request extends the previous one, so a server can reuse its cache."""

import copy

import pytest

from alancode.agent import AlanCodeAgent

from tests.integration.test_bash_block_format import RecordingBashTool
from tests.integration.test_carry_reasoning import CALL, ScriptedBackend


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
