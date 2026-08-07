"""Backend lifecycle behavior owned by AlanCodeAgent."""

import asyncio

import pytest

import alancode.agent as agent_module
from alancode.agent import AlanCodeAgent
from alancode.backends.scripted_backend import ScriptedBackend


class CloseTrackingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_agent_close_closes_backend_once(tmp_path):
    backend = CloseTrackingBackend()
    agent = AlanCodeAgent(backend=backend, cwd=str(tmp_path))

    await agent.close()
    await agent.close()

    assert backend.close_calls == 1


@pytest.mark.asyncio
async def test_recreated_backend_closes_previous_instance(tmp_path):
    backend = CloseTrackingBackend()
    agent = AlanCodeAgent(backend=backend, cwd=str(tmp_path))

    assert agent.update_session_setting("backend", "scripted") is None
    await asyncio.sleep(0)

    assert backend.close_calls == 1
    await agent.close()


def test_failed_backend_recreation_rolls_back_session_setting(
    tmp_path, monkeypatch
):
    backend = CloseTrackingBackend()
    agent = AlanCodeAgent(
        backend=backend,
        model="gpt-4o",
        cwd=str(tmp_path),
    )
    old_settings = dict(agent._settings)

    def fail_creation(settings):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(
        agent_module,
        "_create_backend_from_settings",
        fail_creation,
    )

    error = agent.update_session_setting("model", "claude-sonnet-4-6")

    assert error == "Failed to create backend: backend unavailable"
    assert agent._settings == old_settings
    assert agent._model == "gpt-4o"
    assert agent._backend is backend
