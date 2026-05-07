"""Tests for AlanCodeAgent's programmatic-mode flag and tool selection API."""

from unittest.mock import patch

import pytest

from alancode.agent import AlanCodeAgent
from alancode.providers.scripted_provider import ScriptedProvider, ScriptedResponse


@pytest.fixture
def provider():
    return ScriptedProvider.from_responses([ScriptedResponse(text="ok")])


# ── Tool selection ────────────────────────────────────────────────────────


class TestToolSelection:
    def test_default_includes_all_builtins(self, provider, tmp_path):
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))
        names = {t.name for t in agent._tools}
        assert "Bash" in names
        assert "WebFetch" in names
        assert "GitCommit" in names
        assert "Skill" in names

    def test_programmatic_excludes_external_tools(self, provider, tmp_path):
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path), programmatic=True,
        )
        names = {t.name for t in agent._tools}
        assert "WebFetch" not in names
        assert "GitCommit" not in names
        assert "AskUserQuestion" not in names
        assert "Skill" not in names
        assert "Bash" in names
        assert "Read" in names

    def test_disabled_tools_filters(self, provider, tmp_path):
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path),
            disabled_tools=["Bash", "WebFetch"],
        )
        names = {t.name for t in agent._tools}
        assert "Bash" not in names
        assert "WebFetch" not in names
        assert "Read" in names

    def test_explicit_tools_replaces_default(self, provider, tmp_path):
        from tests.conftest import EchoTool
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path),
            tools=[EchoTool()],
        )
        names = [t.name for t in agent._tools]
        assert names == ["Echo"]

    def test_extra_tools_still_appended(self, provider, tmp_path):
        from tests.conftest import EchoTool
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path),
            programmatic=True,
            extra_tools=[EchoTool()],
        )
        names = {t.name for t in agent._tools}
        assert "Echo" in names
        assert "GitCommit" not in names

    def test_disabled_tools_composes_with_extra(self, provider, tmp_path):
        from tests.conftest import EchoTool
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path),
            disabled_tools=["Echo"],
            extra_tools=[EchoTool()],
        )
        names = [t.name for t in agent._tools]
        assert "Echo" in names


# ── Instruction loaders gated behind programmatic ─────────────────────────


class TestInstructionGating:
    @pytest.mark.asyncio
    async def test_programmatic_skips_global_and_project_instructions(
        self, provider, tmp_path,
    ):
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path), programmatic=True,
        )
        with patch(
            "alancode.agent.load_global_project_instructions",
            return_value="GLOBAL",
        ) as mock_global, patch(
            "alancode.agent.load_project_instructions",
            return_value="PROJECT",
        ) as mock_project, patch(
            "alancode.agent.load_global_memory_index",
            return_value="GLOBAL_MEM",
        ) as mock_global_mem:
            async for _ in agent.query_events_async("hi"):
                pass
        assert not mock_global.called
        assert not mock_project.called
        assert not mock_global_mem.called

    @pytest.mark.asyncio
    async def test_default_calls_global_and_project_instructions(
        self, provider, tmp_path,
    ):
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))
        with patch(
            "alancode.agent.load_global_project_instructions",
            return_value=None,
        ) as mock_global, patch(
            "alancode.agent.load_project_instructions",
            return_value=None,
        ) as mock_project, patch(
            "alancode.agent.load_global_memory_index",
            return_value=None,
        ) as mock_global_mem:
            async for _ in agent.query_events_async("hi"):
                pass
        assert mock_global.called
        assert mock_project.called
        assert mock_global_mem.called


# ── AGT init gated ────────────────────────────────────────────────────────


class TestAgtGating:
    @pytest.mark.asyncio
    async def test_programmatic_skips_init_agt_root(self, provider, tmp_path):
        agent = AlanCodeAgent(
            provider=provider, cwd=str(tmp_path), programmatic=True,
        )
        with patch.object(agent, "_init_agt_root") as mock_init:
            async for _ in agent.query_events_async("hi"):
                pass
        assert not mock_init.called

    @pytest.mark.asyncio
    async def test_default_calls_init_agt_root(self, provider, tmp_path):
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))
        with patch.object(agent, "_init_agt_root") as mock_init:
            async for _ in agent.query_events_async("hi"):
                pass
        assert mock_init.called
