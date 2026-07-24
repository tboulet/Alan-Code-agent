"""Tests for package E: build_system_prompt extraction (#5) and
model-invoked skill tool restriction (#9)."""

from types import SimpleNamespace

import pytest

from alancode.agent import AlanCodeAgent
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call
from alancode.tools.base import ToolUseContext
from alancode.tools.builtin.skill_tool import SkillTool


class StubRegistry:
    """Minimal skill registry: one skill with an allowed-tools restriction."""

    def __init__(self, allowed_tools):
        self._skill = SimpleNamespace(
            name="restricted", allowed_tools=allowed_tools,
        )

    def get(self, name):
        return self._skill if name == "restricted" else None

    def expand(self, name, args):
        return f"Do the restricted thing. Args: {args}"

    def list_all(self):
        return [self._skill]


# ---------------------------------------------------------------------------
# #9 - SkillTool sets the hard filter on the per-turn context
# ---------------------------------------------------------------------------


class TestSkillToolSetsFilter:
    @pytest.mark.asyncio
    async def test_allowed_tools_set_on_context(self):
        tool = SkillTool(StubRegistry(allowed_tools=["Read", "Grep"]))
        ctx = ToolUseContext(cwd=".", messages=[])
        result = await tool.call({"skill": "restricted"}, ctx)
        assert not result.is_error
        assert ctx.active_skill_filter == ["Read", "Grep"]
        # The soft hint stays too
        assert "may only use" in result.data

    @pytest.mark.asyncio
    async def test_no_restriction_leaves_context_untouched(self):
        tool = SkillTool(StubRegistry(allowed_tools=None))
        ctx = ToolUseContext(cwd=".", messages=[])
        result = await tool.call({"skill": "restricted"}, ctx)
        assert not result.is_error
        assert ctx.active_skill_filter is None


class TestLoopEnforcesFilter:
    @pytest.mark.asyncio
    async def test_next_iteration_schemas_are_restricted(self, tmp_path):
        """After the model invokes a restricted skill, the NEXT API call's
        tool schemas contain only the allowed tools (+ Skill) - hard
        enforcement, not just the polite sentence in the prompt."""
        provider = ScriptedProvider.from_responses(
            [
                tool_call("Skill", {"skill": "restricted"}),
                text("done"),
            ],
            fallback=text("done"),
        )
        agent = AlanCodeAgent(
            backend=provider,
            cwd=str(tmp_path),
            programmatic=True,
            permission_mode="yolo",
            custom_system_prompt="You are a test agent.",
            extra_tools=[SkillTool(StubRegistry(allowed_tools=["Read"]))],
        )

        async for _ in agent.query_events_async("use the skill"):
            pass

        assert len(provider.call_log) >= 2
        first_names = {t.name for t in provider.call_log[0]["tools"]}
        second_names = {t.name for t in provider.call_log[1]["tools"]}
        # First call: full tool set (Skill not yet invoked)
        assert "Bash" in first_names
        # Second call: only the allowed tools (+ Skill, always kept)
        assert "Bash" not in second_names
        assert second_names <= {"Read", "FileRead", "Skill"}
        assert second_names & {"Read", "FileRead"}

    @pytest.mark.asyncio
    async def test_filter_clears_at_end_of_turn(self, tmp_path):
        """The restriction lives on the per-turn context: a NEW turn gets
        the full tool set again."""
        provider = ScriptedProvider.from_responses(
            [
                tool_call("Skill", {"skill": "restricted"}),
                text("done"),
                text("second turn"),
            ],
            fallback=text("done"),
        )
        agent = AlanCodeAgent(
            backend=provider,
            cwd=str(tmp_path),
            programmatic=True,
            permission_mode="yolo",
            custom_system_prompt="You are a test agent.",
            extra_tools=[SkillTool(StubRegistry(allowed_tools=["Read"]))],
        )

        async for _ in agent.query_events_async("use the skill"):
            pass
        async for _ in agent.query_events_async("new turn"):
            pass

        third_names = {t.name for t in provider.call_log[2]["tools"]}
        assert "Bash" in third_names  # full set restored


# ---------------------------------------------------------------------------
# #5 - build_system_prompt is the single source of truth
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _agent(self, tmp_path, **kwargs):
        return AlanCodeAgent(
            backend=ScriptedProvider.from_responses([], fallback=text("ok")),
            cwd=str(tmp_path),
            programmatic=True,
            permission_mode="yolo",
            **kwargs,
        )

    def test_includes_custom_prompt(self, tmp_path):
        agent = self._agent(tmp_path, custom_system_prompt="CUSTOM-MARKER agent")
        sections, boundary = agent.build_system_prompt()
        assert isinstance(sections, list) and sections
        assert isinstance(boundary, int) and boundary >= 0
        assert any("CUSTOM-MARKER" in s for s in sections)

    def test_text_tool_format_appends_schemas_section(self, tmp_path):
        agent = self._agent(
            tmp_path,
            custom_system_prompt="test agent",
            tool_call_format="hermes",
        )
        with_format, _ = agent.build_system_prompt()

        plain_agent = self._agent(tmp_path, custom_system_prompt="test agent")
        without_format, _ = plain_agent.build_system_prompt()

        assert len(with_format) == len(without_format) + 1
        assert "tool" in with_format[-1].lower()

    def test_stable_across_calls(self, tmp_path):
        """Preview (UI) and turn (API) calls must produce the same prompt."""
        agent = self._agent(tmp_path, custom_system_prompt="stable agent")
        first, b1 = agent.build_system_prompt()
        second, b2 = agent.build_system_prompt()
        assert first == second
        assert b1 == b2
