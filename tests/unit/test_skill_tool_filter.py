"""Tests for package E: build_system_prompt extraction (#5) and
model-invoked skill tool restriction (#9)."""

from types import SimpleNamespace

import pytest

from alancode.agent import AlanCodeAgent
from alancode.backends.scripted_backend import ScriptedBackend, text, tool_call
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
    async def test_command_activated_filter_is_applied_to_first_call(self, tmp_path):
        """A filter activated by the /skill command applies before the loop starts."""
        backend = ScriptedBackend.from_responses([text("done")])
        agent = AlanCodeAgent(
            backend=backend,
            cwd=str(tmp_path),
            programmatic=True,
            permission_mode="yolo",
            custom_system_prompt="You are a test agent.",
        )
        agent._active_skill_filter = ["Read"]

        async for _ in agent.query_events_async("run the selected skill"):
            pass

        names = {tool.name for tool in backend.call_log[0]["tools"]}
        assert "Bash" not in names
        assert names <= {"Read", "FileRead", "Skill"}
        assert names & {"Read", "FileRead"}

    @pytest.mark.asyncio
    async def test_next_iteration_schemas_are_restricted(self, tmp_path):
        """After the model invokes a restricted skill, the NEXT API call's
        tool schemas contain only the allowed tools (+ Skill) - hard
        enforcement, not just the polite sentence in the prompt."""
        backend = ScriptedBackend.from_responses(
            [
                tool_call("Skill", {"skill": "restricted"}),
                text("done"),
            ],
            fallback=text("done"),
        )
        agent = AlanCodeAgent(
            backend=backend,
            cwd=str(tmp_path),
            programmatic=True,
            permission_mode="yolo",
            custom_system_prompt="You are a test agent.",
            extra_tools=[SkillTool(StubRegistry(allowed_tools=["Read"]))],
        )

        async for _ in agent.query_events_async("use the skill"):
            pass

        assert len(backend.call_log) >= 2
        first_names = {t.name for t in backend.call_log[0]["tools"]}
        second_names = {t.name for t in backend.call_log[1]["tools"]}
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
        backend = ScriptedBackend.from_responses(
            [
                tool_call("Skill", {"skill": "restricted"}),
                text("done"),
                text("second turn"),
            ],
            fallback=text("done"),
        )
        agent = AlanCodeAgent(
            backend=backend,
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

        third_names = {t.name for t in backend.call_log[2]["tools"]}
        assert "Bash" in third_names  # full set restored


# ---------------------------------------------------------------------------
# #5 - build_system_prompt is the single source of truth
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _agent(self, tmp_path, **kwargs):
        return AlanCodeAgent(
            backend=ScriptedBackend.from_responses([], fallback=text("ok")),
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

    def test_append_prompt_is_additive(self, tmp_path):
        agent = self._agent(tmp_path, append_system_prompt="APPEND-MARKER")
        sections, _ = agent.build_system_prompt()
        rendered = "\n".join(sections)
        assert "You are Alan Code" in rendered
        assert "# Using your tools" in rendered
        assert "Memory is currently disabled" in rendered
        assert sections[-1] == "APPEND-MARKER"

    def test_custom_prompt_replaces_builtin_but_accepts_explicit_append(
        self, tmp_path
    ):
        agent = self._agent(
            tmp_path,
            custom_system_prompt="CUSTOM-BASE",
            append_system_prompt="CUSTOM-APPEND",
        )
        sections, boundary = agent.build_system_prompt()
        assert sections == ["CUSTOM-BASE", "CUSTOM-APPEND"]
        assert boundary == 1

    def test_session_start_time_is_agent_scoped(self, tmp_path):
        first = self._agent(tmp_path)
        second = self._agent(tmp_path)
        first._session_started_at = "2026-08-08 10:00"
        second._session_started_at = "2026-08-08 11:00"

        first_prompt, _ = first.build_system_prompt()
        second_prompt, _ = second.build_system_prompt()

        assert "2026-08-08 10:00" in "\n".join(first_prompt)
        assert "2026-08-08 11:00" not in "\n".join(first_prompt)
        assert "2026-08-08 11:00" in "\n".join(second_prompt)

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

    @pytest.mark.asyncio
    async def test_custom_prompt_does_not_remove_native_tool_schemas(self, tmp_path):
        agent = self._agent(tmp_path, custom_system_prompt="CUSTOM-ONLY")

        async for _ in agent.query_events_async("use tools if needed"):
            pass

        call = agent._backend.call_log[0]
        assert call["system"] == ["CUSTOM-ONLY"]
        assert call["tools"]
        assert any(tool.name == "Bash" for tool in call["tools"])
        await agent.close()

    def test_stable_across_calls(self, tmp_path):
        """Preview (UI) and turn (API) calls must produce the same prompt."""
        agent = self._agent(tmp_path, custom_system_prompt="stable agent")
        first, b1 = agent.build_system_prompt()
        second, b2 = agent.build_system_prompt()
        assert first == second
        assert b1 == b2
