"""Tests for AGT Phase 2: SessionState extensions and GitCommit tool."""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.gui.scripted_ui import ScriptedUI
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call
from alancode.session.state import SessionState
from alancode.tools.base import ToolUseContext
from alancode.tools.builtin.git_commit import GitCommitTool
from tests.integration.git_helpers import GitTestRepo


# ═══════════════════════════════════════════════════════════════════════════════
# SessionState AGT properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionStateAGT:
    """Test new AGT properties on SessionState."""

    def test_alan_commits_default(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        assert state.alan_commits == []

    def test_alan_commits_set_get(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        state.alan_commits = ["abc123", "def456"]
        assert state.alan_commits == ["abc123", "def456"]

    def test_add_alan_commit(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        state.add_alan_commit("abc123")
        state.add_alan_commit("def456")
        assert state.alan_commits == ["abc123", "def456"]

    def test_conv_path_default(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        assert state.conv_path == []

    def test_add_to_conv_path(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        state.add_to_conv_path("sha1")
        state.add_to_conv_path("sha2")
        assert state.conv_path == ["sha1", "sha2"]

    def test_compaction_markers(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        state.add_compaction_marker("sha1")
        assert state.compaction_markers == ["sha1"]

    def test_session_root_sha(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        assert state.session_root_sha == ""
        state.session_root_sha = "abc123"
        assert state.session_root_sha == "abc123"

    def test_agent_position_sha(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        assert state.agent_position_sha == ""
        state.agent_position_sha = "def456"
        assert state.agent_position_sha == "def456"

    def test_batch_agt_updates(self, tmp_path):
        state = SessionState(session_id="test", cwd=str(tmp_path))
        with state.batch():
            state.session_root_sha = "root"
            state.agent_position_sha = "root"
            state.add_to_conv_path("root")
        # All should be set after batch
        assert state.session_root_sha == "root"
        assert state.agent_position_sha == "root"
        assert state.conv_path == ["root"]

    def test_persistence(self, tmp_path):
        """AGT properties survive reload from disk."""
        state1 = SessionState(session_id="agt-persist", cwd=str(tmp_path))
        state1.alan_commits = ["sha1", "sha2"]
        state1.session_root_sha = "root_sha"

        # Reload from disk
        state2 = SessionState(session_id="agt-persist", cwd=str(tmp_path))
        assert state2.alan_commits == ["sha1", "sha2"]
        assert state2.session_root_sha == "root_sha"

    def test_backward_compat(self, tmp_path):
        """Old state.json without AGT fields works fine."""
        import json
        state_dir = tmp_path / ".alan" / "sessions" / "old"
        state_dir.mkdir(parents=True)
        # Write old-style state with no AGT fields
        (state_dir / "state.json").write_text(json.dumps({
            "turn_count": 5,
            "total_cost_usd": 0.01,
        }))
        state = SessionState(session_id="old", cwd=str(tmp_path))
        assert state.turn_count == 5
        assert state.alan_commits == []
        assert state.session_root_sha == ""


# ═══════════════════════════════════════════════════════════════════════════════
# GitCommit tool (direct invocation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGitCommitTool:
    """Test GitCommitTool via direct call()."""

    @pytest.mark.asyncio
    async def test_basic_commit(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        state = SessionState(session_id="test-commit", cwd=str(git_repo.path))
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            session_state=state,
        )

        # Create a file to commit
        git_repo.write_file("new.py", "print('hello')")

        result = await tool.call(
            {"message": "Add new.py", "files": ["new.py"]},
            context,
        )

        assert not result.is_error
        assert "Committed" in result.data
        assert "Add new.py" in result.data

        # Verify git state
        assert not git_repo.is_dirty()
        assert "Add new.py" in git_repo.commit_message()

    @pytest.mark.asyncio
    async def test_updates_session_state(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        state = SessionState(session_id="test-state", cwd=str(git_repo.path))
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            session_state=state,
        )

        git_repo.write_file("f.py", "x")
        result = await tool.call({"message": "Test"}, context)

        assert not result.is_error
        sha = git_repo.head_sha()

        # Session state should be updated
        assert sha in state.alan_commits
        assert sha in state.conv_path
        assert state.agent_position_sha == sha

    @pytest.mark.asyncio
    async def test_no_files_adds_all(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            session_state=SessionState(session_id="t", cwd=str(git_repo.path)),
        )

        git_repo.write_file("a.py", "a")
        git_repo.write_file("b.py", "b")

        result = await tool.call({"message": "Add both"}, context)
        assert not result.is_error
        assert not git_repo.is_dirty()

    @pytest.mark.asyncio
    async def test_not_git_repo(self, tmp_path):
        tool = GitCommitTool()
        context = ToolUseContext(cwd=str(tmp_path), messages=[])

        result = await tool.call({"message": "Test"}, context)
        assert result.is_error
        assert "not a git repository" in result.data.lower()

    @pytest.mark.asyncio
    async def test_nothing_to_commit(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            session_state=SessionState(session_id="t", cwd=str(git_repo.path)),
        )

        # Clean tree — nothing to commit
        result = await tool.call({"message": "Empty"}, context)
        assert result.is_error
        assert "nothing to commit" in result.data.lower()

    @pytest.mark.asyncio
    async def test_allow_empty_commit(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            session_state=SessionState(session_id="t", cwd=str(git_repo.path)),
        )

        result = await tool.call(
            {"message": "Memory update", "allow_empty": True},
            context,
        )
        assert not result.is_error
        assert "Committed" in result.data

    @pytest.mark.asyncio
    async def test_empty_message_error(self, git_repo: GitTestRepo):
        tool = GitCommitTool()
        context = ToolUseContext(cwd=str(git_repo.path), messages=[])

        result = await tool.call({"message": ""}, context)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_permission_level_is_write(self):
        tool = GitCommitTool()
        assert tool.permission_level({}) == "write"

    @pytest.mark.asyncio
    async def test_no_session_state_still_works(self, git_repo: GitTestRepo):
        """GitCommit works even without session_state (AGT tracking skipped)."""
        tool = GitCommitTool()
        context = ToolUseContext(
            cwd=str(git_repo.path),
            messages=[],
            # session_state is None (default)
        )

        git_repo.write_file("f.py", "x")
        result = await tool.call({"message": "No state"}, context)
        assert not result.is_error
        assert "Committed" in result.data


# ═══════════════════════════════════════════════════════════════════════════════
# AGT init via agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestAGTInitialization:
    """Test that the agent initializes AGT root on first turn."""

    @pytest.mark.asyncio
    async def test_agt_root_initialized_in_git_repo(self, git_repo: GitTestRepo):
        provider = ScriptedProvider.from_responses([text("Hello")])
        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))

        # Before first turn: no root
        assert agent._session.session_root_sha == ""

        # Run one turn
        async for _ in agent.query_events_async("Hi"):
            pass

        # After first turn: root should be set
        assert agent._session.session_root_sha != ""
        assert agent._session.session_root_sha == git_repo.head_sha()
        assert agent._session.agent_position_sha == git_repo.head_sha()
        assert len(agent._session.conv_path) == 1

    @pytest.mark.asyncio
    async def test_agt_root_not_set_outside_git(self, tmp_path):
        provider = ScriptedProvider.from_responses([text("Hello")])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        async for _ in agent.query_events_async("Hi"):
            pass

        # Not a git repo — AGT root should remain empty
        assert agent._session.session_root_sha == ""

    @pytest.mark.asyncio
    async def test_agt_root_not_reset_on_second_turn(self, git_repo: GitTestRepo):
        provider = ScriptedProvider.from_responses([
            text("First"), text("Second"),
        ])
        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))

        # First turn
        async for _ in agent.query_events_async("Turn 1"):
            pass
        root = agent._session.session_root_sha

        # Make a commit to change HEAD
        git_repo.write_file("new.py", "x")
        git_repo.commit("External commit")

        # Second turn — root should NOT change
        async for _ in agent.query_events_async("Turn 2"):
            pass
        assert agent._session.session_root_sha == root


# ═══════════════════════════════════════════════════════════════════════════════
# GitCommit tool via agent (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGitCommitViaAgent:
    """Test GitCommit tool through the full agent pipeline."""

    @pytest.mark.asyncio
    async def test_agent_uses_git_commit(self, git_repo: GitTestRepo):
        """Agent calls GitCommit tool, commit appears in git."""
        git_repo.write_file("feature.py", "def feature(): pass")

        provider = ScriptedProvider.from_responses([
            tool_call("GitCommit", {"message": "Add feature", "files": ["feature.py"]}),
            text("Committed successfully."),
        ])
        ui = ScriptedUI.from_inputs(["Commit the feature", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))

        await run_session(agent, ui)

        # Verify commit was made
        assert "Add feature" in git_repo.commit_message()
        assert not git_repo.is_dirty()

        # Verify AGT state
        sha = git_repo.head_sha()
        assert sha in agent._session.alan_commits
        assert sha in agent._session.conv_path
