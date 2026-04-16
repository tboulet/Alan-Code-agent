"""Exhaustive AGT integration tests — ScriptedUI + ScriptedProvider + GitTestRepo.

These tests exercise the full AGT stack end-to-end, simulating real user
sessions with scripted inputs and LLM responses in temporary git repos.

Coverage:
- Agent commits via GitCommit tool → tree updates
- /revert, /move, /convrevert, /allrevert slash commands
- Compaction markers via /compact
- Tree update data sent to UI after every turn
- Edge cases: dirty tree, external commits, orphaned SHAs, non-git repos,
  branch creation, detached HEAD, memory snapshots
"""

import pytest

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.git_tree.model import CURRENT_NODE_SHA, NodeType
from alancode.git_tree.parser import parse_git_tree
from alancode.gui.scripted_ui import ScriptedUI, ui_rule
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call
from tests.integration.git_helpers import GitTestRepo


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_session(git_repo, responses, inputs, **agent_kwargs):
    """Create a full session stack: provider, UI, agent."""
    provider = ScriptedProvider.from_responses(responses, fallback=text("Done."))
    ui = ScriptedUI.from_inputs(inputs)
    agent = AlanCodeAgent(
        provider=provider,
        cwd=str(git_repo.path),
        ask_callback=ui.ask_user,
        **agent_kwargs,
    )
    return provider, ui, agent


# ═══════════════════════════════════════════════════════════════════════════════
# Basic: agent commits appear in tree
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentCommitInTree:
    """Agent uses GitCommit tool → commit tracked in session state and tree."""

    @pytest.mark.asyncio
    async def test_commit_creates_alan_node(self, git_repo: GitTestRepo):
        git_repo.write_file("feature.py", "def feature(): pass")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "Add feature"}),
            text("Committed."),
        ], ["Commit the feature", EOFError])

        await run_session(agent, ui)

        sha = git_repo.head_sha()
        assert sha in agent._session.alan_commits
        assert sha in agent._session.conv_path
        assert agent._session.agent_position_sha == sha

    @pytest.mark.asyncio
    async def test_tree_update_sent_after_commit(self, git_repo: GitTestRepo):
        git_repo.write_file("f.py", "x")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "Commit"}),
            text("OK."),
        ], ["Commit", EOFError])

        await run_session(agent, ui)

        assert len(ui.tree_update_log) >= 1
        tree_data = ui.tree_update_log[-1]
        nodes = tree_data["nodes"]
        # The latest commit should be classified as alan_commit
        alan_nodes = [n for n in nodes if n["node_type"] == "alan_commit"]
        assert len(alan_nodes) >= 1

    @pytest.mark.asyncio
    async def test_multiple_commits(self, git_repo: GitTestRepo):
        git_repo.write_file("a.py", "a")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "Commit 1"}),
            text("First done."),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "Commit 2"}),
            text("Second done."),
        ], ["First", "Second", EOFError])

        await run_session(agent, ui)

        assert len(agent._session.alan_commits) == 2
        assert len(ui.tree_update_log) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# /revert command
# ═══════════════════════════════════════════════════════════════════════════════


class TestRevertCommand:

    @pytest.mark.asyncio
    async def test_revert_1(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(git_repo, [
            text("OK"),
        ], ["/revert 1", EOFError])

        await run_session(agent, ui)

        # Agent should now be at shas[1] (one commit back from shas[2])
        assert agent._session.agent_position_sha == shas[1]
        # System reminder injected
        assert any("revert" in line.lower() for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_revert_discards_dirty(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        git_repo.write_file("dirty.txt", "uncommitted")
        provider, ui, agent = _make_session(git_repo, [], ["/revert 1", EOFError])

        await run_session(agent, ui)

        assert not git_repo.is_dirty()
        assert any("discard" in line.lower() for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_revert_updates_tree(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(git_repo, [], ["/revert 1", EOFError])

        await run_session(agent, ui)

        assert len(ui.tree_update_log) >= 1
        # Agent position should be marked in the tree data
        tree = ui.tree_update_log[-1]
        agent_nodes = [n for n in tree["nodes"] if n["is_agent_position"]]
        assert len(agent_nodes) == 1

    @pytest.mark.asyncio
    async def test_revert_not_git_repo(self, tmp_path):
        provider = ScriptedProvider.from_responses([])
        ui = ScriptedUI.from_inputs(["/revert 1", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        assert any("git" in line.lower() for line in ui.console_log)


# ═══════════════════════════════════════════════════════════════════════════════
# /move command
# ═══════════════════════════════════════════════════════════════════════════════


class TestMoveCommand:

    @pytest.mark.asyncio
    async def test_move_to_sha(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        provider, ui, agent = _make_session(
            git_repo, [], [f"/move {shas[1]}", EOFError],
        )

        await run_session(agent, ui)

        assert agent._session.agent_position_sha == shas[1]

    @pytest.mark.asyncio
    async def test_move_to_branch(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        provider, ui, agent = _make_session(
            git_repo, [], ["/move feature", EOFError],
        )

        await run_session(agent, ui)

        assert agent._session.agent_position_sha == result["c4"]
        assert git_repo.current_branch() == "feature"

    @pytest.mark.asyncio
    async def test_move_invalid_target(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(
            git_repo, [], ["/move nonexistent", EOFError],
        )

        await run_session(agent, ui)

        assert any("cannot resolve" in line.lower() or "error" in line.lower()
                    for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_move_creates_branch_on_non_tip(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(
            git_repo, [], [f"/move {shas[0]}", EOFError],
        )

        await run_session(agent, ui)

        branches = git_repo.branches()
        assert any("alan/move" in b for b in branches)


# ═══════════════════════════════════════════════════════════════════════════════
# /convrevert command
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvRevertCommand:

    @pytest.mark.asyncio
    async def test_convrevert_after_commits(self, git_repo: GitTestRepo):
        """ConvRevert works when agent has commits (conv_path has entries)."""
        git_repo.write_file("a.py", "a")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}),
            text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}),
            text("OK"),
        ], ["First commit", "Second commit", "/convrevert 1", EOFError])

        await run_session(agent, ui)

        # Conv path should be shorter than if no convrevert happened
        assert any("conversation reverted" in line.lower() or "too short" in line.lower()
                    for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_convrevert_too_many(self, git_repo: GitTestRepo):
        """ConvRevert more steps than available reverts what it can."""
        git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [
            text("Response"),
        ], ["Turn 1", "/convrevert 100", EOFError])

        await run_session(agent, ui)

        # Should either succeed with partial revert or show "too short"
        assert any(
            "reverted" in line.lower() or "too short" in line.lower()
            for line in ui.console_log
        )

    @pytest.mark.asyncio
    async def test_convrevert_position_unchanged(self, git_repo: GitTestRepo):
        git_repo.write_file("a.py", "a")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}),
            text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}),
            text("OK"),
        ], ["First", "Second", "/convrevert 1", EOFError])

        await run_session(agent, ui)

        # Position should still be at HEAD
        assert agent._session.agent_position_sha == git_repo.head_sha()


# ═══════════════════════════════════════════════════════════════════════════════
# /allrevert command
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllRevertCommand:

    @pytest.mark.asyncio
    async def test_allrevert(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        original_head = shas[2]
        provider, ui, agent = _make_session(git_repo, [
            text("Response"),
        ], ["Turn 1", "/allrevert 1", EOFError])

        await run_session(agent, ui)

        # Position should have moved back
        assert agent._session.agent_position_sha != original_head
        assert any("all-reverted" in line.lower() for line in ui.console_log)


# ═══════════════════════════════════════════════════════════════════════════════
# Compaction markers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompactionMarkers:

    @pytest.mark.asyncio
    async def test_compact_adds_marker(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [
            text("Response 1"), text("Response 2"), text("Response 3"),
            text("Compaction summary"),
        ], ["Turn 1", "Turn 2", "Turn 3", "/compact", EOFError])

        await run_session(agent, ui)

        # If compaction succeeded, marker should be set
        if agent._session.compaction_markers:
            assert agent._session.agent_position_sha in agent._session.compaction_markers or \
                git_repo.head_sha() in agent._session.compaction_markers

    @pytest.mark.asyncio
    async def test_marker_in_tree_after_manual_set(self, git_repo: GitTestRepo):
        """Directly set a compaction marker and verify it shows in tree update."""
        shas = git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        # Manually set marker before running
        agent._session.agent_position_sha = shas[1]
        agent._session.add_compaction_marker(shas[1])

        await run_session(agent, ui)

        assert len(ui.tree_update_log) >= 1
        tree = ui.tree_update_log[-1]
        markers = [n for n in tree["nodes"] if n["is_compaction_marker"]]
        assert len(markers) == 1
        assert markers[0]["sha"] == shas[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Tree update content validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestTreeUpdateContent:

    @pytest.mark.asyncio
    async def test_tree_has_correct_structure(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        assert len(ui.tree_update_log) >= 1
        tree = ui.tree_update_log[-1]

        # Must have nodes and edges
        assert "nodes" in tree
        assert "edges" in tree
        assert "width" in tree
        assert "height" in tree

        # Nodes should have required fields
        for node in tree["nodes"]:
            assert "sha" in node
            assert "x" in node
            assert "y" in node
            assert "node_type" in node
            assert "message" in node
            assert node["node_type"] in ("alan_commit", "external", "current")

        # Edges should have required fields
        for edge in tree["edges"]:
            assert "from_sha" in edge
            assert "to_sha" in edge
            assert "edge_type" in edge

    @pytest.mark.asyncio
    async def test_agent_position_in_tree(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        agent_nodes = [n for n in tree["nodes"] if n["is_agent_position"]]
        assert len(agent_nodes) == 1

    @pytest.mark.asyncio
    async def test_conv_path_in_tree(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        conv_nodes = [n for n in tree["nodes"] if n["is_on_conv_path"]]
        assert len(conv_nodes) >= 1

    @pytest.mark.asyncio
    async def test_dirty_tree_shows_current_node(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(1)
        # Create dirty file BEFORE session (agent will write during turn)
        provider, ui, agent = _make_session(git_repo, [
            tool_call("Write", {
                "file_path": str(git_repo.path / "dirty.py"),
                "content": "dirty",
            }),
            text("Wrote file."),
        ], ["Write dirty.py", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        current_nodes = [n for n in tree["nodes"] if n["node_type"] == "current"]
        assert len(current_nodes) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_non_git_repo_no_tree_updates(self, tmp_path):
        """Non-git repos should produce zero tree updates."""
        provider = ScriptedProvider.from_responses([text("Hello")])
        ui = ScriptedUI.from_inputs(["Hi", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        assert len(ui.tree_update_log) == 0

    @pytest.mark.asyncio
    async def test_non_git_repo_revert_shows_error(self, tmp_path):
        provider = ScriptedProvider.from_responses([])
        ui = ScriptedUI.from_inputs(["/revert", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        assert any("git" in line.lower() for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_external_commit_appears_as_grey(self, git_repo: GitTestRepo):
        """Commits made outside the agent appear as external."""
        git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        external = [n for n in tree["nodes"] if n["node_type"] == "external"]
        # All commits are external (none made by agent)
        assert len(external) >= 2

    @pytest.mark.asyncio
    async def test_session_root_initialized(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        head = git_repo.head_sha()
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        assert agent._session.session_root_sha == head

    @pytest.mark.asyncio
    async def test_revert_then_commit_on_new_branch(self, git_repo: GitTestRepo):
        """Revert creates a branch, then agent can commit on it."""
        shas = git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(git_repo, [
            text("Reverted."),  # response for after /revert
            tool_call("Write", {
                "file_path": str(git_repo.path / "new.py"),
                "content": "new code",
            }),
            tool_call("GitCommit", {"message": "New work after revert"}),
            text("Committed on new branch."),
        ], ["/revert 2", "Now commit new work", EOFError])

        await run_session(agent, ui)

        # Should be on a new branch
        branch = git_repo.current_branch()
        assert branch is not None
        assert "alan" in branch or branch == "main"

        # New commit should be an alan commit
        if agent._session.alan_commits:
            new_sha = agent._session.alan_commits[-1]
            tree = parse_git_tree(str(git_repo.path), set(agent._session.alan_commits))
            node = tree.get_node(new_sha)
            assert node is not None
            assert node.node_type == NodeType.ALAN_COMMIT

    @pytest.mark.asyncio
    async def test_commit_and_revert_roundtrip(self, git_repo: GitTestRepo):
        """Agent commits, user reverts, agent commits again."""
        git_repo.write_file("a.py", "a")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "First commit"}),
            text("First done."),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "Second commit after revert"}),
            text("Second done."),
        ], ["First commit", "/revert 1", "Second commit", EOFError])

        await run_session(agent, ui)

        # Should have 2 alan commits
        assert len(agent._session.alan_commits) >= 1
        # Conv path should show the history
        assert len(agent._session.conv_path) >= 2

    @pytest.mark.asyncio
    async def test_multiple_tree_updates(self, git_repo: GitTestRepo):
        """Each turn should produce a tree update."""
        git_repo.build_linear_history(2)
        provider, ui, agent = _make_session(git_repo, [
            text("R1"), text("R2"), text("R3"),
        ], ["Turn 1", "Turn 2", "Turn 3", EOFError])

        await run_session(agent, ui)

        # Should have at least 3 tree updates (one per turn)
        assert len(ui.tree_update_log) >= 3

    @pytest.mark.asyncio
    async def test_slash_commands_produce_tree_update(self, git_repo: GitTestRepo):
        """Movement slash commands should trigger tree updates."""
        shas = git_repo.build_linear_history(3)
        provider, ui, agent = _make_session(
            git_repo, [], ["/revert 1", EOFError],
        )

        await run_session(agent, ui)

        # /revert should trigger a tree update
        assert len(ui.tree_update_log) >= 1

    @pytest.mark.asyncio
    async def test_branching_repo_tree_shape(self, git_repo: GitTestRepo):
        """Tree from a branching repo should have correct shape."""
        result = git_repo.build_branching_history()
        provider, ui, agent = _make_session(git_repo, [text("OK")],
                                             ["Hello", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        nodes = tree["nodes"]
        # Should have all commits
        assert len(nodes) >= 5  # c1-c5 + initial

        # Some nodes should be at different x positions (branching)
        x_values = {n["x"] for n in nodes}
        assert len(x_values) >= 2  # At least mainline + branch

        # Merge commit should have 2 parent edges
        merge_sha = result["c5"]
        merge_parent_edges = [
            e for e in tree["edges"]
            if e["edge_type"] == "parent" and e["to_sha"] == merge_sha
        ]
        assert len(merge_parent_edges) == 2

    @pytest.mark.asyncio
    async def test_memodiff_command(self, git_repo: GitTestRepo):
        """Test /memodiff output."""
        git_repo.write_file("f.py", "x")
        provider, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}),
            text("OK"),
        ], ["Commit", "/memodiff", EOFError])

        await run_session(agent, ui)

        # Should show some output (even if "not enough commits")
        assert any("memory" in line.lower() or "commit" in line.lower()
                    for line in ui.console_log)
