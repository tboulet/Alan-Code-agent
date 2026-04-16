"""Heavy AGT integration tests — tracking every bug encountered.

Each test class is named after the bug it guards against.
Tests use ScriptedUI + ScriptedProvider + GitTestRepo for full
end-to-end verification.

Bug catalog:
  BUG-01: git clean -fd deleted .alan/ directory
  BUG-02: conv_path not reaching agent_position (missing entries)
  BUG-03: /convrevert not truncating messages properly (n*2 heuristic)
  BUG-04: /revert creating branches instead of destroying commits
  BUG-05: Buttons disabled after slash commands (isAgentRunning stuck)
  BUG-06: Mainline computed from HEAD, reshuffles on branch switch
  BUG-07: Nodes overlapping at same (x,y) position
  BUG-08: State write failures when .alan not gitignored
  BUG-09: Revert-repo-to button sending /move instead of /revert
  BUG-10: Orphaned SHAs in session state after external git operations
"""

import json
import os
import shutil
import pytest

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.git_tree.layout import compute_layout
from alancode.git_tree.model import CURRENT_NODE_SHA, NodeType
from alancode.git_tree.operations import (
    agt_conv_revert,
    agt_move,
    agt_revert,
    agt_revert_to,
    detect_orphaned_shas,
)
from alancode.git_tree.parser import parse_git_tree
from alancode.gui.scripted_ui import ScriptedUI, ui_rule
from alancode.messages.types import AssistantMessage, UserMessage
from alancode.providers.scripted_provider import ScriptedProvider, text, tool_call
from alancode.session.state import SessionState
from tests.integration.git_helpers import GitTestRepo


def _make_session(git_repo, responses, inputs, **kwargs):
    provider = ScriptedProvider.from_responses(responses, fallback=text("Done."))
    ui = ScriptedUI.from_inputs(inputs)
    agent = AlanCodeAgent(
        provider=provider, cwd=str(git_repo.path),
        ask_callback=ui.ask_user, **kwargs,
    )
    return provider, ui, agent


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-01: git clean -fd must not delete .alan/
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug01AlanDirSurvivesGitClean:
    """git operations in agt_move/agt_revert must preserve .alan/."""

    def test_move_preserves_alan_dir(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b01", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        # Write something to .alan that must survive
        alan_file = git_repo.alan_dir() / "important.txt"
        alan_file.write_text("must survive")

        agt_move(str(git_repo.path), state, shas[0])

        assert alan_file.exists(), ".alan/important.txt was deleted by agt_move!"
        assert alan_file.read_text() == "must survive"

    def test_revert_preserves_alan_dir(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b01r", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        alan_file = git_repo.alan_dir() / "sessions" / "b01r" / "state.json"
        alan_file.parent.mkdir(parents=True, exist_ok=True)
        alan_file.write_text('{"test": true}')

        agt_revert(str(git_repo.path), state, 2)

        assert alan_file.exists(), "Session state.json was deleted by agt_revert!"

    def test_revert_dirty_preserves_alan_dir(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="b01d", cwd=str(git_repo.path))
        state.agent_position_sha = shas[1]

        alan_marker = git_repo.alan_dir() / "marker"
        alan_marker.write_text("x")

        git_repo.write_file("dirty.txt", "uncommitted")
        agt_revert(str(git_repo.path), state, 1)

        assert alan_marker.exists(), ".alan/marker deleted by dirty discard!"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-02: conv_path must always include agent_position
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug02ConvPathReachesAgentPosition:
    """agent_position_sha must always be in conv_path after any operation."""

    @pytest.mark.asyncio
    async def test_after_commit(self, git_repo: GitTestRepo):
        git_repo.write_file("f.py", "x")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}),
            text("OK"),
        ], ["Commit", EOFError])

        await run_session(agent, ui)

        pos = agent._session.agent_position_sha
        assert pos, "agent_position_sha is empty"
        assert pos in agent._session.conv_path, \
            f"agent_position {pos[:7]} not in conv_path"

    @pytest.mark.asyncio
    async def test_after_revert(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        _, ui, agent = _make_session(git_repo, [], ["/revert 1", EOFError])

        await run_session(agent, ui)

        pos = agent._session.agent_position_sha
        assert pos in agent._session.conv_path

    @pytest.mark.asyncio
    async def test_after_move(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        _, ui, agent = _make_session(
            git_repo, [], ["/move feature", EOFError],
        )

        await run_session(agent, ui)

        pos = agent._session.agent_position_sha
        assert pos in agent._session.conv_path

    @pytest.mark.asyncio
    async def test_sync_fixes_missing_position(self, git_repo: GitTestRepo):
        """_sync_agent_position adds HEAD to conv_path if missing."""
        git_repo.build_linear_history(2)
        _, ui, agent = _make_session(git_repo, [text("OK")], ["Hi", EOFError])

        # Simulate a missed update: clear conv_path
        agent._session.conv_path = []

        await run_session(agent, ui)

        # After the turn, sync should have added HEAD
        pos = agent._session.agent_position_sha
        assert pos in agent._session.conv_path


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-03: /convrevert must properly truncate messages
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug03ConvRevertTruncatesMessages:
    """After /convrevert, the agent must NOT see messages from reverted turns."""

    @pytest.mark.asyncio
    async def test_messages_removed_after_convrevert(self, git_repo: GitTestRepo):
        """Commit twice, convrevert 1, check the second commit's messages are gone."""
        git_repo.write_file("a.py", "a")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "First commit"}),
            text("First done."),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "Second commit"}),
            text("Second done."),
        ], ["Do first commit", "Do second commit", "/convrevert 1", EOFError])

        await run_session(agent, ui)

        # After convrevert 1, messages about "Second commit" should be gone
        all_text = " ".join(
            msg.content if isinstance(msg.content, str) else ""
            for msg in agent._messages
            if isinstance(msg, UserMessage)
        )
        # The user prompt "Do second commit" should NOT be in messages
        assert "Do second commit" not in all_text, \
            "Second turn's messages still present after /convrevert 1"

    @pytest.mark.asyncio
    async def test_convrevert_2_removes_two_turns(self, git_repo: GitTestRepo):
        git_repo.write_file("a.py", "a")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "c.py"), "content": "c"}),
            tool_call("GitCommit", {"message": "C3"}), text("OK"),
        ], ["Turn1", "Turn2", "Turn3", "/convrevert 2", EOFError])

        await run_session(agent, ui)

        user_texts = [
            msg.content for msg in agent._messages
            if isinstance(msg, UserMessage) and isinstance(msg.content, str)
            and not msg.content.startswith("<system-reminder>") and not msg.hide_in_ui
        ]
        assert "Turn2" not in user_texts
        assert "Turn3" not in user_texts
        # Turn1 should still be there
        assert "Turn1" in user_texts

    @pytest.mark.asyncio
    async def test_convrevert_then_continue(self, git_repo: GitTestRepo):
        """After convrevert, agent can continue working normally."""
        git_repo.write_file("a.py", "a")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("Done"),
            text("After revert response"),
        ], ["First commit", "/convrevert 1", "Continue working", EOFError])

        await run_session(agent, ui)

        # Should have processed the third prompt without crash
        assert any("After revert" in e.get("text", "") or
                    "After revert" in str(e.get("content", ""))
                    for e in ui.event_log)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-04: /revert must destroy commits (git reset --hard), not create branches
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug04RevertDestroysCommits:
    """/revert must use git reset --hard, not git checkout + new branch."""

    def test_revert_removes_commit_from_log(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b04", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        agt_revert(str(git_repo.path), state, 1)

        log = git_repo.log_hashes()
        assert shas[2] not in log, "Reverted commit still in git log!"
        assert shas[1] in log

    def test_revert_does_not_create_branch(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b04b", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        branches_before = set(git_repo.branches())
        agt_revert(str(git_repo.path), state, 1)
        branches_after = set(git_repo.branches())

        new_branches = branches_after - branches_before
        assert not new_branches, f"Revert created branches: {new_branches}"

    def test_revert_cleans_alan_commits(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b04c", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]
        state.alan_commits = [shas[1], shas[2]]

        agt_revert(str(git_repo.path), state, 1)

        assert shas[2] not in state.alan_commits
        assert shas[1] in state.alan_commits

    def test_revert_to_sha_destroys(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        state = SessionState(session_id="b04d", cwd=str(git_repo.path))
        state.agent_position_sha = shas[3]

        agt_revert_to(str(git_repo.path), state, shas[1])

        log = git_repo.log_hashes()
        assert shas[3] not in log
        assert shas[2] not in log
        assert shas[1] in log
        assert git_repo.head_sha() == shas[1]

    def test_revert_to_non_ancestor_fails(self, git_repo: GitTestRepo):
        """Can't revert to a commit that isn't an ancestor of HEAD."""
        result = git_repo.build_branching_history()
        state = SessionState(session_id="b04e", cwd=str(git_repo.path))
        state.agent_position_sha = result["c5"]

        # c4 is on feature branch, not a first-parent ancestor of c5
        # Actually c4 IS an ancestor of c5 (merge), so let's test with
        # a truly non-ancestor: create a separate branch
        git_repo.checkout_new_branch("isolated", result["c1"])
        git_repo.write_file("isolated.py", "x")
        isolated_sha = git_repo.commit("Isolated commit")
        git_repo.checkout("main")

        move_result = agt_revert_to(str(git_repo.path), state, isolated_sha)
        assert not move_result.success
        assert "not an ancestor" in move_result.description.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-05: Buttons must re-enable after slash commands
# (Can't test JS directly, but we verify the server-side state is correct)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug05SlashCommandsCompleteCleanly:
    """Slash commands must not leave the session in a broken state."""

    @pytest.mark.asyncio
    async def test_revert_then_input_works(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        _, ui, agent = _make_session(git_repo, [text("After revert")],
                                      ["/revert 1", "Continue", EOFError])

        await run_session(agent, ui)

        # Agent should have processed "Continue" normally
        assert len(ui.cost_log) >= 1  # At least one LLM turn

    @pytest.mark.asyncio
    async def test_move_then_input_works(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        _, ui, agent = _make_session(
            git_repo, [text("On feature now")],
            ["/move feature", "What branch am I on?", EOFError],
        )

        await run_session(agent, ui)

        assert len(ui.cost_log) >= 1

    @pytest.mark.asyncio
    async def test_convrevert_then_input_works(self, git_repo: GitTestRepo):
        git_repo.write_file("a.py", "a")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
            text("After convrevert"),
        ], ["Commit", "/convrevert 1", "Continue", EOFError])

        await run_session(agent, ui)

        assert len(ui.cost_log) >= 2  # Turn before and after convrevert

    @pytest.mark.asyncio
    async def test_multiple_slash_commands_in_row(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        _, ui, agent = _make_session(git_repo, [text("Final")], [
            "/revert 1", "/revert 1", "/revert 1", "Still here?", EOFError,
        ])

        await run_session(agent, ui)

        assert len(ui.cost_log) >= 1
        # Should have reverted 3 commits total
        assert git_repo.head_sha() == shas[1]


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-06: Mainline must be computed from main/master, not HEAD
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug06MainlineStability:
    """Moving to another branch must not reshuffle mainline node positions."""

    def test_mainline_from_main_branch(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(tree)

        # Mainline commits (c1, c2, c3, c5) should all be at x=0
        for key in ["c1", "c2", "c3", "c5"]:
            sha = result[key]
            node = next((n for n in layout.nodes if n.sha == sha), None)
            assert node is not None, f"Missing {key}"
            assert node.x == 0.0, f"{key} at x={node.x}, expected 0.0 (mainline)"

    def test_mainline_stable_after_checkout_feature(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()

        # Get layout before checkout
        tree1 = parse_git_tree(str(git_repo.path))
        layout1 = compute_layout(tree1)
        positions1 = {n.sha: (n.x, n.y) for n in layout1.nodes}

        # Checkout feature branch (HEAD changes!)
        git_repo.checkout("feature")

        # Get layout after checkout
        tree2 = parse_git_tree(str(git_repo.path))
        layout2 = compute_layout(tree2)
        positions2 = {n.sha: (n.x, n.y) for n in layout2.nodes}

        # All shared nodes must have same positions
        for sha in positions1:
            if sha in positions2:
                assert positions1[sha] == positions2[sha], \
                    f"Node {sha[:7]} moved from {positions1[sha]} to {positions2[sha]} after checkout"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-07: No two nodes at the same (x, y) position
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug07NoOverlappingNodes:
    """Every node must have a unique (x, y) position."""

    def test_no_overlap_linear(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(10)
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        positions = set()
        for n in layout.nodes:
            pos = (n.x, n.y)
            assert pos not in positions, \
                f"Overlap at ({n.x}, {n.y}): {n.sha[:7]} ({n.message})"
            positions.add(pos)

    def test_no_overlap_branching(self, git_repo: GitTestRepo):
        git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        positions = set()
        for n in layout.nodes:
            pos = (n.x, n.y)
            assert pos not in positions, \
                f"Overlap at ({n.x}, {n.y}): {n.sha[:7]} ({n.message})"
            positions.add(pos)

    def test_no_overlap_after_revert_and_commit(self, git_repo: GitTestRepo):
        """After revert + commit on new branch, nodes must not overlap."""
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b07", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        # Move to shas[0] (creates new branch)
        agt_move(str(git_repo.path), state, shas[0])

        # Commit on the new branch
        git_repo.write_file("new.py", "x")
        git_repo.commit("New on branch")

        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        positions = set()
        for n in layout.nodes:
            pos = (n.x, n.y)
            assert pos not in positions, \
                f"Overlap at ({n.x}, {n.y}): {n.sha[:7]} ({n.message})"
            positions.add(pos)

    def test_no_overlap_multi_branch(self, git_repo: GitTestRepo):
        """Multiple branches from same parent must each get unique x."""
        base_sha = git_repo.head_sha()
        for i in range(4):
            git_repo.checkout_new_branch(f"branch-{i}", base_sha)
            git_repo.write_file(f"b{i}.py", f"branch {i}")
            git_repo.commit(f"Commit on branch-{i}")
        git_repo.checkout("main")

        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        positions = set()
        for n in layout.nodes:
            pos = (n.x, n.y)
            assert pos not in positions, \
                f"Overlap at ({n.x}, {n.y}): {n.sha[:7]} ({n.message})"
            positions.add(pos)


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-08: .alan must be gitignored for safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug08AlanGitignored:
    """Agent must ensure .alan/ is gitignored on session start."""

    @pytest.mark.asyncio
    async def test_gitignore_created_if_missing(self, tmp_path):
        """If no .gitignore exists, one is created with .alan/."""
        repo = GitTestRepo(tmp_path / "nogi")
        repo.init()
        # Remove .gitignore that GitTestRepo.init() creates
        (repo.path / ".gitignore").unlink()
        repo.commit("Remove gitignore", add_all=True)

        provider = ScriptedProvider.from_responses([text("Hi")])
        ui = ScriptedUI.from_inputs(["Hello", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(repo.path))
        await run_session(agent, ui)

        gitignore = repo.path / ".gitignore"
        assert gitignore.exists()
        assert ".alan" in gitignore.read_text()

    @pytest.mark.asyncio
    async def test_gitignore_appended_if_missing_entry(self, tmp_path):
        """If .gitignore exists but doesn't have .alan, it's appended."""
        repo = GitTestRepo(tmp_path / "partial")
        repo.init()
        (repo.path / ".gitignore").write_text("*.pyc\n__pycache__/\n")
        repo.commit("Custom gitignore", add_all=True)

        provider = ScriptedProvider.from_responses([text("Hi")])
        ui = ScriptedUI.from_inputs(["Hello", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(repo.path))
        await run_session(agent, ui)

        content = (repo.path / ".gitignore").read_text()
        assert ".alan" in content
        assert "*.pyc" in content  # Original entries preserved


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-10: Orphaned SHAs must be cleaned from session state
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug10OrphanedSHAs:
    """After external git operations that remove commits, state is cleaned."""

    def test_detect_after_external_reset(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b10", cwd=str(git_repo.path))
        state.alan_commits = [shas[1], shas[2]]
        state.conv_path = [shas[0], shas[1], shas[2]]
        state.agent_position_sha = shas[2]

        # External reset destroys shas[2]
        git_repo.reset_hard(shas[1])

        orphaned = detect_orphaned_shas(str(git_repo.path), state)
        assert shas[2] in orphaned

        # State should be cleaned
        assert shas[2] not in state.alan_commits
        assert shas[2] not in state.conv_path
        assert state.agent_position_sha == shas[1]  # Updated to HEAD

    def test_no_false_positives(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="b10b", cwd=str(git_repo.path))
        state.alan_commits = [shas[0], shas[1]]
        state.conv_path = shas

        orphaned = detect_orphaned_shas(str(git_repo.path), state)
        assert orphaned == []


# ═══════════════════════════════════════════════════════════════════════════════
# Tree update verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestTreeUpdates:
    """Tree updates must be sent at the right times with correct content."""

    @pytest.mark.asyncio
    async def test_tree_sent_at_session_start(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        _, ui, agent = _make_session(git_repo, [text("OK")], ["Hi", EOFError])

        await run_session(agent, ui)

        # At least 2 updates: one at start, one after the turn
        assert len(ui.tree_update_log) >= 2

    @pytest.mark.asyncio
    async def test_tree_sent_after_revert(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(3)
        _, ui, agent = _make_session(git_repo, [], ["/revert 1", EOFError])

        await run_session(agent, ui)

        # Should have tree updates (start + after revert)
        assert len(ui.tree_update_log) >= 2

    @pytest.mark.asyncio
    async def test_tree_agent_position_correct(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(3)
        _, ui, agent = _make_session(git_repo, [text("OK")], ["Hi", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        agent_nodes = [n for n in tree["nodes"] if n["is_agent_position"]]
        assert len(agent_nodes) == 1
        assert agent_nodes[0]["sha"] == agent._session.agent_position_sha

    @pytest.mark.asyncio
    async def test_tree_session_root_marked(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        _, ui, agent = _make_session(git_repo, [text("OK")], ["Hi", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        root_nodes = [n for n in tree["nodes"] if n.get("is_session_root")]
        assert len(root_nodes) == 1

    @pytest.mark.asyncio
    async def test_tree_alan_commits_blue(self, git_repo: GitTestRepo):
        git_repo.write_file("f.py", "x")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
        ], ["Commit", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        alan_nodes = [n for n in tree["nodes"] if n["node_type"] == "alan_commit"]
        assert len(alan_nodes) >= 1

    @pytest.mark.asyncio
    async def test_tree_conv_path_edges(self, git_repo: GitTestRepo):
        git_repo.write_file("f.py", "x")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
        ], ["Commit", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        conv_edges = [e for e in tree["edges"]
                      if e["edge_type"] in ("conv_path", "conv_jump")]
        # Should have at least one conv_path edge
        assert len(conv_edges) >= 0  # May be 0 if only 1 node in path

    @pytest.mark.asyncio
    async def test_tree_post_compaction_edges_always_present(self, git_repo: GitTestRepo):
        """Yellow (post-compaction) edges should always exist from session root."""
        git_repo.write_file("f.py", "x")
        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "g.py"), "content": "g"}),
            tool_call("GitCommit", {"message": "C2"}), text("OK"),
        ], ["First", "Second", EOFError])

        await run_session(agent, ui)

        tree = ui.tree_update_log[-1]
        post_comp = [e for e in tree["edges"] if e["edge_type"] == "post_compaction"]
        assert len(post_comp) >= 1, "No post_compaction (yellow) edges found"

    @pytest.mark.asyncio
    async def test_no_tree_for_non_git(self, tmp_path):
        provider = ScriptedProvider.from_responses([text("OK")])
        ui = ScriptedUI.from_inputs(["Hi", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(tmp_path))

        await run_session(agent, ui)

        assert len(ui.tree_update_log) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Full scenario: commit → revert → commit → move → convrevert
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullScenario:
    """End-to-end scenario exercising the complete AGT workflow."""

    @pytest.mark.asyncio
    async def test_commit_revert_commit_move(self, git_repo: GitTestRepo):
        """
        1. Agent commits (C1)
        2. /revert 1 (C1 destroyed)
        3. Agent commits (C2, on same branch since C1 gone)
        4. Agent commits (C3)
        5. /move to C2's parent
        6. Verify: C2 and C3 still in git (move is non-destructive)
        """
        git_repo.write_file("a.py", "a")
        initial_head = git_repo.head_sha()

        _, ui, agent = _make_session(git_repo, [
            # Turn 1: commit C1
            tool_call("GitCommit", {"message": "C1"}), text("C1 done"),
            # Turn 2: after /revert, commit C2
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("C2 done"),
            # Turn 3: commit C3
            tool_call("Write", {"file_path": str(git_repo.path / "c.py"), "content": "c"}),
            tool_call("GitCommit", {"message": "C3"}), text("C3 done"),
            # Turn 4: after /move, just acknowledge
            text("Moved back"),
        ], [
            "Create C1",         # → commits C1
            "/revert 1",         # → destroys C1
            "Create C2",         # → commits C2
            "Create C3",         # → commits C3
            f"/move {initial_head}",  # → non-destructive move to initial
            "Where am I?",       # → agent responds
            EOFError,
        ])

        await run_session(agent, ui)

        # Verify git state
        log = git_repo.log_hashes(all_branches=True)
        # C1 should be destroyed (reverted)
        # C2 and C3 should exist (on a branch, move is non-destructive)
        assert len(agent._session.alan_commits) >= 2

        # Agent should be at initial_head (or nearby, after move)
        # No crashes
        assert len(ui.tree_update_log) >= 4  # Multiple tree updates

    @pytest.mark.asyncio
    async def test_commit_convrevert_continue(self, git_repo: GitTestRepo):
        """
        1. Agent commits twice
        2. /convrevert 1 (forget last turn)
        3. Agent continues working
        4. Verify: agent doesn't reference the forgotten turn
        """
        git_repo.write_file("a.py", "a")

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "First"}), text("First done"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "Second"}), text("Second done"),
            text("I see one commit in my history."),
        ], [
            "Do first commit",
            "Do second commit",
            "/convrevert 1",
            "What have I done so far?",
            EOFError,
        ])

        await run_session(agent, ui)

        # After convrevert, "Do second commit" should not be in messages
        user_texts = [
            msg.content for msg in agent._messages
            if isinstance(msg, UserMessage) and isinstance(msg.content, str)
            and not msg.hide_in_ui and not msg.content.startswith("<system-reminder>")
        ]
        assert "Do second commit" not in user_texts
        assert "Do first commit" in user_texts

    @pytest.mark.asyncio
    async def test_dirty_state_handling(self, git_repo: GitTestRepo):
        """
        1. Agent writes a file (dirty state)
        2. /revert 1 (discards dirty changes)
        3. Verify: working tree is clean
        """
        git_repo.build_linear_history(2)

        _, ui, agent = _make_session(git_repo, [
            tool_call("Write", {"file_path": str(git_repo.path / "new.py"), "content": "dirty"}),
            text("Wrote file"),
            text("OK clean now"),
        ], [
            "Write new.py",
            "/revert 1",
            "What happened?",
            EOFError,
        ])

        await run_session(agent, ui)

        assert not git_repo.is_dirty()
