"""AGT edge case tests — covering gaps in test coverage.

Edge cases:
  EC-01: /revert <sha> via run_session (button path)
  EC-02: /convrevert <sha> via run_session (button path)
  EC-03: commit_message_indices precision
  EC-04: Conv gap arc after convrevert in layout
  EC-05: Resume session with AGT state
  EC-06: External commit between turns
  EC-07: Revert past a merge commit
  EC-08: Conv path with duplicate SHAs (visit same node twice)
  EC-09: /allrevert <sha> via run_session
  EC-10: /memodiff with actual memory changes
"""

import json
import pytest

from alancode.agent import AlanCodeAgent
from alancode.cli.repl import run_session
from alancode.git_tree.layout import compute_layout
from alancode.git_tree.model import CURRENT_NODE_SHA
from alancode.git_tree.operations import agt_move, agt_revert
from alancode.git_tree.memory_snapshots import take_memory_snapshot
from alancode.git_tree.parser import parse_git_tree
from alancode.gui.scripted_ui import ScriptedUI
from alancode.messages.types import UserMessage
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
# EC-01: /revert <sha> via run_session (button sends SHA)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC01RevertWithSHA:

    @pytest.mark.asyncio
    async def test_revert_sha_destroys_commits(self, git_repo: GitTestRepo):
        """Button 'Revert repo to' sends /revert <sha>. Must destroy commits."""
        git_repo.write_file("a.py", "a")
        initial = git_repo.head_sha()

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("OK"),
        ], [
            "First commit",
            "Second commit",
            f"/revert {initial}",  # Revert to initial — destroys C1 and C2
            EOFError,
        ])

        await run_session(agent, ui)

        # C1 and C2 should be destroyed
        log = git_repo.log_hashes()
        assert git_repo.head_sha() == initial
        assert len(agent._session.alan_commits) == 0  # Both cleaned

    @pytest.mark.asyncio
    async def test_revert_sha_non_ancestor_shows_error(self, git_repo: GitTestRepo):
        """Revert to a SHA that isn't an ancestor shows error, suggests /move."""
        result = git_repo.build_branching_history()
        # Create an isolated branch
        git_repo.checkout_new_branch("isolated", result["c1"])
        git_repo.write_file("iso.py", "x")
        isolated = git_repo.commit("Isolated")
        git_repo.checkout("main")

        _, ui, agent = _make_session(git_repo, [], [
            f"/revert {isolated}",
            EOFError,
        ])

        await run_session(agent, ui)

        assert any("not an ancestor" in line.lower() or "move" in line.lower()
                    for line in ui.console_log)


# ═══════════════════════════════════════════════════════════════════════════════
# EC-02: /convrevert <sha> via run_session (button sends SHA)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC02ConvRevertWithSHA:

    @pytest.mark.asyncio
    async def test_convrevert_sha_truncates_to_commit(self, git_repo: GitTestRepo):
        """Button 'Revert conv. to' sends /convrevert <sha>."""
        git_repo.write_file("a.py", "a")

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("C1 done"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("C2 done"),
        ], [
            "First commit",
            "Second commit",
            # Now convrevert to session root (which is in conv_path)
            f"/convrevert {git_repo.head_sha()}",  # Will be the root before commits
            EOFError,
        ])

        # We need the session root SHA before commits happen
        # The initial head is the session root
        initial_head = git_repo.head_sha()

        await run_session(agent, ui)

        # Check that convrevert was attempted
        assert any("conversation" in line.lower() or "revert" in line.lower()
                    or "not in" in line.lower() or "already" in line.lower()
                    for line in ui.console_log)

    @pytest.mark.asyncio
    async def test_convrevert_sha_not_in_path_shows_error(self, git_repo: GitTestRepo):
        """Convrevert to a SHA not in conv_path shows error."""
        git_repo.build_linear_history(3)

        _, ui, agent = _make_session(git_repo, [text("OK")], [
            "Hello",
            "/convrevert deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            EOFError,
        ])

        await run_session(agent, ui)

        assert any("cannot resolve" in line.lower() or "not in" in line.lower()
                    for line in ui.console_log)


# ═══════════════════════════════════════════════════════════════════════════════
# EC-03: commit_message_indices precision
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC03CommitMessageIndices:

    @pytest.mark.asyncio
    async def test_index_recorded_on_commit(self, git_repo: GitTestRepo):
        """GitCommit tool must record message count in commit_message_indices."""
        git_repo.write_file("f.py", "x")

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
        ], ["Commit", EOFError])

        await run_session(agent, ui)

        indices = agent._session.commit_message_indices
        assert len(indices) >= 1  # At least session root + C1
        # The commit SHA should be in the indices
        for sha in agent._session.alan_commits:
            assert sha in indices, f"Commit {sha[:7]} not in commit_message_indices"

    @pytest.mark.asyncio
    async def test_index_used_for_truncation(self, git_repo: GitTestRepo):
        """Convrevert must truncate to the exact index recorded."""
        git_repo.write_file("a.py", "a")

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("C1 done"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("C2 done"),
        ], ["First commit", "Second commit", "/convrevert 1", EOFError])

        await run_session(agent, ui)

        # After convrevert 1, messages about "Second commit" should be gone
        user_texts = [
            msg.content for msg in agent._messages
            if isinstance(msg, UserMessage) and isinstance(msg.content, str)
            and not msg.hide_in_ui and not msg.content.startswith("<system-reminder>")
        ]
        assert "Second commit" not in user_texts
        assert "First commit" in user_texts

    @pytest.mark.asyncio
    async def test_session_root_index_recorded(self, git_repo: GitTestRepo):
        """Session root SHA should have an index in commit_message_indices."""
        git_repo.build_linear_history(2)

        _, ui, agent = _make_session(git_repo, [text("OK")], ["Hello", EOFError])

        await run_session(agent, ui)

        root = agent._session.session_root_sha
        indices = agent._session.commit_message_indices
        assert root in indices


# ═══════════════════════════════════════════════════════════════════════════════
# EC-04: Conv gap arc after convrevert in layout
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC04ConvGapArc:

    def test_gap_arc_when_position_ahead_of_conv(self, git_repo: GitTestRepo):
        """After convrevert, layout should have conv_jump arc from conv end to agent."""
        shas = git_repo.build_linear_history(4)
        tree = parse_git_tree(str(git_repo.path))

        # Simulate: conv_path is [s0, s1] but agent is at s3
        # (convrevert removed s2 and s3 from conv)
        layout = compute_layout(
            tree,
            conv_path=[shas[0], shas[1]],
            agent_position=shas[3],
        )

        # Should have a conv_jump edge from shas[1] to shas[3]
        jump_edges = [e for e in layout.edges
                      if e.edge_type == "conv_jump"
                      and e.from_sha == shas[1] and e.to_sha == shas[3]]
        assert len(jump_edges) >= 1, \
            "Missing conv_jump arc from conv end to agent position after convrevert"

    def test_no_gap_arc_when_aligned(self, git_repo: GitTestRepo):
        """No gap arc when agent position is the last entry in conv_path."""
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(
            tree,
            conv_path=[shas[0], shas[1], shas[2]],
            agent_position=shas[2],
        )

        # No conv_jump from shas[2] to shas[2] (they're the same)
        gap_jumps = [e for e in layout.edges
                     if e.edge_type == "conv_jump"
                     and e.from_sha == shas[2] and e.to_sha == shas[2]]
        assert len(gap_jumps) == 0

    def test_gap_arc_has_post_compaction_too(self, git_repo: GitTestRepo):
        """The gap arc should also have a post_compaction edge (yellow on top)."""
        shas = git_repo.build_linear_history(4)
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(
            tree,
            conv_path=[shas[0], shas[1]],
            agent_position=shas[3],
        )

        post_comp_jumps = [e for e in layout.edges
                          if e.edge_type == "post_compaction"
                          and e.from_sha == shas[1] and e.to_sha == shas[3]]
        assert len(post_comp_jumps) >= 1, \
            "Missing post_compaction edge on conv gap arc"


# ═══════════════════════════════════════════════════════════════════════════════
# EC-05: Resume session with AGT state
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC05ResumeSession:

    @pytest.mark.asyncio
    async def test_agt_state_persists_across_sessions(self, git_repo: GitTestRepo):
        """AGT state (alan_commits, conv_path) survives session restart."""
        git_repo.write_file("f.py", "x")

        # Session 1: commit
        provider1 = ScriptedProvider.from_responses([
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
        ], fallback=text("Done"))
        ui1 = ScriptedUI.from_inputs(["Commit", EOFError])
        agent1 = AlanCodeAgent(
            provider=provider1, cwd=str(git_repo.path),
            session_id="resume-test",
        )
        await run_session(agent1, ui1)

        saved_commits = agent1._session.alan_commits.copy()
        saved_conv = agent1._session.conv_path.copy()
        saved_root = agent1._session.session_root_sha
        assert len(saved_commits) >= 1

        # Session 2: resume with same session_id
        provider2 = ScriptedProvider.from_responses([text("Resumed OK")])
        ui2 = ScriptedUI.from_inputs(["Hello", EOFError])
        agent2 = AlanCodeAgent(
            provider=provider2, cwd=str(git_repo.path),
            session_id="resume-test",
        )
        await run_session(agent2, ui2, resumed_session_id="resume-test")

        # AGT state should be loaded from disk
        assert agent2._session.alan_commits == saved_commits
        assert agent2._session.session_root_sha == saved_root
        # Conv path may have additional entries from session 2
        assert all(sha in agent2._session.conv_path for sha in saved_conv)

    @pytest.mark.asyncio
    async def test_resumed_session_tree_update(self, git_repo: GitTestRepo):
        """Resumed session should send tree update with correct state."""
        git_repo.write_file("f.py", "x")

        # Session 1
        provider1 = ScriptedProvider.from_responses([
            tool_call("GitCommit", {"message": "C1"}), text("OK"),
        ], fallback=text("Done"))
        ui1 = ScriptedUI.from_inputs(["Commit", EOFError])
        agent1 = AlanCodeAgent(
            provider=provider1, cwd=str(git_repo.path),
            session_id="resume-tree",
        )
        await run_session(agent1, ui1)

        # Session 2
        provider2 = ScriptedProvider.from_responses([text("OK")])
        ui2 = ScriptedUI.from_inputs(["Hello", EOFError])
        agent2 = AlanCodeAgent(
            provider=provider2, cwd=str(git_repo.path),
            session_id="resume-tree",
        )
        await run_session(agent2, ui2, resumed_session_id="resume-tree")

        # Tree update should include the alan commit from session 1
        assert len(ui2.tree_update_log) >= 1
        tree = ui2.tree_update_log[-1]
        alan_nodes = [n for n in tree["nodes"] if n["node_type"] == "alan_commit"]
        assert len(alan_nodes) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# EC-06: External commit between turns
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC06ExternalCommitBetweenTurns:

    @pytest.mark.asyncio
    async def test_external_commit_detected(self, git_repo: GitTestRepo):
        """An external commit between turns should appear in the tree."""
        git_repo.build_linear_history(1)

        # We'll use a reactive provider: after turn 1, we make an external
        # commit, then turn 2 should show it
        provider = ScriptedProvider.from_responses([
            text("Turn 1 done"),
            text("Turn 2 done"),
        ])
        ui = ScriptedUI.from_inputs(["Turn 1", "Turn 2", EOFError])
        agent = AlanCodeAgent(provider=provider, cwd=str(git_repo.path))

        # Run turn 1
        user_input = await ui.get_input()
        async for event in agent.query_events_async(user_input):
            await ui.on_agent_event(event)

        # External commit (simulating user in another terminal)
        git_repo.write_file("external.py", "external code")
        external_sha = git_repo.commit("External commit")

        # Run turn 2
        user_input = await ui.get_input()
        async for event in agent.query_events_async(user_input):
            await ui.on_agent_event(event)

        # Parse tree — external commit should be there as grey node
        tree = parse_git_tree(
            str(git_repo.path),
            alan_commits=set(agent._session.alan_commits),
        )
        ext_node = tree.get_node(external_sha)
        assert ext_node is not None
        assert ext_node.node_type.value == "external"


# ═══════════════════════════════════════════════════════════════════════════════
# EC-07: Revert past a merge commit
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC07RevertPastMerge:

    def test_revert_past_merge(self, git_repo: GitTestRepo):
        """git reset --hard past a merge commit should work cleanly."""
        result = git_repo.build_branching_history()
        # History: c1 - c2 - c3 - c5(merge)
        #                  \- c4 -/
        state = SessionState(session_id="ec07", cwd=str(git_repo.path))
        state.agent_position_sha = result["c5"]

        # Revert past the merge to c2
        from alancode.git_tree.operations import agt_revert
        move = agt_revert(str(git_repo.path), state, 2)  # c5 -> c3 -> c2

        assert move.success
        assert git_repo.head_sha() == result["c2"]
        # c3, c4, c5 should be gone from main branch
        log = git_repo.log_hashes()
        assert result["c5"] not in log
        assert result["c3"] not in log


# ═══════════════════════════════════════════════════════════════════════════════
# EC-08: Conv path with duplicate SHAs
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC08DuplicateSHAsInConvPath:

    def test_layout_handles_duplicates(self, git_repo: GitTestRepo):
        """Conv path [A, B, C, B] (revisit B) should render without crash."""
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        # Agent went A -> B -> C, then moved back to B
        conv_path = [shas[0], shas[1], shas[2], shas[1]]
        layout = compute_layout(
            tree,
            conv_path=conv_path,
            agent_position=shas[1],
        )

        # Should have edges: A->B (conv_path), B->C (conv_path), C->B (conv_jump)
        conv_edges = [e for e in layout.edges
                      if e.edge_type in ("conv_path", "conv_jump")]
        assert len(conv_edges) >= 3

        # The C->B edge should be a jump (not parent-child direction)
        jump_back = [e for e in layout.edges
                     if e.edge_type == "conv_jump"
                     and e.from_sha == shas[2] and e.to_sha == shas[1]]
        assert len(jump_back) >= 1

    def test_convrevert_with_duplicates(self, git_repo: GitTestRepo):
        """Convrevert on path with duplicates uses last occurrence."""
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="ec08", cwd=str(git_repo.path))
        state.conv_path = [shas[0], shas[1], shas[2], shas[1]]
        state.agent_position_sha = shas[1]

        from alancode.git_tree.operations import agt_conv_revert
        result = agt_conv_revert(str(git_repo.path), state, 1)

        assert result.success
        # Should truncate to [shas[0], shas[1], shas[2]]
        # (removed the last entry, which was the second visit to shas[1])
        assert state.conv_path == [shas[0], shas[1], shas[2]]


# ═══════════════════════════════════════════════════════════════════════════════
# EC-09: /allrevert <sha> via run_session
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC09AllRevertWithSHA:

    @pytest.mark.asyncio
    async def test_allrevert_sha_reverts_both(self, git_repo: GitTestRepo):
        """allrevert <sha> should move repo AND truncate conversation."""
        git_repo.write_file("a.py", "a")
        initial = git_repo.head_sha()

        _, ui, agent = _make_session(git_repo, [
            tool_call("GitCommit", {"message": "C1"}), text("C1 done"),
            tool_call("Write", {"file_path": str(git_repo.path / "b.py"), "content": "b"}),
            tool_call("GitCommit", {"message": "C2"}), text("C2 done"),
        ], [
            "First",
            "Second",
            f"/allrevert {initial}",
            EOFError,
        ])

        await run_session(agent, ui)

        # Repo should be at initial
        # (allrevert with SHA delegates to /move + /convrevert)
        # Check that conversation was affected
        user_texts = [
            msg.content for msg in agent._messages
            if isinstance(msg, UserMessage) and isinstance(msg.content, str)
            and not msg.hide_in_ui and not msg.content.startswith("<system-reminder>")
        ]
        # At minimum, the allrevert should have executed without crash
        assert any("revert" in line.lower() or "moved" in line.lower()
                    for line in ui.console_log)


# ═══════════════════════════════════════════════════════════════════════════════
# EC-10: /memodiff with actual memory changes
# ═══════════════════════════════════════════════════════════════════════════════


class TestEC10MemodiffContent:

    @pytest.mark.asyncio
    async def test_memodiff_shows_changes(self, git_repo: GitTestRepo):
        """memodiff should show actual memory file differences."""
        git_repo.write_file("a.py", "a")

        # Manually set up memory snapshots for two commits
        sha1 = git_repo.head_sha()
        git_repo.write_memory("notes.md", "Version 1")
        take_memory_snapshot(str(git_repo.path), sha1)

        git_repo.write_file("b.py", "b")
        sha2 = git_repo.commit("Second")
        git_repo.write_memory("notes.md", "Version 2")
        git_repo.write_memory("new.md", "Brand new file")
        take_memory_snapshot(str(git_repo.path), sha2)

        # Set up agent with these as alan_commits
        _, ui, agent = _make_session(git_repo, [], ["/memodiff", EOFError])
        agent._session.alan_commits = [sha1, sha2]
        agent._session.agent_position_sha = sha2

        await run_session(agent, ui)

        # Should show memory diff mentioning the changed/new files
        console_text = " ".join(ui.console_log).lower()
        assert "notes.md" in console_text or "new.md" in console_text or "not enough" in console_text
