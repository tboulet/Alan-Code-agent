"""Tests for AGT Phase 5: Movement operations and memory snapshots."""

import pytest

from alancode.git_tree.memory_snapshots import (
    get_memory_diff,
    restore_memory_snapshot,
    take_memory_snapshot,
)
from alancode.git_tree.operations import (
    agt_all_revert,
    agt_conv_revert,
    agt_move,
    agt_revert,
    detect_orphaned_shas,
)
from alancode.session.state import SessionState
from tests.integration.git_helpers import GitTestRepo


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Snapshots
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemorySnapshots:

    def test_take_snapshot(self, git_repo: GitTestRepo):
        git_repo.write_memory("notes.md", "# Notes\nStuff")
        sha = git_repo.head_sha()
        path = take_memory_snapshot(str(git_repo.path), sha)
        assert path is not None
        assert path.exists()
        assert (path / "notes.md").read_text() == "# Notes\nStuff"

    def test_restore_snapshot(self, git_repo: GitTestRepo):
        git_repo.write_memory("notes.md", "Original")
        sha = git_repo.head_sha()
        take_memory_snapshot(str(git_repo.path), sha)

        # Modify memory
        git_repo.write_memory("notes.md", "Modified")
        assert git_repo.read_memory("notes.md") == "Modified"

        # Restore
        ok = restore_memory_snapshot(str(git_repo.path), sha)
        assert ok
        assert git_repo.read_memory("notes.md") == "Original"

    def test_restore_walks_ancestors(self, git_repo: GitTestRepo):
        # Take snapshot at parent
        git_repo.write_memory("notes.md", "At parent")
        parent_sha = git_repo.head_sha()
        take_memory_snapshot(str(git_repo.path), parent_sha)

        # Create child commit (no snapshot for it)
        git_repo.write_file("f.py", "x")
        child_sha = git_repo.commit("child")

        # Modify memory
        git_repo.write_memory("notes.md", "After child")

        # Restore should find parent's snapshot
        ok = restore_memory_snapshot(str(git_repo.path), child_sha)
        assert ok
        assert git_repo.read_memory("notes.md") == "At parent"

    def test_memory_diff(self, git_repo: GitTestRepo):
        git_repo.write_memory("a.md", "Version 1")
        sha1 = git_repo.head_sha()
        take_memory_snapshot(str(git_repo.path), sha1)

        git_repo.write_file("f.py", "x")
        sha2 = git_repo.commit("c2")
        git_repo.write_memory("a.md", "Version 2")
        git_repo.write_memory("b.md", "New file")
        take_memory_snapshot(str(git_repo.path), sha2)

        diff = get_memory_diff(str(git_repo.path), sha1, sha2)
        assert "a.md" in diff  # Modified
        assert "b.md" in diff  # New file

    def test_no_memory_dir(self, tmp_path):
        result = take_memory_snapshot(str(tmp_path), "abc123")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# agt_move
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgtMove:

    def test_move_to_previous_commit(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]
        state.conv_path = [shas[0], shas[1], shas[2]]

        result = agt_move(str(git_repo.path), state, shas[0])
        assert result.success
        assert state.agent_position_sha == shas[0]
        assert shas[0] in state.conv_path

    def test_move_creates_branch(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        result = agt_move(str(git_repo.path), state, shas[0])
        assert result.success
        # Should have created an alan/move-* branch
        branches = git_repo.branches()
        assert any(b.startswith("alan/move-") for b in branches)

    def test_move_to_branch_tip(self, git_repo: GitTestRepo):
        """Moving to a branch tip should checkout that branch, not create a new one."""
        result = git_repo.build_branching_history()
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = result["c5"]

        move = agt_move(str(git_repo.path), state, result["c4"])
        assert move.success
        # c4 is the tip of feature branch
        assert git_repo.current_branch() == "feature"

    def test_move_invalid_sha(self, git_repo: GitTestRepo):
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        result = agt_move(str(git_repo.path), state, "deadbeef" * 5)
        assert not result.success
        assert "not found" in result.description.lower()

    def test_move_discards_dirty(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[1]

        git_repo.write_file("dirty.txt", "uncommitted")
        assert git_repo.is_dirty()

        result = agt_move(str(git_repo.path), state, shas[0])
        assert result.success
        assert not git_repo.is_dirty()


# ═══════════════════════════════════════════════════════════════════════════════
# agt_revert
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgtRevert:

    def test_revert_1(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        result = agt_revert(str(git_repo.path), state, 1)
        assert result.success
        assert state.agent_position_sha == shas[1]

    def test_revert_n(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[4]

        result = agt_revert(str(git_repo.path), state, 3)
        assert result.success
        assert state.agent_position_sha == shas[1]

    def test_revert_dirty_discards(self, git_repo: GitTestRepo):
        """Revert 1 with dirty tree = discard uncommitted changes."""
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[1]

        git_repo.write_file("dirty.txt", "uncommitted")
        assert git_repo.is_dirty()

        result = agt_revert(str(git_repo.path), state, 1)
        assert result.success
        assert not git_repo.is_dirty()
        # Position stays at HEAD (dirty was discarded)
        assert "Discarded uncommitted" in result.description

    def test_revert_destroys_commits(self, git_repo: GitTestRepo):
        """Revert uses git reset --hard — commits are removed from branch."""
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]
        state.alan_commits = [shas[1], shas[2]]

        result = agt_revert(str(git_repo.path), state, 1)
        assert result.success

        # The destroyed commit should no longer be on the branch
        log = git_repo.log_hashes()
        assert shas[2] not in log
        assert shas[1] in log  # Still exists (not reverted)

        # HEAD should be at shas[1]
        assert git_repo.head_sha() == shas[1]

        # Destroyed commit removed from alan_commits
        assert shas[2] not in state.alan_commits
        assert shas[1] in state.alan_commits

    def test_revert_no_new_branch(self, git_repo: GitTestRepo):
        """Revert should NOT create alan/move-* branches."""
        shas = git_repo.build_linear_history(3)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[2]

        branches_before = set(git_repo.branches())
        agt_revert(str(git_repo.path), state, 1)
        branches_after = set(git_repo.branches())

        # No new branches created
        assert branches_after == branches_before

    def test_revert_past_root_partial(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[1]

        result = agt_revert(str(git_repo.path), state, 100)
        assert result.success
        assert "available" in result.description.lower() or result.success


# ═══════════════════════════════════════════════════════════════════════════════
# agt_conv_revert
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgtConvRevert:

    def test_conv_revert_basic(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.conv_path = [shas[0], shas[1], shas[2], shas[3]]
        state.agent_position_sha = shas[3]

        result = agt_conv_revert(str(git_repo.path), state, 2)
        assert result.success
        assert result.steps_reverted == 2
        assert state.conv_path == [shas[0], shas[1]]
        # Position unchanged
        assert state.agent_position_sha == shas[3]

    def test_conv_revert_past_compaction(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.conv_path = [shas[0], shas[1], shas[2], shas[3], shas[4]]
        state.compaction_markers = [shas[2]]

        result = agt_conv_revert(str(git_repo.path), state, 3)
        assert result.success
        assert state.conv_path == [shas[0], shas[1]]
        # Compaction marker at shas[2] should be removed (not in conv_path anymore)
        assert shas[2] not in state.compaction_markers

    def test_conv_revert_too_short(self, git_repo: GitTestRepo):
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.conv_path = ["sha1"]

        result = agt_conv_revert(str(git_repo.path), state, 1)
        assert not result.success


# ═══════════════════════════════════════════════════════════════════════════════
# agt_all_revert
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgtAllRevert:

    def test_all_revert(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.agent_position_sha = shas[3]
        state.conv_path = [shas[0], shas[1], shas[2], shas[3]]

        result = agt_all_revert(str(git_repo.path), state, 2)
        assert result.success
        # Both position and conv should be reverted
        assert state.agent_position_sha == shas[1]
        # Conv path truncated by 2
        assert len(state.conv_path) <= 3  # Original 4 - 2 + appended by move


# ═══════════════════════════════════════════════════════════════════════════════
# detect_orphaned_shas
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectOrphaned:

    def test_no_orphans(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.alan_commits = [shas[0]]
        state.conv_path = [shas[0], shas[1]]

        orphaned = detect_orphaned_shas(str(git_repo.path), state)
        assert orphaned == []

    def test_detects_orphan(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        state = SessionState(session_id="t", cwd=str(git_repo.path))
        state.alan_commits = ["deadbeef" * 5]
        state.conv_path = [shas[0], "deadbeef" * 5]

        orphaned = detect_orphaned_shas(str(git_repo.path), state)
        assert len(orphaned) == 1
        # State should be cleaned
        assert "deadbeef" * 5 not in state.alan_commits
        assert "deadbeef" * 5 not in state.conv_path
