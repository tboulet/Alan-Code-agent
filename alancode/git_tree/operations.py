"""AGT movement operations — move, revert, conv_revert, all_revert.

All operations use ``agt_move`` as the core primitive.
Revert is a special case of move (walk backward n commits).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alancode.git_tree.memory_snapshots import (
    get_memory_diff,
    restore_memory_snapshot,
    take_memory_snapshot,
)

if TYPE_CHECKING:
    from alancode.session.state import SessionState

logger = logging.getLogger(__name__)


@dataclass
class MoveResult:
    """Result of a movement operation."""
    success: bool
    description: str
    old_sha: str = ""
    new_sha: str = ""
    repo_diff: str = ""
    memory_diff: str = ""
    new_branch: str = ""


@dataclass
class ConvRevertResult:
    """Result of a conversation revert."""
    success: bool
    description: str
    steps_reverted: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Core operations
# ═══════════════════════════════════════════════════════════════════════════════


def agt_move(cwd: str, state: SessionState, target_sha: str) -> MoveResult:
    """Move the agent to a target commit.

    This is the ONE core operation.  All other movement operations
    delegate to this.

    Steps:
    1. Validate target exists in git
    2. Snapshot current memory (if position is an alan commit)
    3. Discard uncommitted changes
    4. Checkout target (create branch if needed)
    5. Restore memory snapshot for target
    6. Update session state
    7. Compute diffs for agent notification
    """
    old_sha = state.agent_position_sha or _git_head(cwd) or ""

    # Validate target
    if not _sha_exists(cwd, target_sha):
        return MoveResult(False, f"Commit {target_sha[:7]} not found in git.")

    # Snapshot current memory before moving
    if old_sha:
        take_memory_snapshot(cwd, old_sha)

    # Discard uncommitted changes (preserve .alan/ directory).
    # If either step fails we must NOT proceed — session state would
    # record a move that didn't actually happen, and the agent's view
    # of reality would diverge from the working tree.
    r = _run_git(cwd, "checkout", "-f")
    if r.returncode != 0:
        return MoveResult(False, f"git checkout -f failed: {r.stderr.strip()}")
    r = _run_git(cwd, "clean", "-fd", "-e", ".alan")
    if r.returncode != 0:
        return MoveResult(False, f"git clean failed: {r.stderr.strip()}")

    # Determine checkout strategy
    target_branches = _branches_at(cwd, target_sha)
    current_branch = _git_current_branch(cwd)
    new_branch = ""

    if target_branches:
        # Target is a branch tip — just checkout that branch
        branch = target_branches[0]
        result = _run_git(cwd, "checkout", branch)
        if result.returncode != 0:
            return MoveResult(False, f"Failed to checkout {branch}: {result.stderr.strip()}")
    elif target_sha == _git_head(cwd):
        pass  # Already there
    else:
        # Need a new branch
        new_branch = _unique_branch_name(cwd, target_sha)
        result = _run_git(cwd, "checkout", "-b", new_branch, target_sha)
        if result.returncode != 0:
            return MoveResult(False, f"Failed to create branch: {result.stderr.strip()}")

    # Restore memory snapshot
    restore_memory_snapshot(cwd, target_sha)

    # Compute diffs
    repo_diff = ""
    if old_sha and old_sha != target_sha:
        diff_result = _run_git(cwd, "diff", "--stat", old_sha, target_sha)
        if diff_result.returncode == 0:
            repo_diff = diff_result.stdout.strip()

    memory_diff = ""
    if old_sha:
        memory_diff = get_memory_diff(cwd, old_sha, target_sha)

    # Update session state
    with state.batch():
        state.agent_position_sha = target_sha
        state.add_to_conv_path(target_sha)

    actual_branch = _git_current_branch(cwd) or new_branch or "detached"
    desc_parts = [f"Moved to {target_sha[:7]} on {actual_branch}"]
    if new_branch:
        desc_parts.append(f"(new branch: {new_branch})")
    if repo_diff:
        desc_parts.append(f"\nFiles changed:\n{repo_diff}")
    if memory_diff:
        desc_parts.append(f"\nMemory changes:\n{memory_diff}")

    return MoveResult(
        success=True,
        description="\n".join(desc_parts),
        old_sha=old_sha,
        new_sha=target_sha,
        repo_diff=repo_diff,
        memory_diff=memory_diff,
        new_branch=new_branch,
    )


def agt_revert(cwd: str, state: SessionState, n: int = 1) -> MoveResult:
    """Destructively revert n commits from the current branch.

    Uses ``git reset --hard HEAD~n`` — the commits are **removed** from
    the branch.  They remain recoverable via ``git reflog`` for ~30 days.

    If there are uncommitted changes and n=1, this just discards them.
    """
    current = state.agent_position_sha or _git_head(cwd)
    if not current:
        return MoveResult(False, "No current position to revert from.")

    # If dirty and n=1, just discard uncommitted changes
    is_dirty = bool(_run_git(cwd, "status", "--porcelain").stdout.strip())
    if is_dirty and n == 1:
        r = _run_git(cwd, "checkout", "-f")
        if r.returncode != 0:
            return MoveResult(False, f"git checkout -f failed: {r.stderr.strip()}")
        r = _run_git(cwd, "clean", "-fd", "-e", ".alan")
        if r.returncode != 0:
            return MoveResult(False, f"git clean failed: {r.stderr.strip()}")
        return MoveResult(
            success=True,
            description=f"Discarded uncommitted changes. Still at {current[:7]}.",
            old_sha=current,
            new_sha=current,
        )

    # Compute effective steps (dirty state counts as 1 step)
    effective_n = n if not is_dirty else n - 1
    if effective_n <= 0:
        effective_n = 1

    # Find target
    target_result = _run_git(cwd, "rev-parse", f"{current}~{effective_n}")
    if target_result.returncode != 0:
        actual_n = 0
        for i in range(effective_n - 1, 0, -1):
            result = _run_git(cwd, "rev-parse", f"{current}~{i}")
            if result.returncode == 0:
                actual_n = i
                break
        if actual_n == 0:
            return MoveResult(False, f"Cannot revert {n} commits from {current[:7]}.")
        target_sha = _run_git(cwd, "rev-parse", f"{current}~{actual_n}").stdout.strip()
    else:
        target_sha = target_result.stdout.strip()

    return _destructive_reset(cwd, state, target_sha)


def agt_revert_to(cwd: str, state: SessionState, target_sha: str) -> MoveResult:
    """Destructively revert to a specific commit SHA.

    All commits between HEAD and *target_sha* on the current branch
    are removed via ``git reset --hard``.  The target must be an
    ancestor of HEAD.
    """
    current = state.agent_position_sha or _git_head(cwd)
    if not current:
        return MoveResult(False, "No current position to revert from.")

    if not _sha_exists(cwd, target_sha):
        return MoveResult(False, f"Commit {target_sha[:7]} not found.")

    # Verify target is an ancestor of current (otherwise reset would be wrong)
    ancestor_check = _run_git(cwd, "merge-base", "--is-ancestor", target_sha, current)
    if ancestor_check.returncode != 0:
        return MoveResult(
            False,
            f"{target_sha[:7]} is not an ancestor of current HEAD. "
            f"Use /move instead for cross-branch navigation.",
        )

    if target_sha == current:
        # Already there — just discard dirty state if any
        is_dirty = bool(_run_git(cwd, "status", "--porcelain").stdout.strip())
        if is_dirty:
            r = _run_git(cwd, "checkout", "-f")
            if r.returncode != 0:
                return MoveResult(False, f"git checkout -f failed: {r.stderr.strip()}")
            r = _run_git(cwd, "clean", "-fd", "-e", ".alan")
            if r.returncode != 0:
                return MoveResult(False, f"git clean failed: {r.stderr.strip()}")
            return MoveResult(True, f"Discarded uncommitted changes. Still at {current[:7]}.",
                              old_sha=current, new_sha=current)
        return MoveResult(True, f"Already at {current[:7]}.", old_sha=current, new_sha=current)

    return _destructive_reset(cwd, state, target_sha)


def _destructive_reset(cwd: str, state: SessionState, target_sha: str) -> MoveResult:
    """Core destructive reset: git reset --hard <target>, update state."""
    old_sha = state.agent_position_sha or _git_head(cwd) or ""

    # Snapshot memory before reverting
    if old_sha:
        take_memory_snapshot(cwd, old_sha)

    # Compute diff before destroying commits
    diff_result = _run_git(cwd, "diff", "--stat", target_sha, old_sha)
    repo_diff = diff_result.stdout.strip() if diff_result.returncode == 0 else ""
    memory_diff = get_memory_diff(cwd, old_sha, target_sha) if old_sha else ""

    # Destructive reset
    reset_result = _run_git(cwd, "reset", "--hard", target_sha)
    if reset_result.returncode != 0:
        return MoveResult(False, f"git reset --hard failed: {reset_result.stderr.strip()}")

    # Restore memory snapshot
    restore_memory_snapshot(cwd, target_sha)

    # Find destroyed SHAs (everything between old and target)
    destroyed_shas: set[str] = set()
    if old_sha and old_sha != target_sha:
        walk = old_sha
        for _ in range(200):  # Safety cap
            if walk == target_sha:
                break
            destroyed_shas.add(walk)
            parent = _run_git(cwd, "rev-parse", f"{walk}~1")
            if parent.returncode != 0:
                break
            walk = parent.stdout.strip()

    # Update session state
    with state.batch():
        state.agent_position_sha = target_sha
        state.add_to_conv_path(target_sha)
        if destroyed_shas:
            state.alan_commits = [s for s in state.alan_commits if s not in destroyed_shas]

    branch = _git_current_branch(cwd) or "detached"
    n_destroyed = len(destroyed_shas)
    desc_parts = [f"Reverted {n_destroyed} commit(s). Now at {target_sha[:7]} on {branch}."]
    if repo_diff:
        desc_parts.append(f"\nFiles changed:\n{repo_diff}")
    if memory_diff:
        desc_parts.append(f"\nMemory changes:\n{memory_diff}")

    return MoveResult(
        success=True,
        description="\n".join(desc_parts),
        old_sha=old_sha,
        new_sha=target_sha,
        repo_diff=repo_diff,
        memory_diff=memory_diff,
    )


def agt_conv_revert(cwd: str, state: SessionState, n: int = 1) -> ConvRevertResult:
    """Revert n steps in the conversation path.

    This truncates the conversation path but does NOT move the git
    position.  The agent's position stays the same.
    """
    conv = state.conv_path
    if len(conv) <= 1:
        return ConvRevertResult(False, "Conversation path is too short to revert.", 0)

    actual_n = min(n, len(conv) - 1)  # Keep at least 1 entry
    new_conv = conv[:-actual_n]

    # Check if we crossed a compaction marker
    markers = state.compaction_markers
    new_markers = [m for m in markers if m in set(new_conv)]

    with state.batch():
        state.conv_path = new_conv
        if len(new_markers) != len(markers):
            state.compaction_markers = new_markers

    return ConvRevertResult(
        success=True,
        description=f"Conversation reverted {actual_n} step(s). Position unchanged.",
        steps_reverted=actual_n,
    )


def agt_all_revert(cwd: str, state: SessionState, n: int = 1) -> MoveResult:
    """Revert both git position AND conversation path by n steps."""
    # First, revert conversation
    conv_result = agt_conv_revert(cwd, state, n)

    # Then, revert git position
    move_result = agt_revert(cwd, state, n)

    if move_result.success:
        move_result.description = (
            f"All-reverted {n} step(s).\n"
            f"Conversation: {conv_result.description}\n"
            f"Position: {move_result.description}"
        )
    return move_result


def detect_orphaned_shas(cwd: str, state: SessionState) -> list[str]:
    """Find SHAs in session state that are no longer reachable from any branch.

    Uses ``git branch --contains`` to check reachability, not just object
    existence (unreachable objects still exist in git's object store after
    ``git reset --hard``).

    Returns the list of orphaned SHAs and cleans them from state.
    """
    orphaned: list[str] = []

    all_shas = set(state.alan_commits) | set(state.conv_path) | set(state.compaction_markers)
    if state.session_root_sha:
        all_shas.add(state.session_root_sha)
    if state.agent_position_sha:
        all_shas.add(state.agent_position_sha)

    for sha in all_shas:
        if not _sha_reachable(cwd, sha):
            orphaned.append(sha)

    if orphaned:
        orphaned_set = set(orphaned)
        with state.batch():
            state.alan_commits = [s for s in state.alan_commits if s not in orphaned_set]
            state.conv_path = [s for s in state.conv_path if s not in orphaned_set]
            state.compaction_markers = [s for s in state.compaction_markers if s not in orphaned_set]
            if state.session_root_sha in orphaned_set:
                state.session_root_sha = _git_head(cwd) or ""
            if state.agent_position_sha in orphaned_set:
                state.agent_position_sha = _git_head(cwd) or ""
        logger.info("Cleaned %d orphaned SHAs from session state", len(orphaned))

    return orphaned


# ═══════════════════════════════════════════════════════════════════════════════
# Git helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _run_git(cwd: str, *args: str) -> subprocess.CompletedProcess:
    """Run a git command."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _git_head(cwd: str) -> str | None:
    result = _run_git(cwd, "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else None


def _git_current_branch(cwd: str) -> str | None:
    result = _run_git(cwd, "symbolic-ref", "--short", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else None


def _sha_exists(cwd: str, sha: str) -> bool:
    result = _run_git(cwd, "cat-file", "-t", sha)
    return result.returncode == 0


def _sha_reachable(cwd: str, sha: str) -> bool:
    """Check if a SHA is reachable from any branch (not just existing as object)."""
    result = _run_git(cwd, "branch", "--all", "--contains", sha)
    return result.returncode == 0 and bool(result.stdout.strip())


def _branches_at(cwd: str, sha: str) -> list[str]:
    """Return branch names whose tip is exactly at this SHA."""
    result = _run_git(cwd, "branch", "--points-at", sha, "--format=%(refname:short)")
    if result.returncode != 0:
        return []
    return [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]


def _unique_branch_name(cwd: str, sha: str) -> str:
    """Generate a unique branch name like alan/move-abc1234."""
    short = sha[:7]
    base = f"alan/move-{short}"
    name = base
    counter = 2
    while True:
        result = _run_git(cwd, "branch", "--list", name)
        if not result.stdout.strip():
            return name
        name = f"{base}-{counter}"
        counter += 1
