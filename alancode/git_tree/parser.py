"""Parse ``git log --all`` into an :class:`AGTTree`.

This is the single entry point for building the tree.  It runs git
commands, parses the output, and returns a fully populated tree.
The tree is never cached — it is re-parsed on every call.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Set

from alancode.git_tree.model import CURRENT_NODE_SHA, AGTNode, AGTTree, NodeType

logger = logging.getLogger(__name__)

# Separator used in git log format string (pipe is unlikely in commit messages,
# but we use a double-pipe with padding to be safe)
_SEP = " ||| "
_FORMAT = f"%H{_SEP}%P{_SEP}%s{_SEP}%an{_SEP}%aI{_SEP}%D"


def parse_git_tree(
    cwd: str,
    alan_commits: Set[str] | None = None,
) -> AGTTree:
    """Parse the git repository at *cwd* into an :class:`AGTTree`.

    Parameters
    ----------
    cwd : str
        Path to the git repository.
    alan_commits : set[str], optional
        Set of commit SHAs that were made by the agent (via GitCommit tool).
        These are classified as ``ALAN_COMMIT``; all others as ``EXTERNAL_COMMIT``.

    Returns
    -------
    AGTTree
        The parsed tree.  If *cwd* is not a git repo or has no commits,
        an empty tree is returned (possibly with a virtual current node
        if there are untracked/modified files).
    """
    alan_commits = alan_commits or set()

    # ── Get HEAD info ────────────────────────────────────────────────
    head_sha = _git_head_sha(cwd)
    current_branch = _git_current_branch(cwd)
    is_dirty = _git_is_dirty(cwd)

    # ── Parse git log ────────────────────────────────────────────────
    nodes: dict[str, AGTNode] = {}
    root_shas: list[str] = []

    log_output = _git_log_all(cwd)
    if log_output:
        for line in log_output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            node = _parse_log_line(line, alan_commits, head_sha)
            if node:
                nodes[node.sha] = node
                if not node.parents:
                    root_shas.append(node.sha)

    # ── Populate children (inverse of parents) ───────────────────────
    for node in nodes.values():
        for parent_sha in node.parents:
            parent = nodes.get(parent_sha)
            if parent and node.sha not in parent.children:
                parent.children.append(node.sha)

    # ── Add virtual current node if dirty ────────────────────────────
    if is_dirty:
        current_node = AGTNode(
            sha=CURRENT_NODE_SHA,
            short_sha="dirty",
            message="Uncommitted changes",
            author="",
            timestamp="",
            parents=[head_sha] if head_sha else [],
            children=[],
            node_type=NodeType.CURRENT_NODE,
            branches=[],
            is_head=False,
        )
        nodes[CURRENT_NODE_SHA] = current_node
        # Add as child of HEAD
        if head_sha and head_sha in nodes:
            nodes[head_sha].children.append(CURRENT_NODE_SHA)

    return AGTTree(
        nodes=nodes,
        root_shas=root_shas,
        head_sha=head_sha,
        is_dirty=is_dirty,
        current_branch=current_branch,
    )


# ── Git commands ─────────────────────────────────────────────────────────────


def _run_git(cwd: str, *args: str) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _git_head_sha(cwd: str) -> str | None:
    """Return the full SHA of HEAD, or None."""
    out = _run_git(cwd, "rev-parse", "HEAD")
    return out.strip() if out else None


def _git_current_branch(cwd: str) -> str | None:
    """Return current branch name, or None if detached."""
    out = _run_git(cwd, "symbolic-ref", "--short", "HEAD")
    return out.strip() if out else None


def _git_is_dirty(cwd: str) -> bool:
    """Check if working tree has uncommitted changes."""
    out = _run_git(cwd, "status", "--porcelain")
    return bool(out and out.strip())


def _git_log_all(cwd: str) -> str | None:
    """Run git log --all with custom format."""
    return _run_git(
        cwd, "log", "--all",
        f"--format={_FORMAT}",
        "--topo-order",
    )


# ── Line parsing ─────────────────────────────────────────────────────────────


def _parse_log_line(
    line: str,
    alan_commits: Set[str],
    head_sha: str | None,
) -> AGTNode | None:
    """Parse a single git log line into an AGTNode."""
    parts = line.split(_SEP)
    # Need at least 5 parts (decorations may be empty/missing when git
    # outputs a trailing separator with no space after it)
    if len(parts) < 5:
        logger.debug("Skipping malformed git log line: %s", line[:80])
        return None

    sha = parts[0].strip()
    parent_str = parts[1].strip()
    message = parts[2].strip()
    author = parts[3].strip()
    timestamp = parts[4].strip()
    decorations = parts[5].strip() if len(parts) > 5 else ""

    parents = parent_str.split() if parent_str else []

    # Parse branch names from decorations like "HEAD -> main, origin/main"
    branches: list[str] = []
    if decorations:
        for dec in decorations.split(","):
            dec = dec.strip()
            if dec.startswith("HEAD -> "):
                branches.append(dec[8:])
            elif dec == "HEAD":
                continue  # Detached HEAD, no branch name
            elif "/" in dec:
                continue  # Skip remote refs like origin/main
            else:
                branches.append(dec)

    node_type = (
        NodeType.ALAN_COMMIT if sha in alan_commits
        else NodeType.EXTERNAL_COMMIT
    )

    return AGTNode(
        sha=sha,
        short_sha=sha[:7],
        message=message,
        author=author,
        timestamp=timestamp,
        parents=parents,
        children=[],
        node_type=node_type,
        branches=branches,
        is_head=(sha == head_sha),
    )
