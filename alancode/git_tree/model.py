"""AGT data model — nodes, edges, and tree structure.

All data is derived from ``git log``.  The tree is never stored
separately — it is re-parsed from git on every update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Type of node in the AGT."""
    ALAN_COMMIT = "alan_commit"        # Blue — committed via GitCommit tool
    EXTERNAL_COMMIT = "external"       # Grey — any other commit
    CURRENT_NODE = "current"           # White dashed — uncommitted changes


# Virtual SHA for the "current node" (dirty working tree)
CURRENT_NODE_SHA = "__dirty__"


@dataclass
class AGTNode:
    """A single node in the Agentic Git Tree."""
    sha: str
    short_sha: str
    message: str
    author: str
    timestamp: str                     # ISO 8601
    parents: list[str]                 # Parent SHAs
    children: list[str] = field(default_factory=list)
    node_type: NodeType = NodeType.EXTERNAL_COMMIT
    branches: list[str] = field(default_factory=list)  # Branch names at this commit
    is_head: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "short_sha": self.short_sha,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "parents": self.parents,
            "children": self.children,
            "node_type": self.node_type.value,
            "branches": self.branches,
            "is_head": self.is_head,
        }


@dataclass
class AGTTree:
    """The full Agentic Git Tree derived from git log.

    Attributes
    ----------
    nodes : dict[str, AGTNode]
        SHA → node mapping.  Includes the virtual ``CURRENT_NODE_SHA``
        if the working tree is dirty.
    root_shas : list[str]
        Commits with no parents (initial commits).
    head_sha : str | None
        The commit that HEAD points to (even if dirty).
    is_dirty : bool
        True if the working tree has uncommitted changes.
    current_branch : str | None
        Current branch name, or None if detached HEAD.
    """

    nodes: dict[str, AGTNode] = field(default_factory=dict)
    root_shas: list[str] = field(default_factory=list)
    head_sha: str | None = None
    is_dirty: bool = False
    current_branch: str | None = None

    # ── Accessors ────────────────────────────────────────────────────

    def get_node(self, sha: str) -> AGTNode | None:
        """Return a node by SHA, or None."""
        return self.nodes.get(sha)

    @property
    def commit_count(self) -> int:
        """Number of real commits (excluding virtual current node)."""
        return sum(
            1 for n in self.nodes.values()
            if n.node_type != NodeType.CURRENT_NODE
        )

    def walk_ancestors(self, sha: str, n: int) -> list[str]:
        """Walk back n ancestors following first-parent links.

        Returns a list of SHAs starting from the immediate parent,
        up to n ancestors.  Stops early if a root is reached.
        """
        result: list[str] = []
        current = sha
        for _ in range(n):
            node = self.nodes.get(current)
            if not node or not node.parents:
                break
            parent = node.parents[0]  # First parent
            result.append(parent)
            current = parent
        return result

    def get_mainline(self) -> list[str]:
        """Return the first-parent chain from HEAD to root.

        This is the "main line" of development — the backbone of the tree.
        """
        if not self.head_sha:
            return []
        chain: list[str] = [self.head_sha]
        current = self.head_sha
        while True:
            node = self.nodes.get(current)
            if not node or not node.parents:
                break
            current = node.parents[0]
            chain.append(current)
        return chain

    def sha_exists(self, sha: str) -> bool:
        """Check if a SHA exists in the tree."""
        return sha in self.nodes

    # ── Serialization ────────────────────────────────────────────────

    def to_json(self) -> dict[str, Any]:
        """Serialize the tree to a JSON-compatible dict."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "root_shas": self.root_shas,
            "head_sha": self.head_sha,
            "is_dirty": self.is_dirty,
            "current_branch": self.current_branch,
            "commit_count": self.commit_count,
        }
