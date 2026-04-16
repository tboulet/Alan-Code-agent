"""Agentic Git Tree (AGT) — git-backed tree model and operations.

The AGT derives a tree visualization from ``git log``, tracks which
commits the agent made, and supports navigation (move, revert).
Non-git repos are completely unaffected.

Usage::

    from alancode.git_tree.parser import parse_git_tree
    tree = parse_git_tree("/path/to/repo", alan_commits={"abc123"})
"""

from alancode.git_tree.model import AGTNode, AGTTree, NodeType

__all__ = ["AGTNode", "AGTTree", "NodeType"]
