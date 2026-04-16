"""Tree layout algorithm — compute (x, y) positions for SVG rendering.

Y-axis: root at top (y=0), newest commits at the bottom.
X-axis: mainline at x=0, branches offset left/right.

The mainline is determined from the ``main`` (or ``master``) branch,
NOT from HEAD — so moving to other branches doesn't reshuffle the layout.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from alancode.git_tree.model import CURRENT_NODE_SHA, AGTTree, NodeType


@dataclass
class LayoutNode:
    """A positioned node for rendering."""
    sha: str
    x: float
    y: float
    node_type: str
    short_sha: str
    message: str
    author: str
    timestamp: str
    branches: list[str]
    is_head: bool
    is_compaction_marker: bool = False
    is_agent_position: bool = False
    is_on_conv_path: bool = False
    is_session_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha": self.sha,
            "x": self.x,
            "y": self.y,
            "node_type": self.node_type,
            "short_sha": self.short_sha,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "branches": self.branches,
            "is_head": self.is_head,
            "is_compaction_marker": self.is_compaction_marker,
            "is_agent_position": self.is_agent_position,
            "is_on_conv_path": self.is_on_conv_path,
            "is_session_root": self.is_session_root,
        }


@dataclass
class LayoutEdge:
    """An edge between two nodes for rendering."""
    from_sha: str
    to_sha: str
    edge_type: str  # "parent", "conv_path", "post_compaction", "conv_jump"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_sha": self.from_sha,
            "to_sha": self.to_sha,
            "edge_type": self.edge_type,
        }


@dataclass
class TreeLayout:
    """Complete layout ready for SVG rendering."""
    nodes: list[LayoutNode] = field(default_factory=list)
    edges: list[LayoutEdge] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "width": self.width,
            "height": self.height,
        }


def compute_layout(
    tree: AGTTree,
    conv_path: list[str] | None = None,
    compaction_markers: list[str] | None = None,
    agent_position: str | None = None,
    session_root: str | None = None,
) -> TreeLayout:
    """Compute positioned nodes and classified edges for rendering."""
    if not tree.nodes:
        return TreeLayout()

    conv_path = conv_path or []
    compaction_markers = compaction_markers or []
    conv_path_set = set(conv_path)
    compaction_set = set(compaction_markers)

    # ── Step 1: Topological sort (parents before children) ───────────
    sorted_shas = _topo_sort(tree)

    # ── Step 2: Y positions (row index, oldest at top) ───────────────
    y_map: dict[str, float] = {}
    for i, sha in enumerate(sorted_shas):
        y_map[sha] = float(i)

    # ── Step 3: X positions (branch lanes) ───────────────────────────
    mainline_set = _find_mainline(tree)
    x_map = _assign_x_positions(tree, sorted_shas, mainline_set)

    # ── Step 4: Build layout nodes ───────────────────────────────────
    layout_nodes: list[LayoutNode] = []
    for sha in sorted_shas:
        node = tree.get_node(sha)
        if not node:
            continue
        layout_nodes.append(LayoutNode(
            sha=sha,
            x=x_map.get(sha, 0.0),
            y=y_map.get(sha, 0.0),
            node_type=node.node_type.value,
            short_sha=node.short_sha,
            message=node.message,
            author=node.author,
            timestamp=node.timestamp,
            branches=node.branches,
            is_head=node.is_head,
            is_compaction_marker=(sha in compaction_set),
            is_agent_position=(sha == agent_position),
            is_on_conv_path=(sha in conv_path_set),
            is_session_root=(sha == session_root),
        ))

    # ── Step 5: Build edges ──────────────────────────────────────────
    layout_edges: list[LayoutEdge] = []

    # Parent edges
    for sha in sorted_shas:
        node = tree.get_node(sha)
        if not node:
            continue
        for parent_sha in node.parents:
            if parent_sha in tree.nodes:
                layout_edges.append(LayoutEdge(parent_sha, sha, "parent"))

    # Conversation path edges (blue)
    for i in range(len(conv_path) - 1):
        src, dst = conv_path[i], conv_path[i + 1]
        if src not in tree.nodes or dst not in tree.nodes:
            continue
        dst_node = tree.get_node(dst)
        is_parent_child = dst_node and src in dst_node.parents
        layout_edges.append(LayoutEdge(
            src, dst, "conv_path" if is_parent_child else "conv_jump",
        ))

    # Post-compaction edges (yellow) — from last compaction marker (or
    # session root) to agent position.  Always shown.
    if conv_path and agent_position:
        start_sha = conv_path[0]
        if compaction_markers:
            last_marker = compaction_markers[-1]
            if last_marker in conv_path_set:
                start_sha = last_marker
        try:
            start_idx = conv_path.index(start_sha)
            for i in range(start_idx, len(conv_path) - 1):
                src, dst = conv_path[i], conv_path[i + 1]
                if src in tree.nodes and dst in tree.nodes:
                    layout_edges.append(LayoutEdge(src, dst, "post_compaction"))
        except ValueError:
            pass

    # Conv gap arc: if agent_position is not the last conv_path entry,
    # draw a dashed arc from the conv end to agent_position.
    # This happens after /convrevert — conversation was cut but agent
    # is still at a later commit.
    if conv_path and agent_position and agent_position in tree.nodes:
        last_conv = conv_path[-1]
        if last_conv != agent_position and last_conv in tree.nodes:
            layout_edges.append(LayoutEdge(last_conv, agent_position, "conv_jump"))
            layout_edges.append(LayoutEdge(last_conv, agent_position, "post_compaction"))

    # ── Step 6: Compute bounds ───────────────────────────────────────
    if layout_nodes:
        min_x = min(n.x for n in layout_nodes)
        max_x = max(n.x for n in layout_nodes)
        max_y = max(n.y for n in layout_nodes)
        width = max_x - min_x + 2
        height = max_y + 1
    else:
        width = height = 0.0

    return TreeLayout(nodes=layout_nodes, edges=layout_edges,
                      width=width, height=height)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _topo_sort(tree: AGTTree) -> list[str]:
    """Topological sort: parents before children, CURRENT_NODE last."""
    in_degree: dict[str, int] = {sha: 0 for sha in tree.nodes}
    for sha, node in tree.nodes.items():
        if sha == CURRENT_NODE_SHA:
            continue
        for child_sha in node.children:
            if child_sha in in_degree and child_sha != CURRENT_NODE_SHA:
                in_degree[child_sha] += 1

    queue = deque(
        sha for sha, deg in in_degree.items()
        if deg == 0 and sha != CURRENT_NODE_SHA
    )
    result: list[str] = []
    while queue:
        sha = queue.popleft()
        result.append(sha)
        node = tree.nodes.get(sha)
        if not node:
            continue
        for child_sha in node.children:
            if child_sha == CURRENT_NODE_SHA or child_sha not in in_degree:
                continue
            in_degree[child_sha] -= 1
            if in_degree[child_sha] == 0:
                queue.append(child_sha)

    visited = set(result)
    for sha in tree.nodes:
        if sha not in visited and sha != CURRENT_NODE_SHA:
            result.append(sha)
    if CURRENT_NODE_SHA in tree.nodes:
        result.append(CURRENT_NODE_SHA)
    return result


def _find_mainline(tree: AGTTree) -> set[str]:
    """Find the mainline (first-parent chain from main/master branch tip).

    Uses ``main`` or ``master`` branch if it exists.  Falls back to HEAD.
    This ensures the mainline doesn't change when the user moves to other branches.
    """
    # Find the main/master branch tip
    main_sha = None
    for sha, node in tree.nodes.items():
        if sha == CURRENT_NODE_SHA:
            continue
        for branch in node.branches:
            if branch in ("main", "master"):
                main_sha = sha
                break
        if main_sha:
            break

    # Fallback to HEAD
    if not main_sha:
        main_sha = tree.head_sha

    if not main_sha:
        return set()

    # Walk first-parent chain
    chain: set[str] = set()
    current = main_sha
    while current:
        chain.add(current)
        node = tree.nodes.get(current)
        if not node or not node.parents:
            break
        current = node.parents[0]
    return chain


def _assign_x_positions(
    tree: AGTTree,
    sorted_shas: list[str],
    mainline_set: set[str],
) -> dict[str, float]:
    """Assign horizontal lanes to nodes.

    Mainline = x=0.  Each non-mainline branch gets its own lane,
    alternating left/right.  A child that shares x with a sibling
    is forced to a new lane.
    """
    x_map: dict[str, float] = {}
    # Track which x positions are taken at each y level to avoid overlap
    used_x_at_y: dict[float, set[float]] = {}
    branch_lanes: dict[str, float] = {}
    next_lane = 1
    lane_sign = 1  # Alternate +/-

    y_map: dict[str, float] = {}
    for i, sha in enumerate(sorted_shas):
        y_map[sha] = float(i)

    for sha in sorted_shas:
        node = tree.get_node(sha)
        if not node:
            continue
        y = y_map.get(sha, 0.0)

        if sha == CURRENT_NODE_SHA:
            head_x = x_map.get(tree.head_sha, 0.0) if tree.head_sha else 0.0
            x_map[sha] = head_x
            continue

        if sha in mainline_set:
            x_map[sha] = 0.0
            used_x_at_y.setdefault(y, set()).add(0.0)
            continue

        # Non-mainline: try to inherit lane from parent on same branch
        assigned = False

        # First try: inherit from non-mainline parent
        for p in node.parents:
            if p in x_map and p not in mainline_set:
                candidate_x = x_map[p]
                if candidate_x not in used_x_at_y.get(y, set()):
                    x_map[sha] = candidate_x
                    used_x_at_y.setdefault(y, set()).add(candidate_x)
                    assigned = True
                    break

        if assigned:
            continue

        # Second try: use branch name lane
        branch_name = node.branches[0] if node.branches else None
        if branch_name and branch_name in branch_lanes:
            candidate_x = branch_lanes[branch_name]
            if candidate_x not in used_x_at_y.get(y, set()):
                x_map[sha] = candidate_x
                used_x_at_y.setdefault(y, set()).add(candidate_x)
                continue

        # New lane needed
        lane_x = next_lane * lane_sign
        # Ensure this lane isn't taken at this y level
        while lane_x in used_x_at_y.get(y, set()):
            next_lane += 1
            lane_sign *= -1
            lane_x = next_lane * lane_sign

        x_map[sha] = lane_x
        used_x_at_y.setdefault(y, set()).add(lane_x)
        if branch_name:
            branch_lanes[branch_name] = lane_x
        next_lane += 1
        lane_sign *= -1

    return x_map
