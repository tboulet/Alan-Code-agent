"""Tests for AGT Phase 3: Tree layout algorithm."""

import pytest

from alancode.git_tree.layout import (
    LayoutEdge,
    LayoutNode,
    TreeLayout,
    compute_layout,
)
from alancode.git_tree.model import CURRENT_NODE_SHA
from alancode.git_tree.parser import parse_git_tree
from tests.integration.git_helpers import GitTestRepo


class TestLinearLayout:
    """Test layout with a simple linear history."""

    def test_all_on_mainline(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        # All nodes should be at x=0 (mainline)
        for node in layout.nodes:
            assert node.x == 0.0, f"Node {node.short_sha} at x={node.x}"

    def test_y_ordering(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        # Oldest should have lowest y, newest highest
        y_by_sha = {n.sha: n.y for n in layout.nodes}
        # shas[0] (oldest) should have lower y than shas[3] (newest)
        assert y_by_sha[shas[0]] < y_by_sha[shas[3]]

    def test_node_count(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        # 3 commits + 1 initial
        assert len(layout.nodes) == 4

    def test_parent_edges(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        parent_edges = [e for e in layout.edges if e.edge_type == "parent"]
        # 3 commits + 1 initial = 3 parent edges (each child points to parent)
        assert len(parent_edges) == 3


class TestBranchingLayout:
    """Test layout with branches."""

    def test_feature_branch_offset(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        x_by_sha = {n.sha: n.x for n in layout.nodes}

        # Mainline nodes at x=0
        assert x_by_sha[result["c1"]] == 0.0
        assert x_by_sha[result["c2"]] == 0.0
        assert x_by_sha[result["c3"]] == 0.0
        assert x_by_sha[result["c5"]] == 0.0  # Merge on mainline

        # Feature branch node at non-zero x
        assert x_by_sha[result["c4"]] != 0.0

    def test_merge_node_has_two_parent_edges(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        # Merge commit c5 should have edges from both c3 and c4
        merge_parent_edges = [
            e for e in layout.edges
            if e.edge_type == "parent" and e.to_sha == result["c5"]
        ]
        assert len(merge_parent_edges) == 2
        parent_shas = {e.from_sha for e in merge_parent_edges}
        assert result["c3"] in parent_shas
        assert result["c4"] in parent_shas


class TestDirtyNodeLayout:
    """Test layout when working tree is dirty."""

    def test_dirty_node_above_head(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        git_repo.write_file("dirty.txt", "x")
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        y_by_sha = {n.sha: n.y for n in layout.nodes}

        # Current node should have the highest y (at the bottom = newest)
        assert CURRENT_NODE_SHA in y_by_sha
        assert y_by_sha[CURRENT_NODE_SHA] > y_by_sha[shas[1]]

    def test_dirty_node_same_x_as_head(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        git_repo.write_file("dirty.txt", "x")
        tree = parse_git_tree(str(git_repo.path))
        layout = compute_layout(tree)

        x_by_sha = {n.sha: n.x for n in layout.nodes}
        assert x_by_sha[CURRENT_NODE_SHA] == x_by_sha[shas[1]]


class TestConvPathEdges:
    """Test conversation path edge classification."""

    def test_conv_path_edges_created(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(4)
        tree = parse_git_tree(str(git_repo.path))

        conv_path = [shas[0], shas[1], shas[2], shas[3]]
        layout = compute_layout(tree, conv_path=conv_path)

        conv_edges = [e for e in layout.edges if e.edge_type == "conv_path"]
        assert len(conv_edges) == 3  # 4 nodes = 3 edges

    def test_conv_jump_edges(self, git_repo: GitTestRepo):
        """When conv_path has a jump (not parent-child), edge_type is conv_jump."""
        shas = git_repo.build_linear_history(5)
        tree = parse_git_tree(str(git_repo.path))

        # Jump from shas[4] back to shas[1] (not parent-child)
        conv_path = [shas[0], shas[1], shas[2], shas[3], shas[4], shas[1]]
        layout = compute_layout(tree, conv_path=conv_path)

        jump_edges = [e for e in layout.edges if e.edge_type == "conv_jump"]
        assert len(jump_edges) >= 1
        # The jump from shas[4] to shas[1]
        assert any(
            e.from_sha == shas[4] and e.to_sha == shas[1]
            for e in jump_edges
        )

    def test_nodes_on_conv_path_marked(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        conv_path = [shas[0], shas[2]]
        layout = compute_layout(tree, conv_path=conv_path)

        on_path = {n.sha for n in layout.nodes if n.is_on_conv_path}
        assert shas[0] in on_path
        assert shas[2] in on_path
        assert shas[1] not in on_path  # Not in conv_path


class TestCompactionEdges:
    """Test post-compaction edge rendering."""

    def test_post_compaction_edges(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        tree = parse_git_tree(str(git_repo.path))

        # Compaction at shas[2], agent at shas[4]
        conv_path = [shas[0], shas[1], shas[2], shas[3], shas[4]]
        compaction_markers = [shas[2]]
        layout = compute_layout(
            tree,
            conv_path=conv_path,
            compaction_markers=compaction_markers,
            agent_position=shas[4],
        )

        post_comp_edges = [e for e in layout.edges if e.edge_type == "post_compaction"]
        # Should have edges from shas[2]→shas[3] and shas[3]→shas[4]
        assert len(post_comp_edges) == 2

    def test_compaction_marker_on_node(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(
            tree,
            compaction_markers=[shas[1]],
        )

        marker_nodes = [n for n in layout.nodes if n.is_compaction_marker]
        assert len(marker_nodes) == 1
        assert marker_nodes[0].sha == shas[1]


class TestAgentPosition:
    """Test agent position marking."""

    def test_agent_position_marked(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(tree, agent_position=shas[2])

        agent_nodes = [n for n in layout.nodes if n.is_agent_position]
        assert len(agent_nodes) == 1
        assert agent_nodes[0].sha == shas[2]


class TestLayoutSerialization:
    """Test JSON serialization of layout."""

    def test_to_json(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        layout = compute_layout(
            tree,
            conv_path=[shas[0], shas[1], shas[2]],
            compaction_markers=[shas[0]],
            agent_position=shas[2],
        )

        j = layout.to_json()
        import json
        serialized = json.dumps(j)
        assert serialized

        assert isinstance(j["nodes"], list)
        assert isinstance(j["edges"], list)
        assert j["width"] >= 0
        assert j["height"] >= 0

    def test_empty_tree_layout(self):
        from alancode.git_tree.model import AGTTree
        tree = AGTTree()
        layout = compute_layout(tree)
        assert layout.nodes == []
        assert layout.edges == []
        j = layout.to_json()
        assert j["width"] == 0.0
