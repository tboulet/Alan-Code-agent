"""Tests for AGT data model and tree parsing (Phase 1)."""

import pytest

from alancode.git_tree.model import CURRENT_NODE_SHA, AGTNode, AGTTree, NodeType
from alancode.git_tree.parser import parse_git_tree
from tests.integration.git_helpers import GitTestRepo


class TestAGTNodeBasics:
    """Test AGTNode and AGTTree data structures."""

    def test_node_to_dict(self):
        node = AGTNode(
            sha="abc123def456",
            short_sha="abc123d",
            message="Test commit",
            author="Test User",
            timestamp="2025-01-01T00:00:00+00:00",
            parents=["parent1"],
            node_type=NodeType.ALAN_COMMIT,
            branches=["main"],
            is_head=True,
        )
        d = node.to_dict()
        assert d["sha"] == "abc123def456"
        assert d["node_type"] == "alan_commit"
        assert d["branches"] == ["main"]
        assert d["is_head"] is True

    def test_empty_tree(self):
        tree = AGTTree()
        assert tree.commit_count == 0
        assert tree.head_sha is None
        assert not tree.is_dirty
        assert tree.get_mainline() == []

    def test_tree_to_json(self):
        tree = AGTTree(
            nodes={"abc": AGTNode(
                sha="abc", short_sha="abc", message="m",
                author="a", timestamp="t", parents=[],
            )},
            root_shas=["abc"],
            head_sha="abc",
        )
        j = tree.to_json()
        assert j["commit_count"] == 1
        assert j["head_sha"] == "abc"
        assert len(j["nodes"]) == 1


class TestParseLinearHistory:
    """Test parsing a linear (no branches) git history."""

    def test_basic_linear(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        # 3 commits + 1 initial from git_repo.init()
        assert tree.commit_count == 4
        assert tree.head_sha == shas[2]
        assert tree.current_branch == "main"
        assert not tree.is_dirty

    def test_parent_child_links(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        # Check parent links
        node2 = tree.get_node(shas[2])
        assert node2 is not None
        assert shas[1] in node2.parents

        node1 = tree.get_node(shas[1])
        assert node1 is not None
        assert shas[0] in node1.parents

        # Check child links (inverse)
        node0 = tree.get_node(shas[0])
        assert node0 is not None
        assert shas[1] in node0.children

    def test_head_marked(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        tree = parse_git_tree(str(git_repo.path))

        head_node = tree.get_node(shas[1])
        assert head_node is not None
        assert head_node.is_head

        # Non-HEAD nodes should not be marked
        other_node = tree.get_node(shas[0])
        assert other_node is not None
        assert not other_node.is_head

    def test_root_shas(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        # Only the initial commit is a root
        assert len(tree.root_shas) == 1
        root = tree.get_node(tree.root_shas[0])
        assert root is not None
        assert root.parents == []

    def test_walk_ancestors(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(5)
        tree = parse_git_tree(str(git_repo.path))

        ancestors = tree.walk_ancestors(shas[4], 3)
        assert len(ancestors) == 3
        assert ancestors[0] == shas[3]
        assert ancestors[1] == shas[2]
        assert ancestors[2] == shas[1]

    def test_walk_ancestors_stops_at_root(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(2)
        tree = parse_git_tree(str(git_repo.path))

        # Ask for more ancestors than exist
        ancestors = tree.walk_ancestors(shas[1], 100)
        assert len(ancestors) <= 3  # shas[0] + initial + maybe one more

    def test_mainline(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))

        mainline = tree.get_mainline()
        assert mainline[0] == shas[2]  # HEAD first
        assert shas[1] in mainline
        assert shas[0] in mainline


class TestParseBranchingHistory:
    """Test parsing repos with branches and merges."""

    def test_branching(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))

        # All commits should be present
        for key in ["c1", "c2", "c3", "c4", "c5"]:
            assert tree.sha_exists(result[key]), f"Missing {key}"

        # Merge commit (c5) should have 2 parents
        merge_node = tree.get_node(result["c5"])
        assert merge_node is not None
        assert len(merge_node.parents) == 2

    def test_branch_names(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))

        # HEAD is c5 on main
        head = tree.get_node(result["c5"])
        assert head is not None
        assert head.is_head
        assert "main" in head.branches

    def test_feature_branch_node(self, git_repo: GitTestRepo):
        result = git_repo.build_branching_history()
        tree = parse_git_tree(str(git_repo.path))

        # c4 is on feature branch
        c4 = tree.get_node(result["c4"])
        assert c4 is not None
        assert "feature" in c4.branches


class TestAlanVsExternalClassification:
    """Test that alan_commits set correctly classifies nodes."""

    def test_classification(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)

        # Mark shas[1] as alan commit
        tree = parse_git_tree(str(git_repo.path), alan_commits={shas[1]})

        alan_node = tree.get_node(shas[1])
        assert alan_node is not None
        assert alan_node.node_type == NodeType.ALAN_COMMIT

        external_node = tree.get_node(shas[0])
        assert external_node is not None
        assert external_node.node_type == NodeType.EXTERNAL_COMMIT

    def test_all_alan(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path), alan_commits=set(shas))

        for sha in shas:
            node = tree.get_node(sha)
            assert node is not None
            assert node.node_type == NodeType.ALAN_COMMIT

    def test_no_alan(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        tree = parse_git_tree(str(git_repo.path))  # No alan_commits

        for sha in shas:
            node = tree.get_node(sha)
            assert node is not None
            assert node.node_type == NodeType.EXTERNAL_COMMIT


class TestDirtyWorkingTree:
    """Test detection of uncommitted changes."""

    def test_dirty_tree_has_current_node(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(1)
        git_repo.write_file("dirty.txt", "uncommitted")

        tree = parse_git_tree(str(git_repo.path))

        assert tree.is_dirty
        current = tree.get_node(CURRENT_NODE_SHA)
        assert current is not None
        assert current.node_type == NodeType.CURRENT_NODE
        assert current.message == "Uncommitted changes"

    def test_dirty_node_has_head_as_parent(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(1)
        git_repo.write_file("dirty.txt", "uncommitted")

        tree = parse_git_tree(str(git_repo.path))

        current = tree.get_node(CURRENT_NODE_SHA)
        assert current is not None
        assert shas[0] in current.parents

        # HEAD node should have current as child
        head = tree.get_node(shas[0])
        assert head is not None
        assert CURRENT_NODE_SHA in head.children

    def test_clean_tree_no_current_node(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)

        tree = parse_git_tree(str(git_repo.path))

        assert not tree.is_dirty
        assert tree.get_node(CURRENT_NODE_SHA) is None

    def test_commit_count_excludes_current_node(self, git_repo: GitTestRepo):
        git_repo.build_linear_history(2)
        git_repo.write_file("dirty.txt", "uncommitted")

        tree = parse_git_tree(str(git_repo.path))

        # 2 commits + 1 initial = 3 real commits, current node excluded
        assert tree.commit_count == 3
        assert len(tree.nodes) == 4  # 3 + current node


class TestDetachedHead:
    """Test parsing when HEAD is detached."""

    def test_detached_head(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        git_repo.checkout(shas[1])  # Detach HEAD at shas[1]

        tree = parse_git_tree(str(git_repo.path))

        assert tree.current_branch is None
        assert tree.head_sha == shas[1]

        head_node = tree.get_node(shas[1])
        assert head_node is not None
        assert head_node.is_head


class TestEmptyRepo:
    """Test parsing a repo with only the initial commit."""

    def test_init_only(self, git_repo: GitTestRepo):
        # git_repo fixture already has 1 initial commit (.gitignore)
        tree = parse_git_tree(str(git_repo.path))

        assert tree.commit_count == 1
        assert tree.head_sha is not None
        assert len(tree.root_shas) == 1
        assert tree.current_branch == "main"


class TestToJson:
    """Test JSON serialization round-trip."""

    def test_serializable(self, git_repo: GitTestRepo):
        shas = git_repo.build_linear_history(3)
        git_repo.write_file("dirty.txt", "uncommitted")

        tree = parse_git_tree(str(git_repo.path), alan_commits={shas[1]})
        j = tree.to_json()

        # Should be a plain dict with no non-serializable objects
        import json
        serialized = json.dumps(j)
        assert serialized  # No exception

        # Check structure
        assert isinstance(j["nodes"], list)
        assert j["is_dirty"] is True
        assert j["current_branch"] == "main"
        assert j["commit_count"] == 4  # 3 + initial

        # Find the alan commit in nodes
        alan_nodes = [n for n in j["nodes"] if n["node_type"] == "alan_commit"]
        assert len(alan_nodes) == 1
        assert alan_nodes[0]["sha"] == shas[1]

        # Find the current node
        current_nodes = [n for n in j["nodes"] if n["node_type"] == "current"]
        assert len(current_nodes) == 1
