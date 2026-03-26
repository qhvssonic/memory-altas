"""Tests for tree index."""

import pytest
from memory_atlas.core.tree_index import TreeIndex, TreeNode


@pytest.fixture
def tree(tmp_path):
    return TreeIndex(tmp_path)


class TestTreeIndex:
    def test_initial_root(self, tree):
        assert tree.root.id == "root"
        assert tree.root.node_type == "root"

    def test_add_child(self, tree):
        child = TreeNode(id="auth", label="Authentication", node_type="topic")
        assert tree.add_child("root", child)
        found = tree.find_node("auth")
        assert found is not None
        assert found.depth == 1

    def test_add_memory_to_node(self, tree):
        child = TreeNode(id="db", label="Database")
        tree.add_child("root", child)
        tree.add_memory_to_node("db", "m001")
        tree.add_memory_to_node("db", "m002")
        node = tree.find_node("db")
        assert "m001" in node.memory_ids
        assert "m002" in node.memory_ids

    def test_no_duplicate_memory(self, tree):
        child = TreeNode(id="x", label="X")
        tree.add_child("root", child)
        tree.add_memory_to_node("x", "m1")
        tree.add_memory_to_node("x", "m1")
        assert tree.find_node("x").memory_ids.count("m1") == 1

    def test_get_node_path(self, tree):
        tree.add_child("root", TreeNode(id="a", label="A"))
        tree.add_child("a", TreeNode(id="b", label="B"))
        path = tree.get_node_path("b")
        assert [n.id for n in path] == ["root", "a", "b"]

    def test_save_and_reload(self, tmp_path):
        t1 = TreeIndex(tmp_path)
        t1.add_child("root", TreeNode(id="topic1", label="Topic 1"))
        t1.add_memory_to_node("topic1", "m1")
        t1.save()

        t2 = TreeIndex(tmp_path)
        assert t2.find_node("topic1") is not None
        assert "m1" in t2.find_node("topic1").memory_ids

    def test_outline(self, tree):
        tree.add_child("root", TreeNode(id="a", label="Auth"))
        tree.add_child("a", TreeNode(id="b", label="JWT"))
        outline = tree.get_outline(max_depth=2)
        assert "Auth" in outline
        assert "JWT" in outline

    def test_add_child_nonexistent_parent(self, tree):
        child = TreeNode(id="orphan", label="Orphan")
        assert not tree.add_child("nonexistent", child)
