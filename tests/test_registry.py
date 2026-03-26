"""Tests for DuckDB registry."""

import pytest
import tempfile
from pathlib import Path
from memory_atlas.core.registry import Registry, MemoryRecord


@pytest.fixture
def registry(tmp_path):
    reg = Registry(tmp_path / "test.duckdb")
    yield reg
    reg.close()


class TestRegistry:
    def test_insert_and_get(self, registry):
        rec = MemoryRecord(id="m001", label="test label", summary="test summary")
        registry.insert_memory(rec)
        got = registry.get_memory("m001")
        assert got is not None
        assert got.label == "test label"
        assert got.summary == "test summary"

    def test_get_nonexistent(self, registry):
        assert registry.get_memory("nope") is None

    def test_update(self, registry):
        registry.insert_memory(MemoryRecord(id="m002", label="old"))
        registry.update_memory("m002", label="new")
        got = registry.get_memory("m002")
        assert got.label == "new"

    def test_delete(self, registry):
        registry.insert_memory(MemoryRecord(id="m003", label="del"))
        registry.delete_memory("m003")
        assert registry.get_memory("m003") is None

    def test_touch_memory(self, registry):
        registry.insert_memory(MemoryRecord(id="m004", label="touch"))
        registry.touch_memory("m004")
        got = registry.get_memory("m004")
        assert got.access_count == 1

    def test_list_memories(self, registry):
        registry.insert_memory(MemoryRecord(id="a", session_id="s1", label="a"))
        registry.insert_memory(MemoryRecord(id="b", session_id="s1", label="b"))
        registry.insert_memory(MemoryRecord(id="c", session_id="s2", label="c"))
        all_mems = registry.list_memories()
        assert len(all_mems) == 3
        s1_mems = registry.list_memories(session_id="s1")
        assert len(s1_mems) == 2

    def test_count(self, registry):
        assert registry.count_memories() == 0
        registry.insert_memory(MemoryRecord(id="x", label="x"))
        assert registry.count_memories() == 1

    def test_vector_search(self, registry):
        registry.insert_memory(MemoryRecord(
            id="v1", label="auth", embedding=[1.0, 0.0, 0.0]
        ))
        registry.insert_memory(MemoryRecord(
            id="v2", label="db", embedding=[0.0, 1.0, 0.0]
        ))
        results = registry.vector_search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) >= 1
        assert results[0][0].id == "v1"  # closest match

    def test_entities(self, registry):
        registry.insert_memory(MemoryRecord(id="e1", label="test"))
        eid = registry.upsert_entity("jwt", "concept")
        registry.link_memory_entity("e1", eid)
        entities = registry.get_entities_for_memory("e1")
        assert len(entities) == 1
        assert entities[0][1] == "jwt"

    def test_topic_transitions(self, registry):
        registry.record_transition("auth", "database", "s1")
        registry.record_transition("auth", "database", "s1")
        registry.record_transition("auth", "deploy", "s1")
        topics = registry.get_likely_next_topics("auth")
        assert len(topics) == 2
        assert topics[0][0] == "database"  # higher count
        assert topics[0][1] == 2

    def test_tree_nodes(self, registry):
        registry.insert_tree_node("root", "", "Root", node_type="root")
        registry.insert_tree_node("auth", "root", "Authentication", node_type="topic", depth=1)
        children = registry.get_tree_children("root")
        assert len(children) == 1
        assert children[0]["label"] == "Authentication"
