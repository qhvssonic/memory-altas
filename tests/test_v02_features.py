"""Tests for v0.2.0 features: import/export, multi-user, clusters, embedder."""

import json
import pytest

from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.io.exporter import Exporter
from memory_atlas.io.importer import Importer
from memory_atlas.core.cluster import ClusterManager
from memory_atlas.llm.embedder import create_embedder


@pytest.fixture
def env(tmp_path):
    reg = Registry(tmp_path / "test.duckdb")
    fs = FileStore(tmp_path)
    yield reg, fs, tmp_path
    reg.close()


class TestExportImport:
    def test_export_and_import_roundtrip(self, env):
        reg, fs, tmp = env
        # Insert a memory
        reg.insert_memory(MemoryRecord(
            id="exp1", session_id="s1", label="test export",
            summary="export summary", embedding=[0.1, 0.2],
            importance_score=0.8,
        ))
        fs.save_chunk(MemoryChunk(id="exp1", content="full content here"))
        eid = reg.upsert_entity("jwt", "concept")
        reg.link_memory_entity("exp1", eid)

        # Export
        export_path = tmp / "export.json"
        exporter = Exporter(reg, fs)
        stats = exporter.export_all(str(export_path))
        assert stats["memories_exported"] == 1
        assert export_path.exists()

        # Verify JSON structure
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert data["version"] == "0.2.0"
        assert len(data["memories"]) == 1
        assert data["memories"][0]["content"] == "full content here"

        # Import into a fresh registry
        reg2 = Registry(tmp / "test2.duckdb")
        fs2 = FileStore(tmp / "store2")
        importer = Importer(reg2, fs2)
        istats = importer.import_file(str(export_path), mode="merge")
        assert istats["imported"] == 1
        assert reg2.get_memory("exp1") is not None
        reg2.close()

    def test_import_merge_skips_existing(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="dup1", label="original"))

        # Create export with same ID
        export_data = {
            "version": "0.2.0", "format": "memory_atlas_export",
            "memories": [{"id": "dup1", "label": "imported", "embedding": []}],
            "entities": [], "tree_nodes": [],
        }
        export_path = tmp / "dup.json"
        export_path.write_text(json.dumps(export_data), encoding="utf-8")

        importer = Importer(reg, fs)
        stats = importer.import_file(str(export_path), mode="merge")
        assert stats["skipped"] == 1
        assert reg.get_memory("dup1").label == "original"  # not overwritten

    def test_import_overwrite(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="ow1", label="old"))

        export_data = {
            "version": "0.2.0", "format": "memory_atlas_export",
            "memories": [{"id": "ow1", "label": "new", "embedding": []}],
            "entities": [], "tree_nodes": [],
        }
        export_path = tmp / "ow.json"
        export_path.write_text(json.dumps(export_data), encoding="utf-8")

        importer = Importer(reg, fs)
        stats = importer.import_file(str(export_path), mode="overwrite")
        assert stats["overwritten"] == 1
        assert reg.get_memory("ow1").label == "new"

    def test_export_session_filter(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="a", session_id="s1", label="a"))
        reg.insert_memory(MemoryRecord(id="b", session_id="s2", label="b"))

        exporter = Exporter(reg, fs)
        stats = exporter.export_session("s1", str(tmp / "s1.json"))
        assert stats["memories_exported"] == 1


class TestMultiUser:
    def test_user_id_isolation(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="u1m1", user_id="alice", label="alice mem"))
        reg.insert_memory(MemoryRecord(id="u2m1", user_id="bob", label="bob mem"))

        alice_mems = reg.list_memories(user_id="alice")
        assert len(alice_mems) == 1
        assert alice_mems[0].user_id == "alice"

        bob_mems = reg.list_memories(user_id="bob")
        assert len(bob_mems) == 1

    def test_agent_id_isolation(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="a1", agent_id="coder", label="code"))
        reg.insert_memory(MemoryRecord(id="a2", agent_id="writer", label="write"))

        coder_mems = reg.list_memories(agent_id="coder")
        assert len(coder_mems) == 1
        assert coder_mems[0].agent_id == "coder"

    def test_combined_filter(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(
            id="c1", user_id="alice", agent_id="coder", session_id="s1", label="x"
        ))
        reg.insert_memory(MemoryRecord(
            id="c2", user_id="alice", agent_id="writer", session_id="s1", label="y"
        ))
        reg.insert_memory(MemoryRecord(
            id="c3", user_id="bob", agent_id="coder", session_id="s1", label="z"
        ))

        results = reg.list_memories(user_id="alice", agent_id="coder")
        assert len(results) == 1
        assert results[0].id == "c1"


class TestCluster:
    def test_create_and_get(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        cluster = cm.create_cluster(
            name="auth-cluster",
            memory_ids=["m1", "m2", "m3"],
            summary="All auth-related memories",
            entity_tags=["auth", "jwt"],
        )
        assert cluster.id
        got = cm.get_cluster(cluster.id)
        assert got is not None
        assert got.name == "auth-cluster"
        assert got.memory_ids == ["m1", "m2", "m3"]

    def test_auto_cluster_by_entity(self, env):
        reg, fs, tmp = env
        reg.insert_memory(MemoryRecord(id="e1", label="jwt mem 1"))
        reg.insert_memory(MemoryRecord(id="e2", label="jwt mem 2"))
        eid = reg.upsert_entity("jwt", "concept")
        reg.link_memory_entity("e1", eid)
        reg.link_memory_entity("e2", eid)

        cm = ClusterManager(reg)
        cluster = cm.auto_cluster_by_entity("jwt")
        assert cluster is not None
        assert len(cluster.memory_ids) == 2

    def test_list_and_delete(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        c1 = cm.create_cluster("c1", ["m1"])
        cm.create_cluster("c2", ["m2"])
        assert len(cm.list_clusters()) == 2

        cm.delete_cluster(c1.id)
        assert len(cm.list_clusters()) == 1


class TestEmbedderExtensions:
    def test_ollama_embedder_creation(self):
        # Just test factory doesn't crash (actual Ollama not available)
        emb = create_embedder(mode="ollama", model="ollama/nomic-embed-text", dim=768)
        assert emb.dim == 768

    def test_custom_embedder(self):
        def my_embed(texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        emb = create_embedder(mode="custom", embed_fn=my_embed, dim=3)
        assert emb.dim == 3
        result = emb.embed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [1.0, 0.0, 0.0]

    def test_custom_requires_fn(self):
        with pytest.raises(ValueError, match="embed_fn"):
            create_embedder(mode="custom")


class TestAutoClusterOnIngest:
    """Test that clusters are auto-created when entity memory count hits threshold."""

    def test_auto_cluster_triggers_at_threshold(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)

        # Insert 2 memories linked to "jwt" — below threshold of 3
        for i in range(2):
            reg.insert_memory(MemoryRecord(id=f"ac{i}", label=f"jwt mem {i}"))
            eid = reg.upsert_entity("jwt", "concept")
            reg.link_memory_entity(f"ac{i}", eid)

        # Should not cluster yet
        result = cm.auto_update_for_memory("ac1", ["jwt"], threshold=3)
        assert len(result) == 0

        # Add 3rd memory — hits threshold
        reg.insert_memory(MemoryRecord(id="ac2", label="jwt mem 2"))
        eid = reg.upsert_entity("jwt", "concept")
        reg.link_memory_entity("ac2", eid)

        result = cm.auto_update_for_memory("ac2", ["jwt"], threshold=3)
        assert len(result) == 1
        assert "jwt" in result[0].entity_tags
        assert len(result[0].memory_ids) == 3

    def test_auto_cluster_updates_existing(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)

        # Create initial cluster with 3 memories
        for i in range(3):
            reg.insert_memory(MemoryRecord(id=f"up{i}", label=f"auth mem {i}"))
            eid = reg.upsert_entity("auth", "concept")
            reg.link_memory_entity(f"up{i}", eid)

        cm.auto_update_for_memory("up2", ["auth"], threshold=3)
        clusters_before = cm.list_clusters()
        assert len(clusters_before) == 1
        assert len(clusters_before[0].memory_ids) == 3

        # Add 4th memory — should update existing cluster, not create new one
        reg.insert_memory(MemoryRecord(id="up3", label="auth mem 3"))
        eid = reg.upsert_entity("auth", "concept")
        reg.link_memory_entity("up3", eid)

        cm.auto_update_for_memory("up3", ["auth"], threshold=3)
        clusters_after = cm.list_clusters()
        assert len(clusters_after) == 1  # still 1 cluster, not 2
        assert len(clusters_after[0].memory_ids) == 4  # now has 4 members


class TestClusterLOD:
    def test_cluster_lod_l0(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        c = cm.create_cluster("auth", ["m1", "m2", "m3"])
        text = cm.get_cluster_lod(c.id, "L0")
        assert "auth" in text
        assert "3 memories" in text

    def test_cluster_lod_l1_generates_summary(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        # Insert real memories so L1 can read labels
        for i in range(3):
            reg.insert_memory(MemoryRecord(id=f"cl{i}", label=f"jwt issue {i}"))
        c = cm.create_cluster("jwt-cluster", [f"cl{i}" for i in range(3)], entity_tags=["jwt"])
        text = cm.get_cluster_lod(c.id, "L1")
        assert "jwt" in text
        assert "3 memories" in text

    def test_cluster_lod_l2_all_summaries(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        for i in range(3):
            reg.insert_memory(MemoryRecord(
                id=f"l2_{i}", label=f"topic {i}", summary=f"detail about topic {i}"
            ))
        c = cm.create_cluster("topics", [f"l2_{i}" for i in range(3)])
        text = cm.get_cluster_lod(c.id, "L2")
        assert "detail about topic 0" in text
        assert "detail about topic 2" in text

    def test_cluster_lod_nonexistent(self, env):
        reg, fs, tmp = env
        cm = ClusterManager(reg)
        assert cm.get_cluster_lod("nope") is None
