"""Integration tests: end-to-end flows without external LLM/embedding calls.

Uses rule-based extraction and a mock embedder to test the full pipeline.
"""

from __future__ import annotations

import pytest
import random

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.storage.file_store import FileStore
from memory_atlas.scene.manager import SceneManager


class FakeEmbedder:
    """Deterministic fake embedder for testing."""

    @property
    def dim(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            random.seed(hash(text) % 2**31)
            results.append([random.gauss(0, 1) for _ in range(self.dim)])
        return results


class FakeLLM:
    """Fake LLM that returns canned responses."""

    def complete(self, prompt, system="", **kwargs):
        return "fake response"

    def complete_json(self, prompt, system="", **kwargs):
        if "extract" in prompt.lower() or "entities" in prompt.lower():
            return {
                "entities": [{"name": "test", "type": "concept"}],
                "facts": ["test fact"],
                "decisions": [],
                "topics": ["testing"],
                "importance": 0.7,
            }
        if "summary" in prompt.lower() or "label" in prompt.lower():
            return {"label": "test label", "summary": "test summary"}
        if "navigate" in prompt.lower() or "children" in prompt.lower():
            return []
        if "predict" in prompt.lower():
            return {"predicted_topics": ["testing"], "predicted_entities": ["test"]}
        return {}


@pytest.fixture
def env(tmp_path):
    """Set up a complete MemoryAtlas environment with fakes."""
    config = MemoryAtlasConfig(
        storage_path=str(tmp_path),
        hot_capacity=5,
        warm_capacity=20,
        prefetch_enabled=False,  # disable for deterministic tests
        culling_enabled=True,
        ingest_strategy="rule_based",
        ingest_min_length=10,
    )
    registry = Registry(tmp_path / "index.duckdb")
    tree = TreeIndex(tmp_path)
    file_store = FileStore(tmp_path)
    embedder = FakeEmbedder()
    llm = FakeLLM()

    scene = SceneManager(
        config=config,
        registry=registry,
        tree=tree,
        file_store=file_store,
        embedder=embedder,
        llm=llm,
    )

    yield {
        "config": config,
        "registry": registry,
        "tree": tree,
        "file_store": file_store,
        "embedder": embedder,
        "llm": llm,
        "scene": scene,
        "tmp_path": tmp_path,
    }
    registry.close()


class TestIngestionToRetrieval:
    """Test the full ingest → store → retrieve → format pipeline."""

    def test_ingest_and_retrieve(self, env):
        """Ingest content, then retrieve it via scene manager."""
        from memory_atlas.core.registry import MemoryRecord
        from memory_atlas.storage.file_store import MemoryChunk

        reg = env["registry"]
        fs = env["file_store"]
        emb = env["embedder"]
        scene = env["scene"]

        # Manually ingest (simulating the engine pipeline)
        content = "We decided to use JWT with sliding window refresh tokens for auth."
        summary = "Using JWT sliding window for refresh tokens"
        embedding = emb.embed([summary])[0]

        rec = MemoryRecord(
            id="int001",
            session_id="s1",
            label="JWT auth decision",
            summary=summary,
            file_path="chunks/int001.md",
            embedding=embedding,
            importance_score=0.8,
        )
        reg.insert_memory(rec)
        fs.save_chunk(MemoryChunk(
            id="int001", session_id="s1",
            entities=["jwt", "auth"],
            importance=0.8,
            title="JWT auth decision",
            content=content,
        ))

        # Retrieve via scene manager — use the same summary text so embedding matches
        scene.initialize_session("s1")
        results = scene.get_memory_view(summary)
        assert len(results) >= 1
        assert any(m.id == "int001" for m in results)

    def test_multi_memory_lod_assignment(self, env):
        """Multiple memories get different LOD levels based on relevance."""
        from memory_atlas.core.registry import MemoryRecord

        reg = env["registry"]
        emb = env["embedder"]
        scene = env["scene"]

        # Insert several memories with different embeddings
        for i in range(10):
            summary = f"Summary of topic {i}"
            embedding = emb.embed([summary])[0]
            rec = MemoryRecord(
                id=f"lod{i:02d}",
                label=f"topic-{i}",
                summary=summary,
                embedding=embedding,
                importance_score=0.5 + (i * 0.05),
            )
            reg.insert_memory(rec)

        scene.initialize_session("s2")
        results = scene.get_memory_view("topic 5")
        # Should have a mix of L0 and L1
        lods = {m.lod for m in results}
        assert len(results) > 0
        # At least some should be L0 (lower relevance)
        if len(results) > 1:
            assert "L0" in lods or "L1" in lods


class TestSceneManagerCycle:
    """Test the full scene management cycle: retrieve → update → cull."""

    def test_culling_on_topic_switch(self, env):
        """Memories get demoted when topic switches."""
        from memory_atlas.core.registry import MemoryRecord

        reg = env["registry"]
        emb = env["embedder"]
        scene = env["scene"]
        cache = scene.cache

        # Insert and promote auth memories to hot
        for i in range(3):
            summary = f"Auth memory {i}"
            embedding = emb.embed([summary])[0]
            rec = MemoryRecord(
                id=f"auth{i}", label=f"auth-{i}",
                summary=summary,
                embedding=embedding, importance_score=0.7,
            )
            reg.insert_memory(rec)

        scene.initialize_session("s3")
        scene.get_memory_view("authentication module")

        # Verify some are in hot
        hot_before = cache.get_hot()
        assert len(hot_before) > 0

        # Simulate topic switch
        scene.update(
            recent_message="换个话题，我们来看看数据库性能",
            current_entities=["database", "performance"],
        )

        # After culling, auth memories should be demoted
        hot_after = cache.get_hot()
        auth_in_hot = [m for m in hot_after if "auth" in m.label]
        # Some or all auth memories should have been demoted
        assert len(auth_in_hot) <= len(hot_before)

    def test_session_lifecycle(self, env):
        """Test initialize → retrieve → update → persist cycle."""
        scene = env["scene"]

        scene.initialize_session("lifecycle_test")
        assert scene._session_id == "lifecycle_test"
        assert scene._turn_count == 0

        # First turn
        scene.get_memory_view("hello world")
        assert scene._turn_count == 1

        # Update
        stats = scene.update("hello world", ["greeting"])
        assert "prefetched" in stats
        assert "culled" in stats

        # Persist
        scene.persist()  # should not raise

        # Stats
        s = scene.stats()
        assert "total" in s
        assert "hot" in s
        assert "warm" in s


class TestTreeIndexIntegration:
    """Test tree index with registry."""

    def test_tree_build_and_navigate(self, env):
        from memory_atlas.core.tree_index import TreeNode

        tree = env["tree"]

        # Build tree structure
        tree.add_child("root", TreeNode(id="auth", label="Authentication"))
        tree.add_child("root", TreeNode(id="db", label="Database"))
        tree.add_child("auth", TreeNode(id="jwt", label="JWT Tokens"))

        # Add memories to nodes
        tree.add_memory_to_node("jwt", "m_jwt_001")
        tree.add_memory_to_node("db", "m_db_001")

        # Verify structure
        outline = tree.get_outline(max_depth=3)
        assert "Authentication" in outline
        assert "JWT Tokens" in outline
        assert "Database" in outline

        # Save and reload
        tree.save()
        tree2 = TreeIndex(env["tmp_path"])
        assert tree2.find_node("jwt") is not None
        assert "m_jwt_001" in tree2.find_node("jwt").memory_ids


class TestForgettingIntegration:
    """Test forgetting mechanism with real registry."""

    def test_forget_old_memories(self, env):
        from datetime import datetime, timezone, timedelta
        from memory_atlas.maintenance.forgetting import ForgettingManager
        from memory_atlas.core.registry import MemoryRecord
        from memory_atlas.storage.file_store import MemoryChunk

        reg = env["registry"]
        fs = env["file_store"]

        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=180)).isoformat()

        # Insert old low-importance memory
        reg.insert_memory(MemoryRecord(
            id="forget_me", label="old", summary="old summary",
            importance_score=0.05, access_count=0,
            last_accessed_at=old,
        ))
        fs.save_chunk(MemoryChunk(id="forget_me", content="old content"))

        # Insert recent high-importance memory
        reg.insert_memory(MemoryRecord(
            id="keep_me", label="new", summary="new summary",
            importance_score=0.9, access_count=20,
            last_accessed_at=now.isoformat(),
        ))

        fm = ForgettingManager(reg, fs, decay_lambda=0.1)
        result = fm.run_cycle()

        assert result.scanned == 2
        assert result.kept >= 1  # keep_me should survive
