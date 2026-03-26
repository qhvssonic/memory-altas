"""Tests for LOD manager."""

from memory_atlas.storage.cache import CachedMemory
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.scene.lod import LODManager


class TestLODManager:
    def test_assign_lod_top_get_l1(self):
        lod = LODManager(file_store=None, max_tokens=5000)
        mems = [
            CachedMemory(id="a", label="la", summary="summary a", importance=0.9),
            CachedMemory(id="b", label="lb", summary="summary b", importance=0.3),
            CachedMemory(id="c", label="lc", summary="summary c", importance=0.1),
            CachedMemory(id="d", label="ld", summary="summary d", importance=0.05),
            CachedMemory(id="e", label="le", summary="summary e", importance=0.02),
        ]
        scores = {"a": 0.9, "b": 0.3, "c": 0.1, "d": 0.05, "e": 0.02}
        result = lod.assign_lod(mems, scores)
        # Top 20% (1 out of 5) should be L1
        l1_count = sum(1 for m in result if m.lod == "L1")
        l0_count = sum(1 for m in result if m.lod == "L0")
        assert l1_count >= 1
        assert l0_count >= 1

    def test_assign_lod_empty(self):
        lod = LODManager(file_store=None)
        assert lod.assign_lod([]) == []

    def test_expand_to_l2(self, tmp_path):
        store = FileStore(tmp_path)
        store.save_chunk(MemoryChunk(id="x", content="full detail here"))
        lod = LODManager(file_store=store)
        mem = CachedMemory(id="x", label="lb", summary="sm")
        expanded = lod.expand_to_l2(mem)
        assert expanded.lod == "L2"
        assert "full detail" in expanded.content

    def test_format_memory_view(self):
        lod = LODManager(file_store=None)
        mems = [
            CachedMemory(id="a", label="tag A", lod="L0"),
            CachedMemory(id="b", summary="summary B", lod="L1"),
        ]
        text = lod.format_memory_view(mems)
        assert "[L0] tag A" in text
        assert "[L1] summary B" in text
        assert "[Memory Context]" in text

    def test_format_empty(self):
        lod = LODManager(file_store=None)
        assert lod.format_memory_view([]) == ""
