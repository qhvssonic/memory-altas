"""Tests for fusion ranker."""

from memory_atlas.core.registry import MemoryRecord
from memory_atlas.retrieval.vector_search import SearchResult
from memory_atlas.retrieval.fusion import FusionRanker


def _sr(mid: str, score: float, source: str = "vector") -> SearchResult:
    return SearchResult(record=MemoryRecord(id=mid, label=mid), score=score, source=source)


class TestFusionRanker:
    def test_basic_fusion(self):
        ranker = FusionRanker(vector_weight=0.6, tree_weight=0.4)
        vec = [_sr("a", 0.9), _sr("b", 0.5)]
        tree = [_sr("b", 0.8, "tree"), _sr("c", 0.7, "tree")]
        results = ranker.fuse(vec, tree, top_k=10)
        ids = [r.record.id for r in results]
        # b appears in both → boosted
        assert "b" in ids
        assert len(results) == 3

    def test_dual_path_boost(self):
        ranker = FusionRanker(vector_weight=0.5, tree_weight=0.5)
        vec = [_sr("a", 0.6)]
        tree = [_sr("a", 0.6, "tree"), _sr("b", 0.9, "tree")]
        results = ranker.fuse(vec, tree, top_k=10)
        # a has dual-path boost (1.2x), b only tree
        a_score = next(r.score for r in results if r.record.id == "a")
        b_score = next(r.score for r in results if r.record.id == "b")
        # a: (0.6*0.5 + 0.6*0.5) * 1.2 = 0.72
        # b: 0.9*0.5 = 0.45
        assert a_score > b_score

    def test_top_k_limit(self):
        ranker = FusionRanker()
        vec = [_sr(f"v{i}", 0.5) for i in range(10)]
        tree = [_sr(f"t{i}", 0.5, "tree") for i in range(10)]
        results = ranker.fuse(vec, tree, top_k=5)
        assert len(results) == 5

    def test_empty_inputs(self):
        ranker = FusionRanker()
        assert ranker.fuse([], [], top_k=5) == []
