"""Dual-path fusion: merge and rank results from vector + tree search."""

from __future__ import annotations

from memory_atlas.retrieval.vector_search import SearchResult


class FusionRanker:
    """Merge results from vector search and tree search with weighted scoring."""

    def __init__(self, vector_weight: float = 0.6, tree_weight: float = 0.4):
        self.vector_weight = vector_weight
        self.tree_weight = tree_weight

    def fuse(
        self,
        vector_results: list[SearchResult],
        tree_results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Merge, deduplicate, and re-rank results from both paths."""
        scored: dict[str, SearchResult] = {}

        # Process vector results
        for r in vector_results:
            mid = r.record.id
            if mid not in scored:
                scored[mid] = SearchResult(
                    record=r.record, score=0.0, source="fusion"
                )
            scored[mid].score += r.score * self.vector_weight

        # Process tree results
        for r in tree_results:
            mid = r.record.id
            if mid not in scored:
                scored[mid] = SearchResult(
                    record=r.record, score=0.0, source="fusion"
                )
            scored[mid].score += r.score * self.tree_weight

        # Boost for appearing in both paths
        vector_ids = {r.record.id for r in vector_results}
        tree_ids = {r.record.id for r in tree_results}
        overlap = vector_ids & tree_ids
        for mid in overlap:
            if mid in scored:
                scored[mid].score *= 1.2  # 20% boost for dual-path hit

        # Sort by fused score
        ranked = sorted(scored.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]
