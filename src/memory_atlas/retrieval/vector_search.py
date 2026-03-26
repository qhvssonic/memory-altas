"""Vector similarity search via DuckDB."""

from __future__ import annotations

from dataclasses import dataclass

from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.llm.embedder import Embedder


@dataclass
class SearchResult:
    """A single search result with score."""

    record: MemoryRecord
    score: float
    source: str = "vector"  # vector / tree / cache


class VectorSearch:
    """Embedding-based similarity search using DuckDB vector operations."""

    def __init__(self, registry: Registry, embedder: Embedder):
        self.registry = registry
        self.embedder = embedder

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search memories by vector similarity."""
        query_emb = self.embedder.embed([query])[0]
        results = self.registry.vector_search(query_emb, top_k=top_k)
        return [
            SearchResult(record=rec, score=score, source="vector")
            for rec, score in results
            if score > 0
        ]
