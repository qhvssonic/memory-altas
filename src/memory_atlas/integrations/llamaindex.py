"""LlamaIndex integration: MemoryAtlas as a retriever/tool for LlamaIndex agents."""

from __future__ import annotations

from typing import Any

from memory_atlas.engine import MemoryEngine


class MemoryAtlasRetriever:
    """LlamaIndex-compatible retriever backed by MemoryAtlas.

    Usage:
        from memory_atlas.integrations.llamaindex import MemoryAtlasRetriever
        retriever = MemoryAtlasRetriever(storage_path="./memory")
        nodes = retriever.retrieve("auth token bug")
    """

    def __init__(self, storage_path: str = "./memory_atlas_data", top_k: int = 5, **kwargs):
        self.engine = MemoryEngine(storage_path=storage_path, **kwargs)
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve memories as a list of dicts (LlamaIndex NodeWithScore style)."""
        memories = self.engine.retrieve(query, top_k=self.top_k)
        return [
            {
                "node_id": mem.id,
                "text": mem.display_text,
                "score": mem.importance,
                "metadata": {
                    "lod": mem.lod,
                    "tier": mem.tier.value if hasattr(mem.tier, "value") else str(mem.tier),
                    "entities": mem.entities,
                },
            }
            for mem in memories
        ]

    def close(self) -> None:
        self.engine.close()
