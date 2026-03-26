"""LOD (Level of Detail) Manager: dynamic precision switching for memories."""

from __future__ import annotations

from memory_atlas.storage.cache import CachedMemory
from memory_atlas.storage.file_store import FileStore


class LODManager:
    """Manage memory precision levels: L0 (label), L1 (summary), L2 (full content).

    Like game engine LOD, closer/more-relevant memories get higher detail.
    """

    def __init__(self, file_store: FileStore, max_tokens: int = 2000):
        self.file_store = file_store
        self.max_tokens = max_tokens

    def assign_lod(
        self, memories: list[CachedMemory], scores: dict[str, float] | None = None
    ) -> list[CachedMemory]:
        """Assign LOD levels based on relevance scores and token budget.

        Strategy:
        - Top 20% by score → L1 (summary)
        - Rest → L0 (label)
        - L2 is only loaded on explicit expand request
        """
        if not memories:
            return memories

        scores = scores or {}
        # Sort by score descending
        sorted_mems = sorted(
            memories,
            key=lambda m: scores.get(m.id, m.importance),
            reverse=True,
        )

        token_budget = self.max_tokens
        top_count = max(1, len(sorted_mems) // 5)  # top 20%

        for i, mem in enumerate(sorted_mems):
            if i < top_count:
                mem.lod = "L1"
            else:
                mem.lod = "L0"

            est_tokens = mem.token_estimate()
            token_budget -= est_tokens
            if token_budget <= 0:
                # Downgrade remaining to L0
                for j in range(i + 1, len(sorted_mems)):
                    sorted_mems[j].lod = "L0"
                break

        return sorted_mems

    def expand_to_l2(self, memory: CachedMemory) -> CachedMemory:
        """Load full L2 content from disk for a specific memory."""
        if memory.content:
            memory.lod = "L2"
            return memory

        chunk = self.file_store.load_chunk(memory.id)
        if chunk:
            memory.content = chunk.content
            memory.lod = "L2"
        return memory

    def format_memory_view(self, memories: list[CachedMemory]) -> str:
        """Format memories at their current LOD into a context string."""
        if not memories:
            return ""

        lines = ["[Memory Context]"]
        for mem in memories:
            prefix = f"[{mem.lod}]"
            lines.append(f"{prefix} {mem.display_text}")
        lines.append("[/Memory Context]")
        return "\n".join(lines)
