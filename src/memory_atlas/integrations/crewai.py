"""CrewAI integration: MemoryAtlas as a tool for CrewAI agents."""

from __future__ import annotations


from memory_atlas.engine import MemoryEngine


class MemoryAtlasTool:
    """CrewAI-compatible tool backed by MemoryAtlas.

    Usage:
        from memory_atlas.integrations.crewai import MemoryAtlasTool
        tool = MemoryAtlasTool(storage_path="./memory")
        result = tool.run("search for auth bugs")
    """

    name: str = "memory_atlas"
    description: str = (
        "Search long-term memory for relevant context. "
        "Input is a natural language query. Returns matching memories."
    )

    def __init__(self, storage_path: str = "./memory_atlas_data", top_k: int = 5, **kwargs):
        self.engine = MemoryEngine(storage_path=storage_path, **kwargs)
        self.top_k = top_k

    def run(self, query: str) -> str:
        """Execute a memory search and return formatted results."""
        memories = self.engine.retrieve(query, top_k=self.top_k)
        if not memories:
            return "No relevant memories found."
        return self.engine.format_memories(memories)

    def close(self) -> None:
        self.engine.close()
