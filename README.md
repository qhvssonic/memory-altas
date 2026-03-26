# MemoryAtlas

> Game-engine inspired memory management for AI agents.

MemoryAtlas brings game engine resource management patterns to AI agent memory:
**predictive prefetching**, **frustum culling**, **LOD (Level of Detail)**, and
**three-tier caching** — concepts no existing memory system offers.

## Why MemoryAtlas?

Existing memory systems (Mem0, Zep, Letta) are all "passive retrieval" — query comes in,
search, return results. MemoryAtlas introduces **active scene management**:

- **Prefetching**: Predicts what memories you'll need next and preloads them
- **Frustum Culling**: Detects topic shifts and actively unloads irrelevant memories
- **LOD**: Three precision levels (L0 label / L1 summary / L2 full) — always token-efficient
- **Three-tier Cache**: Hot (active) → Warm (preloaded) → Cold (disk) — cache hits in microseconds

## Quick Start

```bash
pip install memory-atlas
# For local embeddings:
pip install memory-atlas[local-embedding]
```

### One-line LangChain Integration

```python
from memory_atlas.langchain import MemoryAtlasMiddleware

memory = MemoryAtlasMiddleware(
    storage_path="./my_agent_memory",
    embedding_model="local",
    max_memory_tokens=2000,
)

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[memory],  # That's it.
)
```

### Standalone Usage

```python
from memory_atlas import MemoryEngine

engine = MemoryEngine(storage_path="./memory", embedding_model="local")

# Ingest
engine.ingest("We decided to use JWT with sliding window refresh tokens.")

# Retrieve (scene manager handles caching, LOD, prefetching)
memories = engine.retrieve("authentication token strategy")
for m in memories:
    print(f"[{m.lod}] {m.display_text}")

# Expand to full detail
detail = engine.expand(memories[0].id)
```

## Architecture

```
Scene Manager (core differentiator)
├── Prefetcher      — predictive preloading
├── Frustum Culler  — active irrelevance removal
└── LOD Manager     — dynamic precision switching

Storage: DuckDB (metadata + vectors) + Markdown files (human-readable)
Cache: Hot (dict) → Warm (LRU) → Cold (disk)
Search: Vector similarity + Tree-based reasoning → Fusion ranking
```

## Key Metrics

| Metric | Target |
|---|---|
| Cache hit rate | > 60% |
| Prefetch accuracy | > 50% |
| Token savings (LOD) | > 40% |
| Cache retrieval latency | < 10ms |

## License

MIT
