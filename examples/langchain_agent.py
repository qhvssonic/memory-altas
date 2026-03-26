"""Example: Using MemoryAtlas with a LangChain 1.0 agent.

This demonstrates the one-line integration pattern.
"""

from memory_atlas.langchain import MemoryAtlasMiddleware

# Initialize memory middleware
memory = MemoryAtlasMiddleware(
    storage_path="./my_agent_memory",
    embedding_model="local",       # Use local sentence-transformers
    max_memory_tokens=2000,        # Token budget for memory context
    prefetch_enabled=True,         # Enable predictive prefetching
    culling_enabled=True,          # Enable frustum culling
    ingest_strategy="rule_based",  # Auto-save without LLM evaluation
)

# --- Standalone usage (without LangChain) ---

from memory_atlas import MemoryEngine

engine = MemoryEngine(storage_path="./my_agent_memory", embedding_model="local")

# Ingest some content
ids = engine.ingest(
    "We discussed the JWT token expiration bug in the auth module. "
    "The refresh token has a race condition. We decided to use a sliding window strategy.",
    session_id="session_001",
)
print(f"Ingested {len(ids)} memories: {ids}")

# Retrieve relevant memories
results = engine.retrieve("authentication token issues", top_k=5)
for mem in results:
    print(f"  [{mem.lod}] {mem.display_text[:80]}...")

# Expand a memory to full detail
if results:
    full = engine.expand(results[0].id)
    if full:
        print(f"\nFull content:\n{full.display_text}")

# Check stats
print(f"\nStats: {engine.stats()}")

engine.close()
