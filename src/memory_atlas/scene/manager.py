"""SceneManager: the core orchestrator — game-engine style memory view management."""

from __future__ import annotations

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.llm.provider import LLMProvider
from memory_atlas.llm.embedder import Embedder
from memory_atlas.storage.cache import CachedMemory, CacheManager, CacheTier
from memory_atlas.storage.file_store import FileStore
from memory_atlas.retrieval.vector_search import VectorSearch, SearchResult
from memory_atlas.retrieval.tree_search import TreeSearch
from memory_atlas.retrieval.fusion import FusionRanker
from memory_atlas.scene.prefetcher import Prefetcher
from memory_atlas.scene.culler import FrustumCuller
from memory_atlas.scene.lod import LODManager


class SceneManager:
    """The Scene Manager — orchestrates prefetching, culling, LOD, and caching.

    This is the core differentiator of MemoryAtlas. Like a game engine's scene
    manager, it proactively manages what's in the agent's "view" at any moment.
    """

    def __init__(
        self,
        config: MemoryAtlasConfig,
        registry: Registry,
        tree: TreeIndex,
        file_store: FileStore,
        embedder: Embedder,
        llm: LLMProvider,
    ):
        self.config = config
        self.registry = registry
        self.tree = tree
        self.file_store = file_store
        self.embedder = embedder
        self.llm = llm

        # Cache
        self.cache = CacheManager(
            hot_capacity=config.hot_capacity,
            warm_capacity=config.warm_capacity,
        )

        # Sub-systems
        self.vector_search = VectorSearch(registry, embedder)
        self.tree_search = TreeSearch(registry, tree, llm)
        self.fusion = FusionRanker(config.vector_weight, config.tree_weight)
        self.prefetcher = Prefetcher(registry, self.cache, embedder, llm, config.prefetch_top_k)
        self.culler = FrustumCuller(self.cache, config.culling_overlap_threshold)
        self.lod = LODManager(file_store, config.max_memory_tokens)

        # State
        self._current_entities: list[str] = []
        self._session_id: str = ""
        self._turn_count: int = 0

    def initialize_session(self, session_id: str = "") -> None:
        """Called at before_agent: set up session state."""
        self._session_id = session_id
        self._turn_count = 0
        self.culler.reset()

    def get_memory_view(self, query: str) -> list[CachedMemory]:
        """Called at before_model: retrieve the optimal memory view for the current query.

        Priority: hot cache → warm cache → cold storage (vector + tree search).
        """
        self._turn_count += 1
        results: list[CachedMemory] = []
        scores: dict[str, float] = {}

        # 1. Check hot cache — already in context
        hot = self.cache.get_hot()
        for mem in hot:
            results.append(mem)
            scores[mem.id] = mem.importance + 0.3  # bonus for being hot

        # 2. Check warm cache — preloaded candidates
        warm = self.cache.get_warm()
        query_emb = self.embedder.embed([query])[0]
        for mem in warm:
            if mem.embedding:
                sim = self._cosine_sim(query_emb, mem.embedding)
                if sim > 0.3:
                    results.append(mem)
                    scores[mem.id] = sim

        # 3. Cold search if we need more
        seen_ids = {m.id for m in results}
        if len(results) < self.config.retrieval_top_k:
            cold_results = self._search_cold(query)
            for sr in cold_results:
                if sr.record.id not in seen_ids:
                    cached = self._record_to_cached(sr.record)
                    results.append(cached)
                    scores[cached.id] = sr.score
                    seen_ids.add(cached.id)

        # 4. Promote top results to hot
        sorted_results = sorted(results, key=lambda m: scores.get(m.id, 0), reverse=True)
        top_results = sorted_results[:self.config.hot_capacity]
        for mem in top_results:
            if mem.tier != CacheTier.HOT:
                self.cache.promote_to_hot(mem)
                self.registry.touch_memory(mem.id)

        # 5. Apply LOD
        top_results = self.lod.assign_lod(top_results, scores)

        return top_results

    def update(
        self,
        recent_message: str,
        current_entities: list[str] | None = None,
    ) -> dict:
        """Called at after_model: run scene management cycle.

        Returns stats about what happened.
        """
        entities = current_entities or []
        self._current_entities = entities
        stats = {"prefetched": 0, "culled": 0}

        # Frustum culling — demote irrelevant memories
        if self.config.culling_enabled:
            demoted = self.culler.cull(recent_message, entities)
            stats["culled"] = len(demoted)

        # Predictive prefetching — preload next-turn candidates
        if self.config.prefetch_enabled:
            prefetched = self.prefetcher.prefetch(recent_message, entities)
            stats["prefetched"] = len(prefetched)

        # Record topic transitions
        if entities and self._current_entities:
            for e in entities:
                for prev in self._current_entities:
                    if e != prev:
                        self.registry.record_transition(prev, e, self._session_id)

        return stats

    def persist(self) -> None:
        """Called at after_agent: save state."""
        self.tree.save()

    def learn_transition_patterns(self) -> None:
        """Analyze topic transitions to improve future prefetching."""
        # This is recorded incrementally in update(), no batch needed for v0.1
        pass

    def format_context(self, memories: list[CachedMemory]) -> str:
        """Format memories into a context string for injection into messages."""
        return self.lod.format_memory_view(memories)

    def expand_memory(self, memory_id: str) -> CachedMemory | None:
        """Expand a specific memory to L2 (full content)."""
        mem = self.cache.get(memory_id)
        if mem:
            return self.lod.expand_to_l2(mem)
        return None

    def stats(self) -> dict:
        """Return current scene manager statistics."""
        cache_stats = self.cache.stats()
        total = self.registry.count_memories()
        return {
            "total": total,
            "hot": cache_stats["hot"],
            "warm": cache_stats["warm"],
            "cold": total - cache_stats["hot"] - cache_stats["warm"],
            "turn_count": self._turn_count,
        }

    # --- Internal ---

    def _search_cold(self, query: str) -> list[SearchResult]:
        """Dual-path search on cold storage."""
        vector_results = self.vector_search.search(query, top_k=self.config.retrieval_top_k)
        tree_results = self.tree_search.search(query, top_k=self.config.retrieval_top_k)
        return self.fusion.fuse(vector_results, tree_results, top_k=self.config.retrieval_top_k)

    def _record_to_cached(self, rec) -> CachedMemory:
        """Convert a MemoryRecord to a CachedMemory."""
        entities = [e[1] for e in self.registry.get_entities_for_memory(rec.id)]
        return CachedMemory(
            id=rec.id,
            label=rec.label,
            summary=rec.summary,
            embedding=rec.embedding,
            importance=rec.importance_score,
            access_count=rec.access_count,
            tier=CacheTier.COLD,
            entities=entities,
        )

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
