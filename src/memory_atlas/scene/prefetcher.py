"""Predictive Prefetcher: anticipate next-turn memory needs."""

from __future__ import annotations

from memory_atlas.core.registry import Registry
from memory_atlas.llm.provider import LLMProvider
from memory_atlas.llm.embedder import Embedder
from memory_atlas.storage.cache import CachedMemory, CacheManager, CacheTier


PREDICT_PROMPT = """Based on the current conversation context, predict what topics the user might discuss next.

Current topic/message:
---
{current_context}
---

Recent entities mentioned: {entities}

Return a JSON object:
- "predicted_topics": list of 3-5 topic keywords likely to come up next
- "predicted_entities": list of entity names that might be referenced

Return ONLY valid JSON."""


class Prefetcher:
    """Predict next-turn memory needs and preload to warm cache.

    Strategies:
    - Topic association: historical A→B transitions
    - Entity expansion: related entities from the same domain
    - Temporal locality: recently accessed memories
    """

    def __init__(
        self,
        registry: Registry,
        cache: CacheManager,
        embedder: Embedder,
        llm: LLMProvider,
        top_k: int = 10,
    ):
        self.registry = registry
        self.cache = cache
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k

    def prefetch(
        self,
        current_message: str,
        current_entities: list[str] | None = None,
    ) -> list[str]:
        """Predict and preload memories to warm cache. Returns prefetched IDs."""
        prefetched: list[str] = []

        # Strategy 1: Topic association from historical transitions
        prefetched.extend(self._prefetch_by_transitions(current_entities or []))

        # Strategy 2: Entity expansion — load memories for related entities
        prefetched.extend(self._prefetch_by_entities(current_entities or []))

        # Strategy 3: LLM prediction (if enabled and budget allows)
        if len(prefetched) < self.top_k:
            prefetched.extend(
                self._prefetch_by_prediction(current_message, current_entities or [])
            )

        return list(set(prefetched))[:self.top_k]

    def _prefetch_by_transitions(self, entities: list[str]) -> list[str]:
        """Use historical topic transitions to predict next topics."""
        prefetched: list[str] = []
        for entity in entities[:3]:
            next_topics = self.registry.get_likely_next_topics(entity, top_k=3)
            for topic, _count in next_topics:
                memories = self.registry.get_memories_for_entity(topic)
                for mem in memories[:2]:
                    if not self.cache.get(mem.id):
                        cached = CachedMemory(
                            id=mem.id,
                            label=mem.label,
                            summary=mem.summary,
                            importance=mem.importance_score,
                            entities=[],
                            tier=CacheTier.WARM,
                        )
                        self.cache.promote_to_warm(cached)
                        prefetched.append(mem.id)
        return prefetched

    def _prefetch_by_entities(self, entities: list[str]) -> list[str]:
        """Load memories associated with current entities."""
        prefetched: list[str] = []
        for entity in entities[:5]:
            memories = self.registry.get_memories_for_entity(entity)
            for mem in memories[:3]:
                if not self.cache.get(mem.id):
                    cached = CachedMemory(
                        id=mem.id,
                        label=mem.label,
                        summary=mem.summary,
                        importance=mem.importance_score,
                        tier=CacheTier.WARM,
                    )
                    self.cache.promote_to_warm(cached)
                    prefetched.append(mem.id)
        return prefetched

    def _prefetch_by_prediction(
        self, message: str, entities: list[str]
    ) -> list[str]:
        """Use LLM to predict next topics and preload related memories."""
        prefetched: list[str] = []
        try:
            data = self.llm.complete_json(
                PREDICT_PROMPT.format(
                    current_context=message[:500],
                    entities=", ".join(entities[:10]) or "none",
                ),
                system="Return only valid JSON with predicted_topics and predicted_entities.",
            )
            predicted_topics = data.get("predicted_topics", [])
            predicted_entities = data.get("predicted_entities", [])

            # Search for predicted topics via embedding
            all_predicted = predicted_topics + predicted_entities
            for topic in all_predicted[:5]:
                emb = self.embedder.embed([topic])[0]
                results = self.registry.vector_search(emb, top_k=2)
                for rec, score in results:
                    if score > 0.3 and not self.cache.get(rec.id):
                        cached = CachedMemory(
                            id=rec.id,
                            label=rec.label,
                            summary=rec.summary,
                            importance=rec.importance_score,
                            tier=CacheTier.WARM,
                        )
                        self.cache.promote_to_warm(cached)
                        prefetched.append(rec.id)
        except Exception:
            pass  # Prediction is best-effort
        return prefetched
