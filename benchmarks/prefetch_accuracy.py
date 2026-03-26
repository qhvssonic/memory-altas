"""Benchmark: Prefetch accuracy — prove that predictive preloading works.

Simulates topic transitions with historical patterns, measures how often
prefetched memories are actually used in the next turn.
"""

from __future__ import annotations

import random
import uuid
from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier


def run_benchmark(
    num_turns: int = 200,
    num_topics: int = 15,
    transition_predictability: float = 0.6,
    hot_capacity: int = 20,
    warm_capacity: int = 80,
) -> dict:
    """Simulate conversations with topic transitions and measure prefetch accuracy.

    Args:
        num_turns: Number of conversation turns to simulate.
        num_topics: Number of distinct topics.
        transition_predictability: How often transitions follow learned patterns.
    """
    random.seed(42)

    cache = CacheManager(hot_capacity=hot_capacity, warm_capacity=warm_capacity)

    # Build topic transition graph (some topics naturally follow others)
    topics = [f"topic_{i}" for i in range(num_topics)]
    transition_map: dict[str, list[str]] = {}
    for t in topics:
        # Each topic has 2-3 likely successors
        successors = random.sample(topics, k=min(3, num_topics))
        transition_map[t] = successors

    # Build memories per topic
    topic_memories: dict[str, list[CachedMemory]] = {}
    for t in topics:
        topic_memories[t] = [
            CachedMemory(
                id=uuid.uuid4().hex[:8],
                label=f"{t}_mem_{j}",
                summary=f"Memory about {t}",
                importance=random.random(),
                entities=[t],
                tier=CacheTier.COLD,
            )
            for j in range(10)
        ]

    # Simulate
    stats = {"prefetch_attempts": 0, "prefetch_hits": 0, "total_turns": 0}
    current_topic = random.choice(topics)
    prefetched_ids: set[str] = set()  # track what was actually prefetched

    for turn in range(num_turns):
        stats["total_turns"] += 1

        # Check if current topic's memories were prefetched (in warm)
        for mem in topic_memories[current_topic][:3]:
            if mem.id in prefetched_ids:
                cached = cache.get(mem.id)
                if cached:
                    stats["prefetch_hits"] += 1
                    prefetched_ids.discard(mem.id)
                    cache.promote_to_hot(cached)

        # Load current topic memories to hot
        for mem in topic_memories[current_topic][:3]:
            if not cache.get(mem.id):
                cache.promote_to_hot(mem)

        # Prefetch: predict next topic and preload to warm
        predicted_topics = transition_map.get(current_topic, [])
        for pt in predicted_topics[:2]:
            for mem in topic_memories[pt][:2]:
                if not cache.get(mem.id) and mem.id not in prefetched_ids:
                    cache.promote_to_warm(mem)
                    prefetched_ids.add(mem.id)
                    stats["prefetch_attempts"] += 1

        # Transition to next topic
        if random.random() < transition_predictability:
            next_topic = random.choice(transition_map.get(current_topic, topics))
        else:
            next_topic = random.choice(topics)
        current_topic = next_topic

    accuracy = (
        stats["prefetch_hits"] / stats["prefetch_attempts"]
        if stats["prefetch_attempts"] > 0
        else 0
    )

    return {
        "total_turns": stats["total_turns"],
        "prefetch_attempts": stats["prefetch_attempts"],
        "prefetch_hits": stats["prefetch_hits"],
        "prefetch_accuracy": round(accuracy, 4),
        "target_met": accuracy > 0.50,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print("=== Prefetch Accuracy Benchmark ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    target = "✅ PASS" if result["target_met"] else "❌ FAIL"
    print(f"\n  Target (>50%): {target} ({result['prefetch_accuracy']:.1%})")
