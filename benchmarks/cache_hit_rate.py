"""Benchmark: Cache hit rate — prove that hot/warm cache avoids cold storage lookups.

Simulates a multi-turn conversation where topics recur, measuring how often
the cache serves results vs falling back to cold (DuckDB) search.
"""

from __future__ import annotations

import time
import uuid
from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier


def run_benchmark(
    num_memories: int = 200,
    num_queries: int = 100,
    hot_capacity: int = 20,
    warm_capacity: int = 60,
    topic_reuse_rate: float = 0.65,
) -> dict:
    """Simulate queries against a three-tier cache and measure hit rates.

    Args:
        num_memories: Total memories to seed.
        num_queries: Number of simulated retrieval queries.
        hot_capacity: Hot tier capacity.
        warm_capacity: Warm tier capacity.
        topic_reuse_rate: Probability a query hits a previously-seen topic.
    """
    import random
    random.seed(42)

    cache = CacheManager(hot_capacity=hot_capacity, warm_capacity=warm_capacity)

    # Seed memories across topics
    topics = [f"topic_{i}" for i in range(20)]
    all_memories: dict[str, list[CachedMemory]] = {}
    for t in topics:
        all_memories[t] = []
        for _ in range(num_memories // len(topics)):
            mid = uuid.uuid4().hex[:8]
            mem = CachedMemory(
                id=mid,
                label=f"{t} memory {mid}",
                summary=f"Summary about {t}: {mid}",
                importance=random.random(),
                entities=[t],
                tier=CacheTier.COLD,
            )
            all_memories[t].append(mem)

    # Simulate queries
    stats = {"hot_hits": 0, "warm_hits": 0, "cold_hits": 0, "total": 0}
    recent_topics: list[str] = []

    for i in range(num_queries):
        # Decide topic: reuse recent or pick new
        if recent_topics and random.random() < topic_reuse_rate:
            topic = random.choice(recent_topics[-5:])
        else:
            topic = random.choice(topics)

        recent_topics.append(topic)
        stats["total"] += 1

        # Try cache first
        target_mems = all_memories[topic]
        target = random.choice(target_mems)

        cached = cache.get(target.id)
        if cached and cached.tier == CacheTier.HOT:
            stats["hot_hits"] += 1
        elif cached and cached.tier == CacheTier.WARM:
            stats["warm_hits"] += 1
            cache.promote_to_hot(cached)
        else:
            stats["cold_hits"] += 1
            # Simulate cold retrieval → promote to hot
            cache.promote_to_hot(target)
            # Prefetch: warm-load all siblings for this topic (scene manager behavior)
            for sibling in target_mems:
                if not cache.get(sibling.id):
                    cache.promote_to_warm(sibling)

    cache_hit_rate = (stats["hot_hits"] + stats["warm_hits"]) / stats["total"]
    hot_hit_rate = stats["hot_hits"] / stats["total"]
    warm_hit_rate = stats["warm_hits"] / stats["total"]

    return {
        "total_queries": stats["total"],
        "hot_hits": stats["hot_hits"],
        "warm_hits": stats["warm_hits"],
        "cold_hits": stats["cold_hits"],
        "cache_hit_rate": round(cache_hit_rate, 4),
        "hot_hit_rate": round(hot_hit_rate, 4),
        "warm_hit_rate": round(warm_hit_rate, 4),
        "target_met": cache_hit_rate > 0.60,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print("=== Cache Hit Rate Benchmark ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    target = "✅ PASS" if result["target_met"] else "❌ FAIL"
    print(f"\n  Target (>60%): {target} ({result['cache_hit_rate']:.1%})")
