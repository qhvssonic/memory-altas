"""Benchmark: Latency comparison — cache hit vs cold storage retrieval.

Measures the time difference between:
- Hot/warm cache lookup (in-memory dict/LRU)
- Cold storage lookup (DuckDB vector search)
"""

from __future__ import annotations

import time
import uuid
import random
from pathlib import Path
import tempfile

from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier
from memory_atlas.core.registry import Registry, MemoryRecord


def run_benchmark(
    num_memories: int = 500,
    num_queries: int = 200,
    embedding_dim: int = 64,
) -> dict:
    """Compare latency: cache hit vs DuckDB cold search.

    Args:
        num_memories: Number of memories to seed in both cache and DuckDB.
        num_queries: Number of queries to run for each path.
        embedding_dim: Dimension of fake embeddings.
    """
    random.seed(42)

    # Setup cache
    cache = CacheManager(hot_capacity=100, warm_capacity=300)

    # Setup DuckDB
    tmp_dir = tempfile.mkdtemp()
    registry = Registry(Path(tmp_dir) / "bench.duckdb")

    # Seed data
    memory_ids: list[str] = []
    for i in range(num_memories):
        mid = uuid.uuid4().hex[:10]
        emb = [random.gauss(0, 1) for _ in range(embedding_dim)]

        # Insert into DuckDB
        registry.insert_memory(MemoryRecord(
            id=mid, label=f"memory-{i}", summary=f"summary-{i}",
            embedding=emb, importance_score=random.random(),
        ))

        # Also put half in cache (hot/warm)
        if i < num_memories // 2:
            mem = CachedMemory(
                id=mid, label=f"memory-{i}", summary=f"summary-{i}",
                embedding=emb, importance=random.random(),
                tier=CacheTier.HOT if i < 100 else CacheTier.WARM,
            )
            if i < 100:
                cache.promote_to_hot(mem)
            else:
                cache.promote_to_warm(mem)

        memory_ids.append(mid)

    # Benchmark: cache hits
    cached_ids = memory_ids[:num_memories // 2]
    cache_times: list[float] = []
    for _ in range(num_queries):
        mid = random.choice(cached_ids)
        t0 = time.perf_counter_ns()
        cache.get(mid)
        t1 = time.perf_counter_ns()
        cache_times.append((t1 - t0) / 1_000)  # microseconds

    # Benchmark: cold DuckDB vector search
    cold_times: list[float] = []
    for _ in range(num_queries):
        query_emb = [random.gauss(0, 1) for _ in range(embedding_dim)]
        t0 = time.perf_counter_ns()
        registry.vector_search(query_emb, top_k=10)
        t1 = time.perf_counter_ns()
        cold_times.append((t1 - t0) / 1_000)  # microseconds

    registry.close()

    avg_cache_us = sum(cache_times) / len(cache_times)
    avg_cold_us = sum(cold_times) / len(cold_times)
    p99_cache_us = sorted(cache_times)[int(len(cache_times) * 0.99)]
    p99_cold_us = sorted(cold_times)[int(len(cold_times) * 0.99)]
    speedup = avg_cold_us / avg_cache_us if avg_cache_us > 0 else 0

    return {
        "num_memories": num_memories,
        "num_queries": num_queries,
        "avg_cache_latency_us": round(avg_cache_us, 2),
        "avg_cold_latency_us": round(avg_cold_us, 2),
        "p99_cache_latency_us": round(p99_cache_us, 2),
        "p99_cold_latency_us": round(p99_cold_us, 2),
        "speedup_factor": round(speedup, 1),
        "cache_under_10ms": avg_cache_us < 10_000,
        "cold_under_200ms": avg_cold_us < 200_000,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print("=== Latency Comparison Benchmark ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    cache_ok = "✅" if result["cache_under_10ms"] else "❌"
    cold_ok = "✅" if result["cold_under_200ms"] else "❌"
    print(f"\n  Cache <10ms: {cache_ok} ({result['avg_cache_latency_us']:.0f}µs)")
    print(f"  Cold <200ms: {cold_ok} ({result['avg_cold_latency_us']:.0f}µs)")
    print(f"  Speedup: {result['speedup_factor']}x")
