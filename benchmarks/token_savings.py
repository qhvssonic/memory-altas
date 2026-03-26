"""Benchmark: Token savings — prove that LOD reduces token consumption.

Compares token usage between:
- Baseline: always inject full L2 content
- MemoryAtlas: dynamic LOD (L0/L1/L2 mix)
"""

from __future__ import annotations

import random
from memory_atlas.storage.cache import CachedMemory
from memory_atlas.scene.lod import LODManager


def run_benchmark(
    num_memories: int = 50,
    max_tokens: int = 2000,
) -> dict:
    """Compare token usage: full L2 vs dynamic LOD.

    Args:
        num_memories: Number of memories to include in the view.
        max_tokens: Token budget for memory context.
    """
    random.seed(42)

    # Generate memories with realistic token sizes
    memories: list[CachedMemory] = []
    scores: dict[str, float] = {}
    for i in range(num_memories):
        mid = f"m{i:03d}"
        importance = random.random()
        # L0: ~20 tokens, L1: ~80 tokens, L2: ~500 tokens
        label = f"{'x' * 70} tag-{mid}"  # ~20 tokens
        summary = f"{'y' * 280} summary-{mid}"  # ~80 tokens
        content = f"{'z' * 1800} content-{mid}"  # ~500 tokens
        mem = CachedMemory(
            id=mid,
            label=label,
            summary=summary,
            content=content,
            importance=importance,
            lod="L2",  # baseline: everything at L2
        )
        memories.append(mem)
        scores[mid] = importance

    # Baseline: all L2
    baseline_tokens = sum(m.token_estimate() for m in memories)

    # MemoryAtlas: dynamic LOD
    lod_manager = LODManager(file_store=None, max_tokens=max_tokens)
    optimized = lod_manager.assign_lod(
        [CachedMemory(
            id=m.id, label=m.label, summary=m.summary,
            content=m.content, importance=m.importance,
        ) for m in memories],
        scores,
    )
    optimized_tokens = sum(m.token_estimate() for m in optimized)

    savings_rate = 1 - (optimized_tokens / baseline_tokens) if baseline_tokens > 0 else 0

    lod_dist = {"L0": 0, "L1": 0, "L2": 0}
    for m in optimized:
        lod_dist[m.lod] += 1

    return {
        "num_memories": num_memories,
        "baseline_tokens_l2_all": baseline_tokens,
        "optimized_tokens_lod": optimized_tokens,
        "token_savings_rate": round(savings_rate, 4),
        "lod_distribution": lod_dist,
        "max_token_budget": max_tokens,
        "target_met": savings_rate > 0.40,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print("=== Token Savings Benchmark ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    target = "✅ PASS" if result["target_met"] else "❌ FAIL"
    print(f"\n  Target (>40%): {target} ({result['token_savings_rate']:.1%})")
