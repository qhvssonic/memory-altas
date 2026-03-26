"""Tests for three-tier cache manager."""

import pytest
from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier


def _mem(mid: str, importance: float = 0.5) -> CachedMemory:
    return CachedMemory(id=mid, label=f"label-{mid}", summary=f"summary-{mid}", importance=importance)


class TestCacheManager:
    def test_promote_to_hot(self):
        cache = CacheManager(hot_capacity=3, warm_capacity=5)
        m = _mem("a", 0.8)
        cache.promote_to_hot(m)
        assert cache.get("a") is not None
        assert cache.get("a").tier == CacheTier.HOT

    def test_hot_eviction_to_warm(self):
        cache = CacheManager(hot_capacity=2, warm_capacity=5)
        cache.promote_to_hot(_mem("a", 0.9))
        cache.promote_to_hot(_mem("b", 0.3))
        evicted = cache.promote_to_hot(_mem("c", 0.8))
        # b has lowest importance, should be evicted
        assert "b" in evicted
        assert cache.get("b").tier == CacheTier.WARM

    def test_promote_to_warm(self):
        cache = CacheManager(hot_capacity=3, warm_capacity=2)
        cache.promote_to_warm(_mem("x"))
        assert cache.get("x").tier == CacheTier.WARM

    def test_warm_lru_eviction(self):
        cache = CacheManager(hot_capacity=3, warm_capacity=2)
        cache.promote_to_warm(_mem("a"))
        cache.promote_to_warm(_mem("b"))
        evicted = cache.promote_to_warm(_mem("c"))
        assert "a" in evicted  # oldest evicted
        assert cache.get("a") is None

    def test_demote_hot_to_warm(self):
        cache = CacheManager(hot_capacity=3, warm_capacity=5)
        cache.promote_to_hot(_mem("a"))
        assert cache.demote_to_warm("a")
        assert cache.get("a").tier == CacheTier.WARM

    def test_demote_to_cold(self):
        cache = CacheManager(hot_capacity=3, warm_capacity=5)
        cache.promote_to_hot(_mem("a"))
        mem = cache.demote_to_cold("a")
        assert mem is not None
        assert mem.tier == CacheTier.COLD
        assert cache.get("a") is None

    def test_access_count(self):
        cache = CacheManager()
        cache.promote_to_hot(_mem("a"))
        cache.get("a")
        cache.get("a")
        assert cache.get("a").access_count == 3  # 3rd get

    def test_stats(self):
        cache = CacheManager(hot_capacity=5, warm_capacity=10)
        cache.promote_to_hot(_mem("a"))
        cache.promote_to_warm(_mem("b"))
        s = cache.stats()
        assert s["hot"] == 1
        assert s["warm"] == 1

    def test_clear(self):
        cache = CacheManager()
        cache.promote_to_hot(_mem("a"))
        cache.promote_to_warm(_mem("b"))
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


class TestCachedMemory:
    def test_display_text_l0(self):
        m = CachedMemory(id="x", label="my label", summary="my summary", lod="L0")
        assert m.display_text == "my label"

    def test_display_text_l1(self):
        m = CachedMemory(id="x", label="my label", summary="my summary", lod="L1")
        assert m.display_text == "my summary"

    def test_display_text_l2(self):
        m = CachedMemory(id="x", label="lb", summary="sm", content="full content", lod="L2")
        assert m.display_text == "full content"

    def test_token_estimate(self):
        m = CachedMemory(id="x", summary="a short summary", lod="L1")
        assert m.token_estimate() >= 1
