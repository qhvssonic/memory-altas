"""Tests for frustum culler."""

from memory_atlas.storage.cache import CacheManager, CachedMemory, CacheTier
from memory_atlas.scene.culler import FrustumCuller


def _hot_mem(mid: str, entities: list[str]) -> CachedMemory:
    return CachedMemory(id=mid, label=mid, entities=entities, tier=CacheTier.HOT)


class TestFrustumCuller:
    def test_explicit_switch_demotes(self):
        cache = CacheManager(hot_capacity=10, warm_capacity=20)
        mem = _hot_mem("auth1", ["auth", "jwt"])
        cache.promote_to_hot(mem)

        culler = FrustumCuller(cache)
        demoted = culler.cull("换个话题，我们来看看数据库", ["database"])
        assert "auth1" in demoted

    def test_no_switch_keeps(self):
        cache = CacheManager(hot_capacity=10, warm_capacity=20)
        mem = _hot_mem("auth1", ["auth", "jwt"])
        cache.promote_to_hot(mem)

        culler = FrustumCuller(cache)
        demoted = culler.cull("Tell me more about auth tokens", ["auth"])
        assert "auth1" not in demoted

    def test_idle_rounds_demote(self):
        cache = CacheManager(hot_capacity=10, warm_capacity=20)
        mem = _hot_mem("old1", ["legacy"])
        cache.promote_to_hot(mem)

        culler = FrustumCuller(cache, idle_rounds_limit=2)
        # Round 1: no overlap
        culler.cull("talking about new stuff", ["new"])
        assert cache.get("old1").tier == CacheTier.HOT
        # Round 2: still no overlap → demoted
        demoted = culler.cull("still new stuff", ["new"])
        assert "old1" in demoted

    def test_english_switch_signals(self):
        cache = CacheManager(hot_capacity=10, warm_capacity=20)
        cache.promote_to_hot(_hot_mem("x", ["topic_a"]))

        culler = FrustumCuller(cache)
        demoted = culler.cull("let's move on to something else", ["topic_b"])
        assert "x" in demoted

    def test_reset(self):
        cache = CacheManager()
        culler = FrustumCuller(cache)
        culler._idle_counters["a"] = 5
        culler.reset()
        assert len(culler._idle_counters) == 0
