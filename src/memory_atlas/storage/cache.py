"""Three-tier cache: Hot / Warm / Cold."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class CachedMemory:
    """A memory entry in the cache with tier and LOD metadata."""

    id: str
    label: str = ""          # L0
    summary: str = ""        # L1
    content: str | None = None  # L2 (loaded on demand)
    embedding: list[float] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    tier: CacheTier = CacheTier.COLD
    lod: str = "L1"          # current LOD level
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_text(self) -> str:
        """Return text at current LOD level."""
        if self.lod == "L0":
            return self.label
        elif self.lod == "L1":
            return self.summary or self.label
        else:  # L2
            return self.content or self.summary or self.label

    def token_estimate(self) -> int:
        """Rough token estimate for current LOD."""
        text = self.display_text
        return max(1, len(text) // 4)


class CacheManager:
    """Three-tier cache manager: Hot (active), Warm (LRU), Cold (disk)."""

    def __init__(self, hot_capacity: int = 20, warm_capacity: int = 100):
        self.hot_capacity = hot_capacity
        self.warm_capacity = warm_capacity
        self._hot: dict[str, CachedMemory] = {}
        self._warm: OrderedDict[str, CachedMemory] = OrderedDict()

    # --- Tier access ---

    def get(self, memory_id: str) -> CachedMemory | None:
        """Look up a memory in hot then warm tier. Returns None if only in cold."""
        if memory_id in self._hot:
            mem = self._hot[memory_id]
            mem.access_count += 1
            return mem
        if memory_id in self._warm:
            self._warm.move_to_end(memory_id)
            mem = self._warm[memory_id]
            mem.access_count += 1
            return mem
        return None

    def get_hot(self) -> list[CachedMemory]:
        """Return all memories in the hot tier."""
        return list(self._hot.values())

    def get_warm(self) -> list[CachedMemory]:
        """Return all memories in the warm tier."""
        return list(self._warm.values())

    # --- Tier promotion / demotion ---

    def promote_to_hot(self, memory: CachedMemory) -> list[str]:
        """Move a memory to the hot tier. Returns IDs evicted to warm."""
        evicted: list[str] = []
        # Remove from warm if present
        self._warm.pop(memory.id, None)
        memory.tier = CacheTier.HOT
        self._hot[memory.id] = memory

        # Evict lowest-importance if over capacity
        while len(self._hot) > self.hot_capacity:
            victim_id = min(self._hot, key=lambda k: self._hot[k].importance)
            victim = self._hot.pop(victim_id)
            self._put_warm(victim)
            evicted.append(victim_id)
        return evicted

    def promote_to_warm(self, memory: CachedMemory) -> list[str]:
        """Move a memory to the warm tier. Returns IDs evicted to cold."""
        memory.tier = CacheTier.WARM
        evicted = self._put_warm(memory)
        return evicted

    def demote_to_warm(self, memory_id: str) -> bool:
        """Demote a memory from hot to warm."""
        mem = self._hot.pop(memory_id, None)
        if mem is None:
            return False
        mem.tier = CacheTier.WARM
        self._put_warm(mem)
        return True

    def demote_to_cold(self, memory_id: str) -> CachedMemory | None:
        """Remove a memory from hot/warm entirely (back to cold)."""
        mem = self._hot.pop(memory_id, None) or self._warm.pop(memory_id, None)
        if mem:
            mem.tier = CacheTier.COLD
        return mem

    def _put_warm(self, memory: CachedMemory) -> list[str]:
        """Insert into warm tier, evicting LRU if needed."""
        evicted: list[str] = []
        memory.tier = CacheTier.WARM
        self._warm[memory.id] = memory
        self._warm.move_to_end(memory.id)
        while len(self._warm) > self.warm_capacity:
            old_id, old_mem = self._warm.popitem(last=False)
            old_mem.tier = CacheTier.COLD
            evicted.append(old_id)
        return evicted

    # --- Stats ---

    def stats(self) -> dict[str, int]:
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "hot_capacity": self.hot_capacity,
            "warm_capacity": self.warm_capacity,
        }

    def clear(self) -> None:
        self._hot.clear()
        self._warm.clear()
