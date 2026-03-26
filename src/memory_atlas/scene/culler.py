"""Frustum Culler: detect topic shifts and actively unload irrelevant memories."""

from __future__ import annotations

from memory_atlas.storage.cache import CachedMemory, CacheManager


class FrustumCuller:
    """Detect topic transitions and demote irrelevant memories from hot cache.

    Like a game engine's frustum culler removes off-screen objects,
    this removes memories that are no longer relevant to the current conversation.

    Detection signals:
    - Explicit: user says "switch topic", "let's move on", etc.
    - Implicit: entity overlap between current turn and hot memories drops below threshold
    - Drift: consecutive turns without referencing hot memory entities
    """

    # Explicit topic-switch phrases (multi-language)
    SWITCH_SIGNALS = [
        "换个话题", "先不管", "先放一放", "我们来看看", "另外一个",
        "switch topic", "let's move on", "moving on", "change subject",
        "different topic", "forget about", "never mind", "anyway",
        "by the way", "on another note",
    ]

    def __init__(
        self,
        cache: CacheManager,
        overlap_threshold: float = 0.3,
        idle_rounds_limit: int = 3,
    ):
        self.cache = cache
        self.overlap_threshold = overlap_threshold
        self.idle_rounds_limit = idle_rounds_limit
        self._idle_counters: dict[str, int] = {}  # memory_id → rounds without reference

    def cull(
        self,
        current_message: str,
        current_entities: list[str],
    ) -> list[str]:
        """Evaluate hot memories and demote irrelevant ones. Returns demoted IDs."""
        demoted: list[str] = []

        # Check for explicit topic switch
        explicit_switch = self._detect_explicit_switch(current_message)

        hot_memories = self.cache.get_hot()
        if not hot_memories:
            return demoted

        current_entity_set = set(e.lower() for e in current_entities)

        for mem in hot_memories:
            mem_entity_set = set(e.lower() for e in mem.entities)

            # Calculate entity overlap
            if mem_entity_set and current_entity_set:
                overlap = len(mem_entity_set & current_entity_set) / len(mem_entity_set)
            else:
                overlap = 0.0

            should_demote = False

            # Signal 1: Explicit switch — demote all low-overlap memories
            if explicit_switch and overlap < 0.5:
                should_demote = True

            # Signal 2: Entity drift — overlap below threshold
            if overlap < self.overlap_threshold:
                self._idle_counters[mem.id] = self._idle_counters.get(mem.id, 0) + 1
                if self._idle_counters[mem.id] >= self.idle_rounds_limit:
                    should_demote = True
            else:
                # Reset idle counter if still relevant
                self._idle_counters.pop(mem.id, None)

            if should_demote:
                self.cache.demote_to_warm(mem.id)
                self._idle_counters.pop(mem.id, None)
                demoted.append(mem.id)

        return demoted

    def _detect_explicit_switch(self, message: str) -> bool:
        """Check if the message contains explicit topic-switch signals."""
        msg_lower = message.lower()
        return any(signal in msg_lower for signal in self.SWITCH_SIGNALS)

    def reset(self) -> None:
        """Reset idle counters (e.g., on new session)."""
        self._idle_counters.clear()
