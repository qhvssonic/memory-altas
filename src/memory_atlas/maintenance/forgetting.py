"""Forgetting mechanism: decay-based memory archival and compression.

Activity score formula (from DESIGN.md):
  activity = importance × e^(-λ × days_since_access) × log(access_count + 1)

Memories below the threshold are compressed (L2→L1 only) or archived.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass

from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.storage.file_store import FileStore


@dataclass
class ForgetResult:
    """Result of a forgetting cycle."""

    scanned: int = 0
    compressed: int = 0
    archived: int = 0
    kept: int = 0


class ForgettingManager:
    """Decay-based forgetting: compress or archive low-activity memories."""

    def __init__(
        self,
        registry: Registry,
        file_store: FileStore,
        decay_lambda: float = 0.1,
        compress_threshold: float = 0.1,
        archive_threshold: float = 0.02,
    ):
        self.registry = registry
        self.file_store = file_store
        self.decay_lambda = decay_lambda
        self.compress_threshold = compress_threshold
        self.archive_threshold = archive_threshold

    def compute_activity(self, record: MemoryRecord, now: datetime | None = None) -> float:
        """Compute activity score for a memory.

        activity = importance × e^(-λ × days_since_access) × log(access_count + 1)
        """
        now = now or datetime.now(timezone.utc)

        # Parse last_accessed_at
        days_since = 0.0
        if record.last_accessed_at:
            try:
                last = datetime.fromisoformat(record.last_accessed_at.replace("Z", "+00:00"))
                delta = now - last
                days_since = max(0, delta.total_seconds() / 86400)
            except (ValueError, TypeError):
                days_since = 30.0  # assume old if unparseable

        importance = record.importance_score or 0.5
        access_count = record.access_count or 0

        activity = (
            importance
            * math.exp(-self.decay_lambda * days_since)
            * math.log(access_count + 1 + 1)  # +1 to avoid log(1)=0 for count=0
        )
        return round(activity, 6)

    def run_cycle(self, limit: int = 500) -> ForgetResult:
        """Run one forgetting cycle: scan memories and compress/archive low-activity ones."""
        result = ForgetResult()
        memories = self.registry.list_memories(limit=limit)
        now = datetime.now(timezone.utc)

        for rec in memories:
            result.scanned += 1
            activity = self.compute_activity(rec, now)

            if activity < self.archive_threshold:
                # Archive: delete L2 file, keep only L0 label in DB
                self.file_store.delete_chunk(rec.id)
                self.registry.update_memory(
                    rec.id,
                    file_path="",
                    summary="",
                    cache_tier="archived",
                )
                result.archived += 1

            elif activity < self.compress_threshold:
                # Compress: delete L2 file, keep L0+L1 in DB
                self.file_store.delete_chunk(rec.id)
                self.registry.update_memory(
                    rec.id,
                    file_path="",
                    cache_tier="cold",
                )
                result.compressed += 1

            else:
                result.kept += 1

        return result
