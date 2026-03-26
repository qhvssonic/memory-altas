"""Import memories from portable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.storage.file_store import FileStore, MemoryChunk


class Importer:
    """Import memories from a JSON export file."""

    def __init__(self, registry: Registry, file_store: FileStore):
        self.registry = registry
        self.file_store = file_store

    def import_file(
        self,
        input_path: str | Path,
        mode: Literal["merge", "overwrite"] = "merge",
    ) -> dict:
        """Import memories from a JSON file.

        Args:
            input_path: Path to the JSON export file.
            mode:
                "merge" — skip existing IDs, only add new ones.
                "overwrite" — replace existing memories with imported data.

        Returns:
            Stats dict with counts.
        """
        data = json.loads(Path(input_path).read_text(encoding="utf-8"))
        memories = data.get("memories", [])

        stats = {"total": len(memories), "imported": 0, "skipped": 0, "overwritten": 0}

        for entry in memories:
            mid = entry["id"]
            existing = self.registry.get_memory(mid)

            if existing and mode == "merge":
                stats["skipped"] += 1
                continue

            if existing and mode == "overwrite":
                self.registry.delete_memory(mid)
                self.file_store.delete_chunk(mid)
                stats["overwritten"] += 1

            # Insert memory record
            rec = MemoryRecord(
                id=mid,
                session_id=entry.get("session_id", ""),
                created_at=entry.get("created_at", ""),
                updated_at=entry.get("updated_at", ""),
                label=entry.get("label", ""),
                summary=entry.get("summary", ""),
                embedding=entry.get("embedding", []),
                importance_score=entry.get("importance_score", 0.5),
                access_count=entry.get("access_count", 0),
                last_accessed_at=entry.get("last_accessed_at", ""),
                parent_node=entry.get("parent_node", ""),
                cache_tier=entry.get("cache_tier", "cold"),
            )
            self.registry.insert_memory(rec)

            # Save L2 content if present
            content = entry.get("content")
            if content:
                chunk = MemoryChunk(
                    id=mid,
                    session_id=entry.get("session_id", ""),
                    created_at=entry.get("created_at", ""),
                    entities=[e["name"] for e in entry.get("entities", [])],
                    importance=entry.get("importance_score", 0.5),
                    title=entry.get("label", ""),
                    content=content,
                )
                self.file_store.save_chunk(chunk)

            # Restore entities
            for ent in entry.get("entities", []):
                eid = self.registry.upsert_entity(ent["name"], ent.get("type", "concept"))
                self.registry.link_memory_entity(mid, eid)

            stats["imported"] += 1

        return stats
