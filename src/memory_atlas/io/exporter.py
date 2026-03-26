"""Export memories to portable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from memory_atlas.core.registry import Registry
from memory_atlas.storage.file_store import FileStore


class Exporter:
    """Export memories to a portable JSON file."""

    def __init__(self, registry: Registry, file_store: FileStore):
        self.registry = registry
        self.file_store = file_store

    def export_all(self, output_path: str | Path, include_l2: bool = True) -> dict:
        """Export all memories to a JSON file.

        Args:
            output_path: Path to write the JSON file.
            include_l2: Whether to include full L2 content from Markdown files.

        Returns:
            Stats dict with count of exported items.
        """
        return self._export(output_path, session_id=None, include_l2=include_l2)

    def export_session(
        self, session_id: str, output_path: str | Path, include_l2: bool = True
    ) -> dict:
        """Export memories for a specific session."""
        return self._export(output_path, session_id=session_id, include_l2=include_l2)

    def _export(
        self, output_path: str | Path, session_id: str | None, include_l2: bool
    ) -> dict:
        memories = self.registry.list_memories(session_id=session_id, limit=100000)
        export_data: dict[str, Any] = {
            "version": "0.2.0",
            "format": "memory_atlas_export",
            "memories": [],
            "entities": [],
            "tree_nodes": [],
        }

        for rec in memories:
            entry: dict[str, Any] = {
                "id": rec.id,
                "session_id": rec.session_id,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "label": rec.label,
                "summary": rec.summary,
                "importance_score": rec.importance_score,
                "access_count": rec.access_count,
                "last_accessed_at": rec.last_accessed_at,
                "parent_node": rec.parent_node,
                "cache_tier": rec.cache_tier,
                "embedding": rec.embedding,
                "entities": [
                    {"id": e[0], "name": e[1], "type": e[2]}
                    for e in self.registry.get_entities_for_memory(rec.id)
                ],
            }
            if include_l2:
                chunk = self.file_store.load_chunk(rec.id)
                entry["content"] = chunk.content if chunk else None
            export_data["memories"].append(entry)

        # Export tree structure
        root = self.registry.get_tree_root()
        if root:
            children = self.registry.get_tree_children(root["id"])
            export_data["tree_nodes"] = [root] + children

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "memories_exported": len(export_data["memories"]),
            "path": str(output),
        }
