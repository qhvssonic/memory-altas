"""Memory Cluster: group related memories like game engine Asset Bundles."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from memory_atlas.core.registry import Registry


@dataclass
class MemoryCluster:
    """A group of related memories bundled together.

    Like a game engine's Asset Bundle, a cluster packages related memories
    so they can be loaded/unloaded as a unit.
    """

    id: str
    name: str
    summary: str = ""
    memory_ids: list[str] = field(default_factory=list)
    entity_tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class ClusterManager:
    """Manage memory clusters: create, auto-group, and retrieve."""

    def __init__(self, registry: Registry):
        self.registry = registry
        self._ensure_table()

    def _ensure_table(self) -> None:
        self.registry.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_clusters (
                id VARCHAR PRIMARY KEY,
                name VARCHAR,
                summary TEXT,
                memory_ids JSON,
                entity_tags JSON,
                importance FLOAT DEFAULT 0.5,
                metadata JSON
            )
        """)

    def create_cluster(
        self, name: str, memory_ids: list[str], summary: str = "",
        entity_tags: list[str] | None = None,
    ) -> MemoryCluster:
        """Manually create a cluster from a list of memory IDs."""
        cluster = MemoryCluster(
            id=uuid.uuid4().hex[:12],
            name=name,
            summary=summary,
            memory_ids=memory_ids,
            entity_tags=entity_tags or [],
        )
        self._save(cluster)
        return cluster

    def auto_cluster_by_entity(self, entity_name: str) -> MemoryCluster | None:
        """Auto-create a cluster from all memories linked to an entity."""
        memories = self.registry.get_memories_for_entity(entity_name)
        if not memories:
            return None
        ids = [m.id for m in memories]
        return self.create_cluster(
            name=f"cluster:{entity_name}",
            memory_ids=ids,
            summary=f"Auto-clustered {len(ids)} memories about '{entity_name}'",
            entity_tags=[entity_name],
        )

    def get_cluster(self, cluster_id: str) -> MemoryCluster | None:
        row = self.registry.conn.execute(
            "SELECT * FROM memory_clusters WHERE id = ?", [cluster_id]
        ).fetchone()
        if not row:
            return None
        return self._row_to_cluster(row)

    def list_clusters(self) -> list[MemoryCluster]:
        rows = self.registry.conn.execute(
            "SELECT * FROM memory_clusters ORDER BY importance DESC"
        ).fetchall()
        return [self._row_to_cluster(r) for r in rows]

    def delete_cluster(self, cluster_id: str) -> bool:
        self.registry.conn.execute(
            "DELETE FROM memory_clusters WHERE id = ?", [cluster_id]
        )
        return True

    def _save(self, cluster: MemoryCluster) -> None:
        import json
        self.registry.conn.execute(
            """INSERT OR REPLACE INTO memory_clusters
               (id, name, summary, memory_ids, entity_tags, importance, metadata)
               VALUES (?,?,?,?,?,?,?)""",
            [
                cluster.id, cluster.name, cluster.summary,
                json.dumps(cluster.memory_ids),
                json.dumps(cluster.entity_tags),
                cluster.importance,
                json.dumps(cluster.metadata),
            ],
        )

    @staticmethod
    def _row_to_cluster(row: tuple) -> MemoryCluster:
        import json
        return MemoryCluster(
            id=row[0],
            name=row[1] or "",
            summary=row[2] or "",
            memory_ids=json.loads(row[3]) if row[3] else [],
            entity_tags=json.loads(row[4]) if row[4] else [],
            importance=float(row[5]) if row[5] is not None else 0.5,
            metadata=json.loads(row[6]) if row[6] else {},
        )
