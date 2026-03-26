"""DuckDB-based metadata registry for memories, entities, and links."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryRecord:
    """A row in the memories table."""

    id: str
    session_id: str = ""
    user_id: str = "default"
    agent_id: str = "default"
    created_at: str = ""
    updated_at: str = ""
    label: str = ""
    summary: str = ""
    file_path: str = ""
    embedding: list[float] = field(default_factory=list)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed_at: str = ""
    parent_node: str = ""
    cache_tier: str = "cold"
    metadata: dict[str, Any] = field(default_factory=dict)


class Registry:
    """DuckDB metadata registry — the 'Asset Registry' of MemoryAtlas."""

    DDL = """
    CREATE TABLE IF NOT EXISTS memories (
        id VARCHAR PRIMARY KEY,
        session_id VARCHAR,
        user_id VARCHAR DEFAULT 'default',
        agent_id VARCHAR DEFAULT 'default',
        created_at VARCHAR,
        updated_at VARCHAR,
        label VARCHAR,
        summary TEXT,
        file_path VARCHAR,
        embedding FLOAT[],
        importance_score FLOAT DEFAULT 0.5,
        access_count INTEGER DEFAULT 0,
        last_accessed_at VARCHAR,
        parent_node VARCHAR,
        cache_tier VARCHAR DEFAULT 'cold',
        metadata JSON
    );

    CREATE TABLE IF NOT EXISTS entities (
        id VARCHAR PRIMARY KEY,
        name VARCHAR,
        type VARCHAR,
        first_seen VARCHAR,
        last_seen VARCHAR
    );

    CREATE TABLE IF NOT EXISTS memory_entities (
        memory_id VARCHAR,
        entity_id VARCHAR,
        relation VARCHAR,
        PRIMARY KEY (memory_id, entity_id)
    );

    CREATE TABLE IF NOT EXISTS memory_links (
        source_id VARCHAR,
        target_id VARCHAR,
        link_type VARCHAR,
        PRIMARY KEY (source_id, target_id)
    );

    CREATE TABLE IF NOT EXISTS tree_nodes (
        id VARCHAR PRIMARY KEY,
        parent_id VARCHAR,
        label VARCHAR,
        summary TEXT,
        node_type VARCHAR,
        depth INTEGER,
        children_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS topic_transitions (
        id VARCHAR PRIMARY KEY,
        from_topic VARCHAR,
        to_topic VARCHAR,
        session_id VARCHAR,
        timestamp VARCHAR,
        transition_count INTEGER DEFAULT 1
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        for stmt in self.DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    # --- Memories CRUD ---

    def insert_memory(self, rec: MemoryRecord) -> str:
        now = _now()
        if not rec.id:
            rec.id = _uuid()
        if not rec.created_at:
            rec.created_at = now
        rec.updated_at = now
        rec.last_accessed_at = now

        self.conn.execute(
            """INSERT INTO memories
               (id, session_id, user_id, agent_id, created_at, updated_at, label, summary,
                file_path, embedding, importance_score, access_count,
                last_accessed_at, parent_node, cache_tier, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                rec.id, rec.session_id, rec.user_id, rec.agent_id,
                rec.created_at, rec.updated_at,
                rec.label, rec.summary, rec.file_path,
                rec.embedding, rec.importance_score, rec.access_count,
                rec.last_accessed_at, rec.parent_node, rec.cache_tier,
                str(rec.metadata) if rec.metadata else "{}",
            ],
        )
        return rec.id

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", [memory_id]
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def update_memory(self, memory_id: str, **kwargs: Any) -> bool:
        sets = []
        vals = []
        for k, v in kwargs.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return False
        vals.append(memory_id)
        self.conn.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", vals
        )
        return True

    def delete_memory(self, memory_id: str) -> bool:
        self.conn.execute("DELETE FROM memory_entities WHERE memory_id = ?", [memory_id])
        self.conn.execute(
            "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
            [memory_id, memory_id],
        )
        self.conn.execute("DELETE FROM memories WHERE id = ?", [memory_id])
        return True

    def touch_memory(self, memory_id: str) -> None:
        """Update access count and timestamp."""
        self.conn.execute(
            """UPDATE memories
               SET access_count = access_count + 1, last_accessed_at = ?
               WHERE id = ?""",
            [_now(), memory_id],
        )

    def list_memories(
        self, session_id: str | None = None, user_id: str | None = None,
        agent_id: str | None = None, limit: int = 100,
    ) -> list[MemoryRecord]:
        conditions = []
        params: list = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # --- Vector search ---

    def vector_search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[MemoryRecord, float]]:
        """Cosine similarity search using DuckDB array functions."""
        rows = self.conn.execute(
            """SELECT *,
                      list_cosine_similarity(embedding, ?::FLOAT[]) AS score
               FROM memories
               WHERE embedding IS NOT NULL AND len(embedding) > 0
               ORDER BY score DESC
               LIMIT ?""",
            [query_embedding, top_k],
        ).fetchall()
        results = []
        for row in rows:
            score = row[-1]  # last column is the computed score
            rec = self._row_to_record(row[:-1])
            results.append((rec, float(score) if score else 0.0))
        return results

    # --- Entities ---

    def upsert_entity(
        self, name: str, entity_type: str, entity_id: str | None = None
    ) -> str:
        now = _now()
        existing = self.conn.execute(
            "SELECT id FROM entities WHERE name = ? AND type = ?",
            [name, entity_type],
        ).fetchone()
        if existing:
            eid = existing[0]
            self.conn.execute(
                "UPDATE entities SET last_seen = ? WHERE id = ?", [now, eid]
            )
            return eid
        eid = entity_id or _uuid()
        self.conn.execute(
            "INSERT INTO entities (id, name, type, first_seen, last_seen) VALUES (?,?,?,?,?)",
            [eid, name, entity_type, now, now],
        )
        return eid

    def link_memory_entity(
        self, memory_id: str, entity_id: str, relation: str = "mentions"
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO memory_entities (memory_id, entity_id, relation)
               VALUES (?,?,?)""",
            [memory_id, entity_id, relation],
        )

    def get_entities_for_memory(self, memory_id: str) -> list[tuple[str, str, str]]:
        """Returns list of (entity_id, name, type)."""
        return self.conn.execute(
            """SELECT e.id, e.name, e.type
               FROM memory_entities me JOIN entities e ON me.entity_id = e.id
               WHERE me.memory_id = ?""",
            [memory_id],
        ).fetchall()

    def get_memories_for_entity(self, entity_name: str) -> list[MemoryRecord]:
        rows = self.conn.execute(
            """SELECT m.* FROM memories m
               JOIN memory_entities me ON m.id = me.memory_id
               JOIN entities e ON me.entity_id = e.id
               WHERE e.name = ?
               ORDER BY m.created_at DESC""",
            [entity_name],
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # --- Topic transitions ---

    def record_transition(
        self, from_topic: str, to_topic: str, session_id: str
    ) -> None:
        existing = self.conn.execute(
            "SELECT id, transition_count FROM topic_transitions WHERE from_topic = ? AND to_topic = ?",
            [from_topic, to_topic],
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE topic_transitions SET transition_count = transition_count + 1, timestamp = ? WHERE id = ?",
                [_now(), existing[0]],
            )
        else:
            self.conn.execute(
                "INSERT INTO topic_transitions (id, from_topic, to_topic, session_id, timestamp) VALUES (?,?,?,?,?)",
                [_uuid(), from_topic, to_topic, session_id, _now()],
            )

    def get_likely_next_topics(
        self, current_topic: str, top_k: int = 5
    ) -> list[tuple[str, int]]:
        """Get most likely next topics based on historical transitions."""
        rows = self.conn.execute(
            """SELECT to_topic, transition_count
               FROM topic_transitions
               WHERE from_topic = ?
               ORDER BY transition_count DESC
               LIMIT ?""",
            [current_topic, top_k],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # --- Tree nodes ---

    def insert_tree_node(
        self, node_id: str, parent_id: str, label: str,
        summary: str = "", node_type: str = "topic", depth: int = 0,
    ) -> str:
        self.conn.execute(
            """INSERT INTO tree_nodes (id, parent_id, label, summary, node_type, depth)
               VALUES (?,?,?,?,?,?)""",
            [node_id, parent_id, label, summary, node_type, depth],
        )
        if parent_id:
            self.conn.execute(
                "UPDATE tree_nodes SET children_count = children_count + 1 WHERE id = ?",
                [parent_id],
            )
        return node_id

    def get_tree_children(self, parent_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, label, summary, node_type, depth, children_count FROM tree_nodes WHERE parent_id = ?",
            [parent_id],
        ).fetchall()
        return [
            {"id": r[0], "label": r[1], "summary": r[2],
             "node_type": r[3], "depth": r[4], "children_count": r[5]}
            for r in rows
        ]

    def get_tree_root(self) -> dict | None:
        row = self.conn.execute(
            "SELECT id, label, summary, node_type, depth, children_count FROM tree_nodes WHERE node_type = 'root' LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {"id": row[0], "label": row[1], "summary": row[2],
                "node_type": row[3], "depth": row[4], "children_count": row[5]}

    # --- Stats ---

    def count_memories(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.conn.close()

    # --- Internal ---

    def _row_to_record(self, row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            session_id=row[1] or "",
            user_id=row[2] or "default",
            agent_id=row[3] or "default",
            created_at=row[4] or "",
            updated_at=row[5] or "",
            label=row[6] or "",
            summary=row[7] or "",
            file_path=row[8] or "",
            embedding=list(row[9]) if row[9] else [],
            importance_score=float(row[10]) if row[10] is not None else 0.5,
            access_count=int(row[11]) if row[11] is not None else 0,
            last_accessed_at=row[12] or "",
            parent_node=row[13] or "",
            cache_tier=row[14] or "cold",
            metadata=row[15] if row[15] else {},
        )
