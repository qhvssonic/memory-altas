"""MemoryEngine: the framework-agnostic core engine."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_atlas.config import MemoryAtlasConfig
from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.core.tree_index import TreeIndex
from memory_atlas.llm.provider import LLMProvider
from memory_atlas.llm.embedder import create_embedder, Embedder
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.storage.cache import CachedMemory
from memory_atlas.ingestion.chunker import Chunker
from memory_atlas.ingestion.extractor import Extractor
from memory_atlas.ingestion.summarizer import Summarizer
from memory_atlas.scene.manager import SceneManager
from memory_atlas.maintenance.forgetting import ForgettingManager, ForgetResult
from memory_atlas.io.exporter import Exporter
from memory_atlas.io.importer import Importer
from memory_atlas.core.cluster import ClusterManager


class MemoryEngine:
    """Framework-agnostic memory engine. The SceneManager is the brain."""

    def __init__(self, storage_path: str = "./memory_atlas_data", **kwargs):
        self.config = MemoryAtlasConfig(storage_path=storage_path, **kwargs)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save()

        # Core components
        self.registry = Registry(self.storage_path / "index.duckdb")
        self.tree = TreeIndex(self.storage_path)
        self.file_store = FileStore(self.storage_path)

        # LLM + Embedding
        self.llm = LLMProvider(
            model=self.config.llm_model, api_key=self.config.llm_api_key
        )
        self.embedder: Embedder = create_embedder(
            mode=self.config.embedding_model,
            dim=self.config.embedding_dim,
        )

        # Ingestion pipeline
        self.chunker = Chunker()
        self.extractor = Extractor(self.llm)
        self.summarizer = Summarizer(self.llm)

        # Scene Manager (the differentiator)
        self.scene = SceneManager(
            config=self.config,
            registry=self.registry,
            tree=self.tree,
            file_store=self.file_store,
            embedder=self.embedder,
            llm=self.llm,
        )

        # Cluster Manager (auto-bundling)
        self.cluster_mgr = ClusterManager(self.registry)

    # --- Ingestion ---

    def ingest(
        self,
        content: str,
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Full ingestion pipeline: chunk → extract → summarize → embed → store."""
        chunks = self.chunker.chunk(content)
        memory_ids: list[str] = []

        for chunk in chunks:
            mid = uuid.uuid4().hex[:12]
            now = datetime.now(timezone.utc).isoformat()

            # Extract entities and facts
            extraction = self.extractor.extract(chunk.text)

            # Generate summaries
            entity_names = [e["name"] for e in extraction.entities]
            summary_result = self.summarizer.summarize(chunk.text, entity_names)

            # Generate embedding from L1 summary
            embedding = self.embedder.embed([summary_result.summary])[0]

            # Store L2 content as Markdown
            mem_chunk = MemoryChunk(
                id=mid,
                session_id=session_id,
                created_at=now,
                entities=entity_names,
                importance=extraction.importance,
                title=summary_result.label,
                content=chunk.text,
            )
            file_path = self.file_store.save_chunk(mem_chunk)

            # Store metadata in DuckDB
            record = MemoryRecord(
                id=mid,
                session_id=session_id,
                label=summary_result.label,
                summary=summary_result.summary,
                file_path=str(file_path),
                embedding=embedding,
                importance_score=extraction.importance,
            )
            self.registry.insert_memory(record)

            # Store entities and link them
            for ent in extraction.entities:
                eid = self.registry.upsert_entity(ent["name"], ent.get("type", "concept"))
                self.registry.link_memory_entity(mid, eid)

            # Auto-cluster: if an entity now has enough memories, bundle them
            self.cluster_mgr.auto_update_for_memory(
                mid, entity_names, threshold=self.config.auto_cluster_threshold
            )

            # Add to tree index
            self._index_in_tree(mid, extraction.topics, summary_result.label)

            memory_ids.append(mid)

        return memory_ids

    def maybe_ingest(
        self,
        content: str,
        session_id: str = "",
    ) -> list[str] | None:
        """Conditionally ingest based on strategy (rule_based or llm)."""
        if self.config.ingest_strategy == "rule_based":
            if len(content.strip()) < self.config.ingest_min_length:
                return None
            return self.ingest(content, session_id)
        else:
            # LLM evaluation
            try:
                result = self.llm.complete_json(
                    f"Should this content be saved as a long-term memory? "
                    f"Content: {content[:500]}\n"
                    f'Return {{"should_save": true/false, "reason": "..."}}',
                    system="You decide if content is worth remembering long-term.",
                )
                if result.get("should_save"):
                    return self.ingest(content, session_id)
            except Exception:
                # Fallback to rule-based
                if len(content.strip()) >= self.config.ingest_min_length:
                    return self.ingest(content, session_id)
        return None

    def bulk_ingest(
        self, conversations: list[str], session_id: str = ""
    ) -> list[str]:
        """Ingest multiple conversations/documents."""
        all_ids: list[str] = []
        for conv in conversations:
            ids = self.ingest(conv, session_id)
            all_ids.extend(ids)
        return all_ids

    # --- Retrieval ---

    def retrieve(self, query: str, top_k: int = 10) -> list[CachedMemory]:
        """Retrieve relevant memories using the scene manager."""
        return self.scene.get_memory_view(query)[:top_k]

    def expand(self, memory_id: str) -> CachedMemory | None:
        """Expand a memory to L2 (full content)."""
        return self.scene.expand_memory(memory_id)

    def format_memories(self, memories: list[CachedMemory]) -> str:
        """Format memories into a context string."""
        return self.scene.format_context(memories)

    # --- Stats ---

    def stats(self) -> dict:
        """Return engine statistics."""
        return self.scene.stats()

    def forget(self, limit: int = 500) -> ForgetResult:
        """Run a forgetting cycle: compress/archive low-activity memories."""
        fm = ForgettingManager(
            self.registry, self.file_store,
            decay_lambda=self.config.decay_lambda,
        )
        return fm.run_cycle(limit=limit)

    # --- Import / Export ---

    def export_memories(
        self, output_path: str, session_id: str | None = None, include_l2: bool = True
    ) -> dict:
        """Export memories to a JSON file."""
        exporter = Exporter(self.registry, self.file_store)
        if session_id:
            return exporter.export_session(session_id, output_path, include_l2)
        return exporter.export_all(output_path, include_l2)

    def import_memories(
        self, input_path: str, mode: str = "merge"
    ) -> dict:
        """Import memories from a JSON file. mode: 'merge' or 'overwrite'."""
        importer = Importer(self.registry, self.file_store)
        return importer.import_file(input_path, mode=mode)

    def close(self) -> None:
        """Clean up resources."""
        self.scene.persist()
        self.registry.close()

    # --- Internal ---

    def _index_in_tree(
        self, memory_id: str, topics: list[str], label: str
    ) -> None:
        """Place a new memory in the tree index."""
        if not topics:
            self.tree.add_memory_to_node("root", memory_id)
            return

        # Find or create topic node
        primary_topic = topics[0]
        node = self.tree.find_node(primary_topic)
        if not node:
            from memory_atlas.core.tree_index import TreeNode
            new_node = TreeNode(
                id=primary_topic,
                label=primary_topic,
                summary=label,
                node_type="topic",
            )
            self.tree.add_child("root", new_node)
        self.tree.add_memory_to_node(primary_topic, memory_id)
