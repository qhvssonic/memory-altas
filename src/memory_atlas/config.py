"""Configuration management for MemoryAtlas."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal


@dataclass
class MemoryAtlasConfig:
    """Core configuration for MemoryAtlas engine."""

    # Storage
    storage_path: str = "./memory_atlas_data"

    # Multi-tenant
    user_id: str = "default"
    agent_id: str = "default"

    # Embedding
    embedding_model: Literal["local", "openai", "cohere"] = "local"
    embedding_dim: int = 384  # all-MiniLM-L6-v2 default
    openai_api_key: str | None = None

    # LLM
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str | None = None

    # Cache tiers
    max_memory_tokens: int = 2000
    hot_capacity: int = 20
    warm_capacity: int = 100
    warm_to_hot_ratio: float = 5.0

    # Scene manager
    prefetch_enabled: bool = True
    prefetch_top_k: int = 10
    culling_enabled: bool = True
    culling_overlap_threshold: float = 0.3
    lod_default: Literal["L0", "L1", "L2"] = "L1"

    # Ingestion
    ingest_strategy: Literal["llm", "rule_based"] = "rule_based"
    ingest_min_length: int = 50
    ingest_require_entities: bool = False

    # Retrieval
    retrieval_top_k: int = 10
    vector_weight: float = 0.6
    tree_weight: float = 0.4

    # Decay
    decay_lambda: float = 0.1  # importance decay rate

    def save(self, path: Path | None = None) -> None:
        """Save config to JSON file."""
        target = path or Path(self.storage_path) / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MemoryAtlasConfig:
        """Load config from JSON file."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
