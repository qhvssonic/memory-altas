"""Markdown file store for L2 memory content (chunks/*.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MemoryChunk:
    """A single memory chunk with frontmatter metadata."""

    id: str
    session_id: str = ""
    created_at: str = ""
    entities: list[str] = field(default_factory=list)
    importance: float = 0.5
    title: str = ""
    content: str = ""

    def to_markdown(self) -> str:
        """Serialize to Markdown with YAML frontmatter."""
        entities_str = ", ".join(self.entities)
        lines = [
            "---",
            f"id: {self.id}",
            f"session_id: {self.session_id}",
            f"created_at: {self.created_at}",
            f"entities: [{entities_str}]",
            f"importance: {self.importance}",
            "---",
            "",
        ]
        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")
        if self.content:
            lines.append(self.content)
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> MemoryChunk:
        """Parse a Markdown file with YAML frontmatter into a MemoryChunk."""
        fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        if not fm_match:
            return cls(id="unknown", content=text)

        fm_text, body = fm_match.group(1), fm_match.group(2).strip()
        meta: dict = {}
        for line in fm_text.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()

        # Parse entities list
        entities_raw = meta.get("entities", "[]")
        entities_raw = entities_raw.strip("[]")
        entities = [e.strip() for e in entities_raw.split(",") if e.strip()]

        # Extract title from body
        title = ""
        content = body
        title_match = re.match(r"^#\s+(.+)\n?(.*)", body, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            content = title_match.group(2).strip()

        return cls(
            id=meta.get("id", "unknown"),
            session_id=meta.get("session_id", ""),
            created_at=meta.get("created_at", ""),
            entities=entities,
            importance=float(meta.get("importance", 0.5)),
            title=title,
            content=content,
        )


class FileStore:
    """Manages Markdown chunk files on disk."""

    def __init__(self, storage_path: str | Path):
        self.base_path = Path(storage_path)
        self.chunks_dir = self.base_path / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def save_chunk(self, chunk: MemoryChunk) -> Path:
        """Write a MemoryChunk to disk as Markdown."""
        path = self.chunks_dir / f"{chunk.id}.md"
        path.write_text(chunk.to_markdown(), encoding="utf-8")
        return path

    def load_chunk(self, chunk_id: str) -> MemoryChunk | None:
        """Read a MemoryChunk from disk."""
        path = self.chunks_dir / f"{chunk_id}.md"
        if not path.exists():
            return None
        return MemoryChunk.from_markdown(path.read_text(encoding="utf-8"))

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk file."""
        path = self.chunks_dir / f"{chunk_id}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    def list_chunks(self) -> list[str]:
        """List all chunk IDs on disk."""
        return [p.stem for p in self.chunks_dir.glob("*.md")]
