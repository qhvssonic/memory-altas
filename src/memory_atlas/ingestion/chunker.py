"""Semantic chunker: splits conversations/documents into meaningful segments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A semantically coherent segment of content."""

    text: str
    index: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Chunker:
    """Split input content into semantic chunks.

    Strategies:
    - 'turn': Split by conversation turns (default for dialogues)
    - 'paragraph': Split by double newlines
    - 'fixed': Fixed-size with overlap
    """

    def __init__(
        self,
        strategy: str = "paragraph",
        max_chunk_size: int = 1000,
        overlap: int = 100,
    ):
        self.strategy = strategy
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        if self.strategy == "turn":
            return self._chunk_by_turns(text)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text)
        else:
            return self._chunk_fixed(text)

    def _chunk_by_turns(self, text: str) -> list[Chunk]:
        """Split dialogue by speaker turns (Human:/Assistant:/User:/AI:)."""
        import re
        parts = re.split(r"\n(?=(?:Human|Assistant|User|AI|System):)", text)
        chunks = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                chunks.append(Chunk(text=part, index=i))
        return chunks or [Chunk(text=text, index=0)]

    def _chunk_by_paragraph(self, text: str) -> list[Chunk]:
        """Split by double newlines, merging small paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        buffer = ""
        idx = 0
        for para in paragraphs:
            if len(buffer) + len(para) > self.max_chunk_size and buffer:
                chunks.append(Chunk(text=buffer.strip(), index=idx))
                idx += 1
                buffer = ""
            buffer += para + "\n\n"
        if buffer.strip():
            chunks.append(Chunk(text=buffer.strip(), index=idx))
        return chunks or [Chunk(text=text, index=0)]

    def _chunk_fixed(self, text: str) -> list[Chunk]:
        """Fixed-size chunking with overlap."""
        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.max_chunk_size
            chunks.append(Chunk(text=text[start:end], index=idx))
            start = end - self.overlap
            idx += 1
        return chunks
