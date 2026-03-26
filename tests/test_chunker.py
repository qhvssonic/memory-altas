"""Tests for semantic chunker."""

from memory_atlas.ingestion.chunker import Chunker


class TestChunker:
    def test_paragraph_chunking(self):
        text = "First paragraph about auth.\n\nSecond paragraph about database.\n\nThird one."
        chunks = Chunker(strategy="paragraph").chunk(text)
        assert len(chunks) >= 1
        assert "auth" in chunks[0].text

    def test_turn_chunking(self):
        text = "Human: How do I fix the auth bug?\nAssistant: Check the JWT expiry.\nHuman: Thanks!"
        chunks = Chunker(strategy="turn").chunk(text)
        assert len(chunks) >= 2

    def test_fixed_chunking(self):
        text = "a" * 500
        chunks = Chunker(strategy="fixed", max_chunk_size=200, overlap=50).chunk(text)
        assert len(chunks) >= 2

    def test_empty_input(self):
        chunks = Chunker().chunk("")
        assert len(chunks) == 1

    def test_paragraph_merge_small(self):
        text = "Short.\n\nAlso short.\n\nStill short."
        chunks = Chunker(strategy="paragraph", max_chunk_size=1000).chunk(text)
        # Small paragraphs should be merged
        assert len(chunks) == 1
