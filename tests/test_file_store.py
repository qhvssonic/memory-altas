"""Tests for Markdown file store."""

import pytest
import tempfile
from pathlib import Path
from memory_atlas.storage.file_store import FileStore, MemoryChunk


@pytest.fixture
def store(tmp_path):
    return FileStore(tmp_path)


class TestMemoryChunk:
    def test_roundtrip(self):
        chunk = MemoryChunk(
            id="c001",
            session_id="s1",
            created_at="2026-03-26T10:00:00Z",
            entities=["auth", "jwt"],
            importance=0.85,
            title="JWT token bug",
            content="Refresh token has a race condition.",
        )
        md = chunk.to_markdown()
        parsed = MemoryChunk.from_markdown(md)
        assert parsed.id == "c001"
        assert parsed.session_id == "s1"
        assert parsed.entities == ["auth", "jwt"]
        assert parsed.importance == 0.85
        assert parsed.title == "JWT token bug"
        assert "race condition" in parsed.content

    def test_from_markdown_no_frontmatter(self):
        chunk = MemoryChunk.from_markdown("Just plain text")
        assert chunk.id == "unknown"
        assert chunk.content == "Just plain text"


class TestFileStore:
    def test_save_and_load(self, store):
        chunk = MemoryChunk(id="t001", title="Test", content="Hello world")
        store.save_chunk(chunk)
        loaded = store.load_chunk("t001")
        assert loaded is not None
        assert loaded.id == "t001"
        assert "Hello world" in loaded.content

    def test_load_nonexistent(self, store):
        assert store.load_chunk("nope") is None

    def test_delete(self, store):
        store.save_chunk(MemoryChunk(id="d001", content="delete me"))
        assert store.delete_chunk("d001")
        assert store.load_chunk("d001") is None

    def test_list_chunks(self, store):
        store.save_chunk(MemoryChunk(id="a", content="x"))
        store.save_chunk(MemoryChunk(id="b", content="y"))
        ids = store.list_chunks()
        assert set(ids) == {"a", "b"}
