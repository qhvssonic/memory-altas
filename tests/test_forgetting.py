"""Tests for forgetting mechanism."""

import pytest
from datetime import datetime, timezone, timedelta

from memory_atlas.core.registry import Registry, MemoryRecord
from memory_atlas.storage.file_store import FileStore, MemoryChunk
from memory_atlas.maintenance.forgetting import ForgettingManager


@pytest.fixture
def setup(tmp_path):
    reg = Registry(tmp_path / "test.duckdb")
    fs = FileStore(tmp_path)
    yield reg, fs
    reg.close()


class TestForgettingManager:
    def test_activity_score_recent_high(self, setup):
        reg, fs = setup
        fm = ForgettingManager(reg, fs, decay_lambda=0.1)
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id="m1", importance_score=0.9, access_count=10,
            last_accessed_at=now.isoformat(),
        )
        score = fm.compute_activity(rec, now)
        assert score > 0.5  # recent + important + accessed = high

    def test_activity_score_old_low(self, setup):
        reg, fs = setup
        fm = ForgettingManager(reg, fs, decay_lambda=0.1)
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()
        rec = MemoryRecord(
            id="m2", importance_score=0.2, access_count=0,
            last_accessed_at=old,
        )
        score = fm.compute_activity(rec, now)
        assert score < 0.05  # old + unimportant + never accessed = low

    def test_compress_low_activity(self, setup):
        reg, fs = setup
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=90)).isoformat()

        # Insert a low-activity memory with L2 file
        reg.insert_memory(MemoryRecord(
            id="low1", label="old stuff", summary="old summary",
            importance_score=0.1, access_count=0,
            last_accessed_at=old, file_path="chunks/low1.md",
        ))
        fs.save_chunk(MemoryChunk(id="low1", content="old content"))

        fm = ForgettingManager(reg, fs, decay_lambda=0.1,
                               compress_threshold=0.5, archive_threshold=0.001)
        result = fm.run_cycle()
        assert result.compressed >= 1
        assert fs.load_chunk("low1") is None  # L2 file deleted

    def test_archive_very_low_activity(self, setup):
        reg, fs = setup
        now = datetime.now(timezone.utc)
        very_old = (now - timedelta(days=365)).isoformat()

        reg.insert_memory(MemoryRecord(
            id="ancient", label="ancient", summary="ancient summary",
            importance_score=0.01, access_count=0,
            last_accessed_at=very_old, file_path="chunks/ancient.md",
        ))
        fs.save_chunk(MemoryChunk(id="ancient", content="ancient content"))

        fm = ForgettingManager(reg, fs, decay_lambda=0.1,
                               compress_threshold=0.5, archive_threshold=0.5)
        result = fm.run_cycle()
        assert result.archived >= 1
        rec = reg.get_memory("ancient")
        assert rec.summary == ""  # summary cleared
        assert rec.cache_tier == "archived"

    def test_keep_high_activity(self, setup):
        reg, fs = setup
        now = datetime.now(timezone.utc)

        reg.insert_memory(MemoryRecord(
            id="active", label="active", summary="active summary",
            importance_score=0.9, access_count=50,
            last_accessed_at=now.isoformat(),
        ))

        fm = ForgettingManager(reg, fs, decay_lambda=0.1)
        result = fm.run_cycle()
        assert result.kept >= 1
        assert result.compressed == 0
        assert result.archived == 0
