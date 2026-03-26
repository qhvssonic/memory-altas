"""Tests for configuration management."""

from memory_atlas.config import MemoryAtlasConfig


class TestConfig:
    def test_defaults(self):
        cfg = MemoryAtlasConfig()
        assert cfg.max_memory_tokens == 2000
        assert cfg.embedding_model == "local"
        assert cfg.hot_capacity == 20

    def test_save_and_load(self, tmp_path):
        cfg = MemoryAtlasConfig(storage_path=str(tmp_path), max_memory_tokens=3000)
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = MemoryAtlasConfig.load(path)
        assert loaded.max_memory_tokens == 3000

    def test_load_nonexistent(self, tmp_path):
        cfg = MemoryAtlasConfig.load(tmp_path / "nope.json")
        assert cfg.max_memory_tokens == 2000  # defaults

    def test_custom_kwargs(self):
        cfg = MemoryAtlasConfig(
            hot_capacity=50,
            warm_capacity=200,
            prefetch_enabled=False,
        )
        assert cfg.hot_capacity == 50
        assert cfg.warm_capacity == 200
        assert cfg.prefetch_enabled is False
