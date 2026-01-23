"""
Tests for LRU adapter cache.

Run with:
    pytest tests/test_adapter_cache.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestAdapterCacheLRU:
    """Unit tests for LRU cache behavior."""

    def test_cache_initialization(self) -> None:
        """Test cache initializes with correct defaults."""
        from services.adapter_cache import AdapterCache

        cache = AdapterCache(max_size=5, evict_disk=False)

        assert cache._max_size == 5
        assert cache._evict_disk is False
        assert cache.size() == 0

    def test_cache_add_and_get(self) -> None:
        """Test adding and retrieving adapters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test123"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("services.adapter_cache.adapter_storage") as mock_storage:
                mock_storage.resolve_adapter_path.return_value = adapter_path

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=5)

                # First get should resolve from storage
                result = cache.get("test123")
                assert result == adapter_path
                assert cache.size() == 1
                mock_storage.resolve_adapter_path.assert_called_once_with("test123")

                # Second get should hit cache (no storage call)
                mock_storage.reset_mock()
                result2 = cache.get("test123")
                assert result2 == adapter_path
                mock_storage.resolve_adapter_path.assert_not_called()

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter paths
            paths = {}
            for i in range(4):
                p = Path(tmpdir) / f"adapter{i}"
                p.mkdir()
                (p / "adapter_config.json").write_text("{}")
                paths[f"adapter{i}"] = p

            with patch("services.adapter_cache.adapter_storage") as mock_storage:

                def resolve_side_effect(adapter_hash: str) -> Path:
                    return paths.get(adapter_hash)

                mock_storage.resolve_adapter_path.side_effect = resolve_side_effect
                mock_storage.clear_adapter_cache = MagicMock()

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=3, evict_disk=False)

                # Add 3 adapters (fills cache)
                cache.get("adapter0")
                cache.get("adapter1")
                cache.get("adapter2")
                assert cache.size() == 3

                # Access adapter0 to make it recently used
                cache.get("adapter0")

                # Add adapter3 - should evict adapter1 (LRU)
                cache.get("adapter3")
                assert cache.size() == 3
                assert cache.contains("adapter0")  # Recently used
                assert not cache.contains("adapter1")  # Evicted (LRU)
                assert cache.contains("adapter2")
                assert cache.contains("adapter3")  # Newly added

    def test_cache_lru_eviction_with_disk(self) -> None:
        """Test LRU eviction also clears disk cache when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for i in range(3):
                p = Path(tmpdir) / f"adapter{i}"
                p.mkdir()
                (p / "adapter_config.json").write_text("{}")
                paths[f"adapter{i}"] = p

            with patch("services.adapter_cache.adapter_storage") as mock_storage:

                def resolve_side_effect(adapter_hash: str) -> Path:
                    return paths.get(adapter_hash)

                mock_storage.resolve_adapter_path.side_effect = resolve_side_effect
                mock_storage.clear_adapter_cache = MagicMock()

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=2, evict_disk=True)

                # Fill cache
                cache.get("adapter0")
                cache.get("adapter1")

                # Add third - should evict adapter0 and clear from disk
                cache.get("adapter2")

                mock_storage.clear_adapter_cache.assert_called_once_with("adapter0")

    def test_cache_returns_none_for_missing(self) -> None:
        """Test cache returns None for adapters not in storage."""
        with patch("services.adapter_cache.adapter_storage") as mock_storage:
            mock_storage.resolve_adapter_path.return_value = None

            from services.adapter_cache import AdapterCache

            cache = AdapterCache(max_size=5)

            result = cache.get("nonexistent")
            assert result is None
            assert cache.size() == 0

    def test_cache_contains(self) -> None:
        """Test contains check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test123"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("services.adapter_cache.adapter_storage") as mock_storage:
                mock_storage.resolve_adapter_path.return_value = adapter_path

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=5)

                assert not cache.contains("test123")
                cache.get("test123")
                assert cache.contains("test123")

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test123"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("services.adapter_cache.adapter_storage") as mock_storage:
                mock_storage.resolve_adapter_path.return_value = adapter_path

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=5)
                cache.get("test123")
                assert cache.size() == 1

                cache.clear()
                assert cache.size() == 0
                assert not cache.contains("test123")

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test123"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("services.adapter_cache.adapter_storage") as mock_storage:
                mock_storage.resolve_adapter_path.return_value = adapter_path
                mock_storage.get_cache_stats.return_value = {
                    "cached_adapters": 1,
                    "total_size_bytes": 1000,
                }

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=10, evict_disk=False)
                cache.get("test123")
                cache.get("test123")  # Access twice

                stats = cache.get_stats()

                assert stats["lru_cache"]["size"] == 1
                assert stats["lru_cache"]["max_size"] == 10
                assert stats["lru_cache"]["evict_disk"] is False
                assert len(stats["lru_cache"]["entries"]) == 1
                assert stats["lru_cache"]["entries"][0]["hash"] == "test123"
                assert stats["lru_cache"]["entries"][0]["access_count"] == 2

    def test_cache_thread_safety(self) -> None:
        """Test cache is thread-safe with concurrent access."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for i in range(10):
                p = Path(tmpdir) / f"adapter{i}"
                p.mkdir()
                (p / "adapter_config.json").write_text("{}")
                paths[f"adapter{i}"] = p

            with patch("services.adapter_cache.adapter_storage") as mock_storage:

                def resolve_side_effect(adapter_hash: str) -> Path:
                    return paths.get(adapter_hash)

                mock_storage.resolve_adapter_path.side_effect = resolve_side_effect
                mock_storage.clear_adapter_cache = MagicMock()

                from services.adapter_cache import AdapterCache

                cache = AdapterCache(max_size=5)
                errors = []

                def worker(adapter_ids: list) -> None:
                    try:
                        for adapter_id in adapter_ids:
                            cache.get(adapter_id)
                    except Exception as e:
                        errors.append(e)

                threads = [
                    threading.Thread(
                        target=worker, args=([f"adapter{i % 10}" for i in range(100)],)
                    )
                    for _ in range(5)
                ]

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                assert len(errors) == 0
                assert cache.size() <= 5  # Should not exceed max


class TestAdapterCacheGlobalInstance:
    """Tests for global cache instance."""

    def test_get_cache_singleton(self) -> None:
        """Test get_cache returns singleton."""
        from services import adapter_cache

        # Reset singleton for test
        adapter_cache._cache_instance = None

        cache1 = adapter_cache.get_cache()
        cache2 = adapter_cache.get_cache()

        assert cache1 is cache2

    def test_get_adapter_path_convenience(self) -> None:
        """Test convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test456"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("services.adapter_cache.adapter_storage") as mock_storage:
                mock_storage.resolve_adapter_path.return_value = adapter_path

                from services import adapter_cache

                # Reset singleton
                adapter_cache._cache_instance = None

                result = adapter_cache.get_adapter_path("test456")
                assert result == adapter_path


if __name__ == "__main__":
    print("Running adapter cache tests...")

    test = TestAdapterCacheLRU()

    test.test_cache_initialization()
    print("  ✓ test_cache_initialization")

    test.test_cache_add_and_get()
    print("  ✓ test_cache_add_and_get")

    test.test_cache_lru_eviction()
    print("  ✓ test_cache_lru_eviction")

    test.test_cache_returns_none_for_missing()
    print("  ✓ test_cache_returns_none_for_missing")

    test.test_cache_contains()
    print("  ✓ test_cache_contains")

    test.test_cache_clear()
    print("  ✓ test_cache_clear")

    test.test_cache_stats()
    print("  ✓ test_cache_stats")

    test.test_cache_thread_safety()
    print("  ✓ test_cache_thread_safety")

    print("\nAll tests passed!")
