"""
LRU Cache for LoRA adapters.

This service provides an LRU (Least Recently Used) cache for adapter paths.
It wraps adapter_storage to add access tracking and eviction.

Key responsibilities:
1. Track which adapters have been accessed recently (LRU order)
2. Evict least recently used adapters when at capacity
3. Coordinate with vLLM's max_loras setting for GPU memory

Architecture:
    Request → AdapterCache.get(hash) → adapter_storage.resolve_adapter_path() → Path
                     ↓
              Update LRU order
                     ↓
              Evict if over capacity

Based on spike test results:
- Cold load overhead: ~145ms (acceptable)
- Cached swap overhead: ~0ms (vLLM handles GPU memory)
- Disk cache prevents S3 round-trips

Environment Variables:
    LLM_PROXY_ADAPTER_CACHE_MAX_SIZE: Max adapters to track (default: 10)
    LLM_PROXY_ADAPTER_CACHE_EVICT_DISK: Evict from disk cache too (default: false)
"""

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from services import adapter_storage

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Metadata for a cached adapter."""

    hash: str
    path: Path
    last_accessed: datetime
    access_count: int = 0


class AdapterCache:
    """
    LRU cache for LoRA adapter paths.

    Thread-safe implementation using OrderedDict for LRU tracking.
    Coordinates with adapter_storage for disk/S3 operations.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        evict_disk: Optional[bool] = None,
    ):
        """
        Initialize the adapter cache.

        Args:
            max_size: Maximum number of adapters to track (default from env)
            evict_disk: Whether to evict from disk cache too (default from env)
        """
        self._max_size = max_size or int(
            os.getenv("LLM_PROXY_ADAPTER_CACHE_MAX_SIZE", "10")
        )
        self._evict_disk = evict_disk if evict_disk is not None else (
            os.getenv("LLM_PROXY_ADAPTER_CACHE_EVICT_DISK", "false").lower() == "true"
        )
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        logger.info(
            "AdapterCache initialized: max_size=%d, evict_disk=%s",
            self._max_size,
            self._evict_disk,
        )

    def get(self, adapter_hash: str) -> Optional[Path]:
        """
        Get adapter path, resolving from S3 if needed.

        This is the main entry point. It:
        1. Checks if adapter is in LRU cache
        2. If not, resolves via adapter_storage (disk/S3)
        3. Updates LRU order
        4. Evicts if over capacity

        Args:
            adapter_hash: Unique adapter identifier

        Returns:
            Path to local adapter directory, or None if not found
        """
        with self._lock:
            # Check LRU cache first
            if adapter_hash in self._cache:
                return self._touch(adapter_hash)

            # Resolve via storage layer
            path = adapter_storage.resolve_adapter_path(adapter_hash)
            if path is None:
                logger.warning("Adapter %s not found", adapter_hash)
                return None

            # Add to cache
            self._add(adapter_hash, path)
            return path

    def _touch(self, adapter_hash: str) -> Path:
        """
        Update access time and move to end of LRU order.

        Must be called with lock held.
        """
        entry = self._cache[adapter_hash]
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        # Move to end (most recently used)
        self._cache.move_to_end(adapter_hash)

        logger.debug(
            "Cache hit: %s (access_count=%d)",
            adapter_hash,
            entry.access_count,
        )
        return entry.path

    def _add(self, adapter_hash: str, path: Path) -> None:
        """
        Add adapter to cache, evicting if necessary.

        Must be called with lock held.
        """
        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            self._evict_lru()

        # Add new entry
        entry = CacheEntry(
            hash=adapter_hash,
            path=path,
            last_accessed=datetime.now(),
            access_count=1,
        )
        self._cache[adapter_hash] = entry

        logger.info(
            "Cache add: %s (size=%d/%d)",
            adapter_hash,
            len(self._cache),
            self._max_size,
        )

    def _evict_lru(self) -> None:
        """
        Evict least recently used adapter.

        Must be called with lock held.
        """
        if not self._cache:
            return

        # Pop from front (least recently used)
        adapter_hash, entry = self._cache.popitem(last=False)

        logger.info(
            "Cache evict: %s (last_accessed=%s, access_count=%d)",
            adapter_hash,
            entry.last_accessed.isoformat(),
            entry.access_count,
        )

        # Optionally evict from disk cache too
        if self._evict_disk:
            adapter_storage.clear_adapter_cache(adapter_hash)
            logger.info("Disk cache evicted: %s", adapter_hash)

    def contains(self, adapter_hash: str) -> bool:
        """Check if adapter is in LRU cache (not disk cache)."""
        with self._lock:
            return adapter_hash in self._cache

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all cached adapters."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats including LRU and disk cache info
        """
        with self._lock:
            entries = [
                {
                    "hash": e.hash,
                    "path": str(e.path),
                    "last_accessed": e.last_accessed.isoformat(),
                    "access_count": e.access_count,
                }
                for e in self._cache.values()
            ]

            disk_stats = adapter_storage.get_cache_stats()

            return {
                "lru_cache": {
                    "size": len(self._cache),
                    "max_size": self._max_size,
                    "evict_disk": self._evict_disk,
                    "entries": entries,
                },
                "disk_cache": disk_stats,
            }


# Global singleton instance
_cache_instance: Optional[AdapterCache] = None
_cache_lock = threading.Lock()


def get_cache() -> AdapterCache:
    """
    Get the global AdapterCache instance.

    Returns:
        The singleton AdapterCache
    """
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = AdapterCache()
    return _cache_instance


def get_adapter_path(adapter_hash: str) -> Optional[Path]:
    """
    Convenience function to get adapter path from global cache.

    Args:
        adapter_hash: Unique adapter identifier

    Returns:
        Path to local adapter directory, or None if not found
    """
    return get_cache().get(adapter_hash)
