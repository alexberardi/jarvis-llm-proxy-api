"""
Adapter storage service for LoRA adapters.

This service handles:
1. Resolving adapter hash to local path
2. Downloading adapter zip from S3 (if not cached locally)
3. Extracting to local directory

Local structure (flat by hash):
    /tmp/jarvis-adapters/
    └── {hash}/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...

S3 structure (for remote storage):
    s3://{bucket}/{prefix}/{hash}/adapter.zip

Environment Variables:
    LLM_PROXY_ADAPTER_DIR: Local adapter directory (default: /tmp/jarvis-adapters)
    LLM_PROXY_ADAPTER_BUCKET: S3 bucket name (default: jarvis-llm-proxy)
    LLM_PROXY_ADAPTER_PREFIX: S3 key prefix (default: adapters)
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from storage import object_store

logger = logging.getLogger(__name__)


def _get_bucket() -> str:
    """Get the adapter bucket name from environment."""
    return os.getenv("LLM_PROXY_ADAPTER_BUCKET", "jarvis-llm-proxy")


def _get_prefix() -> str:
    """Get the adapter key prefix from environment."""
    return os.getenv("LLM_PROXY_ADAPTER_PREFIX", "adapters")


def _get_adapter_dir() -> Path:
    """Get the local adapter directory."""
    adapter_dir = os.getenv("LLM_PROXY_ADAPTER_DIR", "/tmp/jarvis-adapters")
    return Path(adapter_dir)


def _build_s3_key(adapter_hash: str) -> str:
    """
    Build S3 key for an adapter.

    Args:
        adapter_hash: Unique adapter identifier (e.g., "b2b8ccb4")

    Returns:
        S3 key like "adapters/b2b8ccb4/adapter.zip"
    """
    prefix = _get_prefix()
    return f"{prefix}/{adapter_hash}/adapter.zip"


def _get_local_adapter_dir(adapter_hash: str) -> Path:
    """
    Get local directory path for a cached adapter.

    Args:
        adapter_hash: Unique adapter identifier

    Returns:
        Path like /tmp/jarvis-adapters/cache/b2b8ccb4/
    """
    return _get_adapter_dir() / adapter_hash


def adapter_exists_locally(adapter_hash: str) -> bool:
    """
    Check if an adapter is already cached locally.

    Args:
        adapter_hash: Unique adapter identifier

    Returns:
        True if adapter directory exists and contains expected files
    """
    adapter_dir = _get_local_adapter_dir(adapter_hash)
    if not adapter_dir.exists():
        return False

    # Check for expected adapter files (either format)
    has_safetensors = (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapters.safetensors").exists()
    )
    has_config = (adapter_dir / "adapter_config.json").exists()

    return has_safetensors or has_config


def adapter_exists_in_s3(adapter_hash: str) -> bool:
    """
    Check if an adapter exists in S3.

    Args:
        adapter_hash: Unique adapter identifier

    Returns:
        True if adapter zip exists in S3
    """
    bucket = _get_bucket()
    key = _build_s3_key(adapter_hash)
    return object_store.object_exists(bucket, key)


def download_adapter(adapter_hash: str, force: bool = False) -> Optional[Path]:
    """
    Download and extract an adapter from S3 to local cache.

    Args:
        adapter_hash: Unique adapter identifier
        force: If True, re-download even if cached locally

    Returns:
        Path to local adapter directory, or None if download failed

    Raises:
        RuntimeError: If S3 download fails
    """
    adapter_dir = _get_local_adapter_dir(adapter_hash)

    # Check local cache first
    if not force and adapter_exists_locally(adapter_hash):
        logger.info("Adapter %s already cached at %s", adapter_hash, adapter_dir)
        return adapter_dir

    # Download from S3
    bucket = _get_bucket()
    key = _build_s3_key(adapter_hash)

    logger.info("Downloading adapter %s from s3://%s/%s", adapter_hash, bucket, key)

    try:
        # Download zip to temp location
        zip_path = _get_adapter_dir() / f"{adapter_hash}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        object_store.download_file(bucket, key, zip_path)

        # Extract zip
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(adapter_dir)

        # Clean up zip
        zip_path.unlink()

        logger.info("Adapter %s extracted to %s", adapter_hash, adapter_dir)
        return adapter_dir

    except Exception as exc:
        logger.exception("Failed to download adapter %s: %s", adapter_hash, exc)
        # Clean up partial download
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        raise


def resolve_adapter_path(adapter_hash: str) -> Optional[Path]:
    """
    Resolve adapter hash to local path, downloading if necessary.

    This is the main entry point for adapter resolution. It:
    1. Checks local cache
    2. Downloads from S3 if not cached
    3. Returns local path for backend to use

    Args:
        adapter_hash: Unique adapter identifier

    Returns:
        Path to local adapter directory, or None if not found

    Raises:
        RuntimeError: If adapter not found in S3 or download fails
    """
    # Check local cache
    if adapter_exists_locally(adapter_hash):
        adapter_dir = _get_local_adapter_dir(adapter_hash)
        logger.debug("Adapter %s found in cache: %s", adapter_hash, adapter_dir)
        return adapter_dir

    # Check S3
    if not adapter_exists_in_s3(adapter_hash):
        logger.warning("Adapter %s not found in S3", adapter_hash)
        return None

    # Download and extract
    return download_adapter(adapter_hash)


def clear_adapter_cache(adapter_hash: Optional[str] = None) -> None:
    """
    Clear adapter cache.

    Args:
        adapter_hash: If provided, clear only this adapter. Otherwise clear all.
    """
    if adapter_hash:
        adapter_dir = _get_local_adapter_dir(adapter_hash)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
            logger.info("Cleared adapter cache: %s", adapter_hash)
    else:
        cache_dir = _get_adapter_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all adapter cache")


def get_cache_stats() -> dict:
    """
    Get statistics about the adapter cache.

    Returns:
        Dict with cache statistics
    """
    cache_dir = _get_adapter_dir()
    if not cache_dir.exists():
        return {"cached_adapters": 0, "total_size_bytes": 0}

    adapters = [d for d in cache_dir.iterdir() if d.is_dir()]
    total_size = sum(
        f.stat().st_size
        for adapter_dir in adapters
        for f in adapter_dir.rglob("*")
        if f.is_file()
    )

    return {
        "cached_adapters": len(adapters),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "adapter_hashes": [d.name for d in adapters],
    }
