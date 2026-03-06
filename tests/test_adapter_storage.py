"""
Tests for adapter storage service (S3/MinIO download and caching).

These tests mock the S3 client to run without actual S3/MinIO.
The settings service is mocked to return None so get_setting() falls
back to environment variables (the values patched in each test).

Run with:
    pytest tests/test_adapter_storage.py -v
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

import pytest

# Ensure get_setting() always falls back to env vars in tests
# (the real settings service would read from the DB, overriding env patches).
_no_settings_service = patch(
    "services.settings_helpers._get_settings_service", return_value=None
)


class TestAdapterStorageUnit:
    """Unit tests with mocked S3."""

    def test_build_s3_key(self) -> None:
        """Test S3 key construction."""
        with _no_settings_service:
            from services.adapter_storage import _build_s3_key

            key = _build_s3_key("abc123")
            assert key == "adapters/abc123/adapter.zip"

    def test_build_s3_key_with_custom_prefix(self) -> None:
        """Test S3 key with custom prefix."""
        with _no_settings_service, \
             patch.dict(os.environ, {"LLM_PROXY_ADAPTER_PREFIX": "custom/path"}):
            from services.adapter_storage import _build_s3_key

            key = _build_s3_key("xyz789")
            assert key == "custom/path/xyz789/adapter.zip"

    def test_get_local_adapter_dir(self) -> None:
        """Test local adapter directory path construction."""
        with _no_settings_service, \
             patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": "/custom/cache"}):
            from services.adapter_storage import _get_local_adapter_dir

            path = _get_local_adapter_dir("test123")
            assert path == Path("/custom/cache/test123")

    def test_adapter_exists_locally_when_missing(self) -> None:
        """Test that missing adapter returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}):
                from services.adapter_storage import adapter_exists_locally

                assert adapter_exists_locally("nonexistent") is False

    def test_adapter_exists_locally_when_present(self) -> None:
        """Test that existing adapter returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter directory with expected files
            adapter_dir = Path(tmpdir) / "test123"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake")

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}):
                from services.adapter_storage import adapter_exists_locally

                assert adapter_exists_locally("test123") is True

    def test_adapter_exists_in_s3(self) -> None:
        """Test S3 existence check."""
        with _no_settings_service, \
             patch("storage.object_store.object_exists") as mock_exists:
            mock_exists.return_value = True

            from services.adapter_storage import adapter_exists_in_s3

            result = adapter_exists_in_s3("abc123")

            assert result is True
            mock_exists.assert_called_once_with(
                "jarvis-llm-proxy",
                "adapters/abc123/adapter.zip"
            )

    def test_download_adapter_success(self) -> None:
        """Test successful adapter download and extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock zip file
            zip_content = _create_mock_adapter_zip()

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}), \
                 patch("storage.object_store.download_file") as mock_download:
                # Mock download_file to write our test zip
                def fake_download(bucket, key, local_path):
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(zip_content)
                    return local_path

                mock_download.side_effect = fake_download

                from services.adapter_storage import download_adapter

                result = download_adapter("test456")

                assert result is not None
                assert result.exists()
                assert (result / "adapter_config.json").exists()

    def test_download_adapter_uses_cache(self) -> None:
        """Test that download is skipped when adapter is cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create cached adapter
            adapter_dir = Path(tmpdir) / "cached123"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "adapters.safetensors").write_bytes(b"cached")

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}), \
                 patch("storage.object_store.download_file") as mock_download:
                from services.adapter_storage import download_adapter

                result = download_adapter("cached123")

                # Should return cached path without downloading
                assert result == adapter_dir
                mock_download.assert_not_called()

    def test_resolve_adapter_path_downloads_if_missing(self) -> None:
        """Test that resolve_adapter_path downloads from S3 if not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_content = _create_mock_adapter_zip()

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}), \
                 patch("storage.object_store.object_exists") as mock_exists, \
                 patch("storage.object_store.download_file") as mock_download:

                mock_exists.return_value = True

                def fake_download(bucket, key, local_path):
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(zip_content)
                    return local_path

                mock_download.side_effect = fake_download

                from services.adapter_storage import resolve_adapter_path

                result = resolve_adapter_path("new789")

                assert result is not None
                assert result.exists()
                mock_download.assert_called_once()

    def test_resolve_adapter_path_returns_none_if_not_in_s3(self) -> None:
        """Test that resolve_adapter_path returns None if adapter not in S3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}), \
                 patch("storage.object_store.object_exists") as mock_exists:
                mock_exists.return_value = False

                from services.adapter_storage import resolve_adapter_path

                result = resolve_adapter_path("missing999")

                assert result is None

    def test_clear_adapter_cache_single(self) -> None:
        """Test clearing a single adapter from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two adapters
            (Path(tmpdir) / "keep123").mkdir()
            (Path(tmpdir) / "delete456").mkdir()

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}):
                from services.adapter_storage import clear_adapter_cache

                clear_adapter_cache("delete456")

                assert (Path(tmpdir) / "keep123").exists()
                assert not (Path(tmpdir) / "delete456").exists()

    def test_clear_adapter_cache_all(self) -> None:
        """Test clearing all adapters from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapters
            (Path(tmpdir) / "adapter1").mkdir()
            (Path(tmpdir) / "adapter2").mkdir()

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}):
                from services.adapter_storage import clear_adapter_cache

                clear_adapter_cache()

                # Cache dir should exist but be empty
                assert Path(tmpdir).exists()
                assert list(Path(tmpdir).iterdir()) == []

    def test_get_cache_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter with files
            adapter_dir = Path(tmpdir) / "stats123"
            adapter_dir.mkdir()
            (adapter_dir / "config.json").write_text('{"test": true}')
            (adapter_dir / "weights.bin").write_bytes(b"x" * 1000)

            with _no_settings_service, \
                 patch.dict(os.environ, {"LLM_PROXY_ADAPTER_DIR": tmpdir}):
                from services.adapter_storage import get_cache_stats

                stats = get_cache_stats()

                assert stats["cached_adapters"] == 1
                assert stats["total_size_bytes"] > 0
                assert "stats123" in stats["adapter_hashes"]


def _create_mock_adapter_zip() -> bytes:
    """Create a mock adapter zip file in memory."""
    import io

    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as zf:
        zf.writestr("adapter_config.json", '{"rank": 8, "alpha": 16}')
        zf.writestr("adapter_model.safetensors", b"fake_weights_data")

    return buffer.getvalue()
