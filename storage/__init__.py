"""Storage module for S3-compatible object storage."""

from .object_store import get_bytes, put_bytes, uri_for, download_file, object_exists

__all__ = ["get_bytes", "put_bytes", "uri_for", "download_file", "object_exists"]
