"""
Object store abstraction for S3-compatible storage (AWS S3 and MinIO).

This module provides a unified interface for storing and retrieving objects
from S3-compatible storage, supporting both AWS S3 and MinIO through
configuration.

Environment Variables:
    S3_ENDPOINT_URL: MinIO or custom S3-compatible endpoint (optional)
    S3_FORCE_PATH_STYLE: Use path-style addressing (default: true for MinIO)
    S3_REGION: AWS region (default: us-east-1)
    AWS_ACCESS_KEY_ID: Access key
    AWS_SECRET_ACCESS_KEY: Secret key

Adapted from jarvis-recipes-server for adapter storage.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@lru_cache
def _get_s3_client() -> BaseClient:
    """
    Get or create a boto3 S3 client configured for AWS S3 or MinIO.

    Configuration is determined by environment variables:
    - S3_ENDPOINT_URL: If set, uses MinIO (or custom S3-compatible endpoint)
    - S3_FORCE_PATH_STYLE: If true, uses path-style addressing (required for MinIO)
    - S3_REGION: AWS region (default: us-east-1)
    - AWS_ACCESS_KEY_ID: Access key
    - AWS_SECRET_ACCESS_KEY: Secret key
    """
    endpoint_url = os.getenv("S3_ENDPOINT_URL")
    force_path_style = os.getenv("S3_FORCE_PATH_STYLE", "true").lower() == "true"
    region = os.getenv("S3_REGION", "us-east-1")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    client_kwargs = {}

    # Configure path-style addressing if needed (required for MinIO)
    if force_path_style:
        client_kwargs["config"] = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        )

    # Configure endpoint for MinIO or custom S3-compatible storage
    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    # Configure credentials
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key

    client = boto3.client("s3", region_name=region, **client_kwargs)

    return client


def put_bytes(bucket: str, key: str, content_type: str, data: bytes) -> str:
    """
    Upload bytes to object storage and return the URI.

    Args:
        bucket: Bucket name
        key: Object key (path)
        content_type: MIME type (e.g., "application/zip")
        data: Bytes to upload

    Returns:
        Full URI in format s3://bucket/key

    Raises:
        RuntimeError: If upload fails
    """
    try:
        client = _get_s3_client()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        uri = uri_for(bucket, key)
        logger.debug("Uploaded object to %s", uri)
        return uri
    except ClientError as exc:
        logger.exception("Failed to upload object to s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 upload failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error uploading object to s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 upload failed: {exc}") from exc


def get_bytes(bucket: str, key: str) -> bytes:
    """
    Download bytes from object storage.

    Args:
        bucket: Bucket name
        key: Object key (path)

    Returns:
        Object contents as bytes

    Raises:
        RuntimeError: If download fails
    """
    try:
        client = _get_s3_client()
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        logger.exception("Failed to download object from s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 download failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error downloading object from s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 download failed: {exc}") from exc


def download_file(bucket: str, key: str, local_path: Path) -> Path:
    """
    Download an object from S3 to a local file.

    Args:
        bucket: Bucket name
        key: Object key (path)
        local_path: Local file path to write to

    Returns:
        The local_path for chaining

    Raises:
        RuntimeError: If download fails
    """
    try:
        client = _get_s3_client()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(local_path))
        logger.debug("Downloaded s3://%s/%s to %s", bucket, key, local_path)
        return local_path
    except ClientError as exc:
        logger.exception("Failed to download s3://%s/%s to %s: %s", bucket, key, local_path, exc)
        raise RuntimeError(f"S3 download failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error downloading s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 download failed: {exc}") from exc


def object_exists(bucket: str, key: str) -> bool:
    """
    Check if an object exists in S3.

    Args:
        bucket: Bucket name
        key: Object key (path)

    Returns:
        True if object exists, False otherwise
    """
    try:
        client = _get_s3_client()
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "404":
            return False
        logger.exception("Error checking object existence s3://%s/%s: %s", bucket, key, exc)
        raise RuntimeError(f"S3 head_object failed: {exc}") from exc


def uri_for(bucket: str, key: str) -> str:
    """
    Generate a full URI for an object in object storage.

    Uses the s3:// scheme for both AWS S3 and MinIO (S3-compatible).
    The actual endpoint is determined by configuration, not the URI.

    Args:
        bucket: Bucket name
        key: Object key (path)

    Returns:
        Full URI in format s3://bucket/key
    """
    return f"s3://{bucket}/{key}"
