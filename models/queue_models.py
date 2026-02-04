"""Queue-related Pydantic models.

Models for job queue requests and responses used by the async processing system.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel

from models.api_models import Message, ResponseFormat


class ArtifactRefs(BaseModel):
    """References to external artifacts for job processing."""

    input_uri: Optional[str] = None
    schema_uri: Optional[str] = None
    prompt_uri: Optional[str] = None


class SamplingSettings(BaseModel):
    """Sampling parameters for LLM generation."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None


class TimeoutSettings(BaseModel):
    """Timeout configuration for job processing."""

    overall_seconds: Optional[int] = None
    per_attempt_seconds: Optional[int] = None


class CallbackInfo(BaseModel):
    """Callback configuration for job completion notification."""

    url: str
    auth_type: Optional[str] = None
    token: Optional[str] = None


class QueueRequest(BaseModel):
    """Request payload for queued LLM jobs."""

    artifacts: Optional[ArtifactRefs] = None
    model: str
    messages: List[Message]
    response_format: Optional[ResponseFormat] = None
    sampling: Optional[SamplingSettings] = None
    timeouts: Optional[TimeoutSettings] = None


class AdapterTrainRequest(BaseModel):
    """Request payload for adapter training jobs."""

    node_id: str
    base_model_id: str
    dataset_ref: dict
    dataset_hash: Optional[str] = None
    params: Optional[dict] = None


class EnqueueRequest(BaseModel):
    """Request to enqueue a job for async processing."""

    job_id: str
    job_type: str
    created_at: str
    priority: Optional[str] = "normal"
    trace_id: Optional[str] = None
    idempotency_key: str
    job_type_version: str = "v1"
    ttl_seconds: int = 86400
    metadata: Optional[dict] = None
    request: dict
    callback: CallbackInfo


class EnqueueResponse(BaseModel):
    """Response from job enqueue operation."""

    accepted: bool
    job_id: str
    deduped: bool = False
