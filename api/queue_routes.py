"""Job queue API routes.

Endpoints for enqueueing async LLM processing jobs.
"""

import logging
import os
import time
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from auth.app_auth import require_app_auth
from models.queue_models import (
    AdapterTrainRequest,
    EnqueueRequest,
    EnqueueResponse,
    QueueRequest,
)
from queues.redis_queue import (
    get_redis_connection,
    mark_deduped,
    existing_dedup,
    enqueue_job,
)
from services.training_job_service import create_queued_training_job

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/internal/queue", tags=["queue"])


def _parse_created_at(ts: str) -> float:
    """Parse a timestamp string to Unix timestamp."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        try:
            return float(ts)
        except (ValueError, TypeError):
            return time.time()


@router.post(
    "/enqueue",
    response_model=EnqueueResponse,
    dependencies=[Depends(require_app_auth)],
)
async def enqueue_job_endpoint(req: EnqueueRequest, request: Request):
    """Enqueue a job for asynchronous LLM processing."""
    job_type = (req.job_type or "").lower()
    if job_type == "adapter_train":
        # Validate adapter training payload
        try:
            AdapterTrainRequest(**(req.request or {}))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid adapter_train request: {e}",
                        "code": "invalid_adapter_request",
                    }
                },
            )
    else:
        # Validate standard LLM request payload
        try:
            llm_request = QueueRequest(**(req.request or {}))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Invalid request: {e}",
                        "code": "invalid_request",
                    }
                },
            )
        # Basic validation for schema requirement
        if (
            llm_request.response_format
            and llm_request.response_format.type == "json_object"
        ):
            if llm_request.response_format.json_schema is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "json_schema required when response_format.type=json_object",
                            "code": "missing_schema",
                        }
                    },
                )

    now = time.time()
    created_ts = _parse_created_at(req.created_at)
    if created_ts + req.ttl_seconds < now:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Job already expired",
                    "code": "expired",
                }
            },
        )

    conn = get_redis_connection()
    # Dedup check
    if existing_dedup(conn, req.job_id, req.idempotency_key):
        # Even for deduped requests, ensure DB record exists for adapter training
        if job_type == "adapter_train":
            train_req = req.request or {}
            create_queued_training_job(
                job_id=req.job_id,
                node_id=train_req.get("node_id", ""),
                base_model_id=train_req.get("base_model_id", ""),
                dataset_hash=train_req.get("dataset_hash"),
            )
        return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=True)

    if not mark_deduped(conn, req.job_id, req.idempotency_key, req.ttl_seconds):
        # Another writer won the race - ensure DB record exists
        if job_type == "adapter_train":
            train_req = req.request or {}
            create_queued_training_job(
                job_id=req.job_id,
                node_id=train_req.get("node_id", ""),
                base_model_id=train_req.get("base_model_id", ""),
                dataset_hash=train_req.get("dataset_hash"),
            )
        return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=True)

    payload = req.dict()
    payload["received_at"] = datetime.utcnow().isoformat()
    payload["queue_name"] = os.getenv("LLM_PROXY_QUEUE_NAME", "llm_proxy_jobs")

    try:
        enqueue_job(payload, req.ttl_seconds, queue_name=payload["queue_name"])
    except Exception as e:
        # Roll back dedupe key on failure to enqueue
        conn.delete(f"llmproxy:dedupe:{req.job_id}:{req.idempotency_key}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_server_error",
                    "message": f"Failed to enqueue job: {e}",
                    "code": "enqueue_failed",
                }
            },
        )

    # For adapter training jobs, create DB record immediately with QUEUED status
    if job_type == "adapter_train":
        train_req = req.request or {}
        create_queued_training_job(
            job_id=req.job_id,
            node_id=train_req.get("node_id", ""),
            base_model_id=train_req.get("base_model_id", ""),
            dataset_hash=train_req.get("dataset_hash"),
        )

    return EnqueueResponse(accepted=True, job_id=req.job_id, deduped=False)
