import os
import signal
import sys
import time
import asyncio
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from queues.redis_queue import current_timestamp_ms


def _setup_work_horse_signals():
    """Set up signal handlers for the RQ work horse process.

    This MUST be called at the start of every job because:
    1. RQ forks a work horse to run each job
    2. Signal handlers may be reset during fork/exec
    3. SIGPIPE from inherited sockets/pipes will kill the process otherwise
    """
    # Ignore SIGPIPE - prevents death from broken pipes/sockets
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)


def _elapsed_ms(start_ms: int) -> int:
    return max(0, current_timestamp_ms() - start_ms)


def _safe_print(msg: str):
    """Print with [worker] prefix, ignoring broken pipe errors."""
    try:
        print(f"[worker] {msg}", flush=True)
    except BrokenPipeError:
        pass


def process_llm_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ task: process a queued LLM job and invoke callback.
    """
    # CRITICAL: Set up signal handlers first thing in the work horse
    # This prevents SIGPIPE from killing the forked process
    _setup_work_horse_signals()

    # Debug: confirm we reached the job function (if this doesn't print, SIGPIPE is earlier)
    sys.stderr.write(f"[worker] DEBUG: process_llm_job started, pid={os.getpid()}\n")
    sys.stderr.flush()

    job_type = (payload.get("job_type") or "").lower()
    if job_type == "adapter_train":
        return _process_adapter_train_job(payload)
    if job_type == "vision_inference":
        return _process_vision_job(payload)
    return _process_chat_job(payload)


def _process_chat_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    started_ms = current_timestamp_ms()

    job_id = payload.get("job_id")
    ttl_seconds = payload.get("ttl_seconds") or 0
    created_at = payload.get("created_at")
    callback = payload.get("callback") or {}
    metadata = payload.get("metadata") or {}
    request_body = payload.get("request") or {}
    trace_id = payload.get("trace_id")

    per_attempt_timeout = (
        payload.get("request", {}).get("timeouts", {}).get("per_attempt_seconds")
        or os.getenv("LLM_PROXY_PER_ATTEMPT_TIMEOUT", None)
    )
    per_attempt_timeout = float(per_attempt_timeout) if per_attempt_timeout else None

    # Expiry check
    now_s = time.time()
    if ttl_seconds and created_at:
        try:
            created_s = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            try:
                created_s = float(created_at)
            except (ValueError, TypeError):
                created_s = now_s
        if created_s + ttl_seconds < now_s:
            return _send_callback(
                callback,
                job_id,
                status="failed",
                error={"code": "expired", "message": "Job expired before processing"},
                result=None,
                timing={"processing_ms": _elapsed_ms(started_ms)},
                trace_id=trace_id,
            )

    status = "failed"
    result = None
    error: Optional[Dict[str, Any]] = None
    processing_ms = 0

    _safe_print(f"📦 Payload: {payload}")
    _safe_print(f"▶️  Starting job_id={job_id} job_type={payload.get('job_type')} model={request_body.get('model')}")

    try:
        # Direct call via model service if configured, otherwise in-process
        from models.api_models import ChatCompletionRequest
        import httpx

        sampling = (request_body.get("sampling") or {})
        response_format = request_body.get("response_format")

        req_obj = ChatCompletionRequest(
            model=request_body.get("model"),
            messages=request_body.get("messages") or [],
            temperature=sampling.get("temperature", request_body.get("temperature", 0.7)),
            max_tokens=sampling.get("max_tokens") or request_body.get("max_tokens"),
            stream=False,
            response_format=response_format,
        )

        model_service_url = os.getenv("MODEL_SERVICE_URL")
        internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
        if model_service_url:
            url = model_service_url.rstrip("/") + "/internal/model/chat"
            headers = {}
            if internal_token:
                headers["X-Internal-Token"] = internal_token
            timeout_s = float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))
            _safe_print(f"🌐 Posting to model service {url} timeout={timeout_s}s job_id={job_id}")
            with httpx.Client(timeout=float(os.getenv("MODEL_SERVICE_TIMEOUT", "60"))) as client:
                resp = client.post(url, json=req_obj.dict(), headers=headers)
                processing_ms = _elapsed_ms(started_ms)
                if resp.status_code != 200:
                    raise RuntimeError(f"Model service error {resp.status_code}: {resp.text}")
                data = resp.json()
                result = {"content": data.get("content")}
                status = "succeeded"
        else:
            from services.chat_runner import run_chat_completion
            from main import model_manager
            resp = asyncio.run(run_chat_completion(model_manager, req_obj, allow_images=False))
            processing_ms = _elapsed_ms(started_ms)
            result = {
                "content": resp.content
            }
            status = "succeeded"
        _safe_print(f"✅ Completed job_id={job_id} status={status} processing_ms={processing_ms}")
    except HTTPException as he:
        processing_ms = _elapsed_ms(started_ms)
        error = {"code": he.detail.get("error", {}).get("type", "llm_error"), "message": str(he.detail)}
        _safe_print(f"❌ Job failed (HTTPException) job_id={job_id} code={error.get('code')} msg={error.get('message')}")
    except Exception as exc:
        processing_ms = _elapsed_ms(started_ms)
        tb = traceback.format_exc()
        error = {"code": "exception", "message": str(exc), "traceback": tb}
        _safe_print(f"❌ Job exception job_id={job_id} error={exc}\n{tb}")

    timing = {"processing_ms": processing_ms}
    return _send_callback(
        callback,
        job_id=job_id,
        job_type=payload.get("job_type"),
        status=status,
        error=error,
        result=result,
        timing=timing,
        trace_id=trace_id,
        metadata=metadata,
    )


def _process_adapter_train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    started_ms = current_timestamp_ms()
    job_id = payload.get("job_id")
    ttl_seconds = payload.get("ttl_seconds") or 0
    created_at = payload.get("created_at")
    callback = payload.get("callback") or {}
    metadata = payload.get("metadata") or {}
    request_body = payload.get("request") or {}
    trace_id = payload.get("trace_id")

    # Expiry check
    now_s = time.time()
    if ttl_seconds and created_at:
        try:
            created_s = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            try:
                created_s = float(created_at)
            except (ValueError, TypeError):
                created_s = now_s
        if created_s + ttl_seconds < now_s:
            return _send_callback(
                callback,
                job_id,
                status="failed",
                error={"code": "expired", "message": "Job expired before processing"},
                result=None,
                timing={"processing_ms": _elapsed_ms(started_ms)},
                trace_id=trace_id,
            )

    status = "failed"
    result = None
    error: Optional[Dict[str, Any]] = None
    processing_ms = 0

    _safe_print(f"📦 Payload: {payload}")
    _safe_print(f"▶️  Starting job_id={job_id} job_type=adapter_train node_id={request_body.get('node_id')}")

    try:
        from services.adapter_training import run_adapter_training

        result = run_adapter_training(request_body, job_id=job_id, ttl_seconds=ttl_seconds)
        processing_ms = _elapsed_ms(started_ms)
        status = "succeeded"
        _safe_print(f"✅ Completed job_id={job_id} status={status} processing_ms={processing_ms}")
    except Exception as exc:
        processing_ms = _elapsed_ms(started_ms)
        tb = traceback.format_exc()
        error = {"code": "exception", "message": str(exc), "traceback": tb}
        _safe_print(f"❌ Job exception job_id={job_id} error={exc}\n{tb}")

    timing = {"processing_ms": processing_ms}
    return _send_callback(
        callback,
        job_id=job_id,
        job_type=payload.get("job_type"),
        status=status,
        error=error,
        result=result,
        timing=timing,
        trace_id=trace_id,
        metadata=metadata,
    )


def _process_vision_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a vision inference job with model swap pattern."""
    started_ms = current_timestamp_ms()
    job_id = payload.get("job_id")
    ttl_seconds = payload.get("ttl_seconds") or 0
    created_at = payload.get("created_at")
    callback = payload.get("callback") or {}
    metadata = payload.get("metadata") or {}
    request_body = payload.get("request") or {}
    trace_id = payload.get("trace_id")

    # Expiry check
    now_s = time.time()
    if ttl_seconds and created_at:
        try:
            created_s = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            try:
                created_s = float(created_at)
            except (ValueError, TypeError):
                created_s = now_s
        if created_s + ttl_seconds < now_s:
            return _send_callback(
                callback,
                job_id=job_id,
                job_type="vision_inference",
                status="failed",
                error={"code": "expired", "message": "Job expired before processing"},
                result=None,
                timing={"processing_ms": _elapsed_ms(started_ms)},
                trace_id=trace_id,
                metadata=metadata,
            )

    status = "failed"
    result = None
    error: Optional[Dict[str, Any]] = None
    processing_ms = 0

    _safe_print(f"📦 Vision Payload: {payload}")
    _safe_print(f"🔭 Starting vision job_id={job_id}")

    try:
        from services.vision_inference import run_vision_inference

        result = run_vision_inference(request_body, job_id=job_id)
        processing_ms = _elapsed_ms(started_ms)
        status = "succeeded"
        _safe_print(f"✅ Vision completed job_id={job_id} status={status} processing_ms={processing_ms}")
    except Exception as exc:
        processing_ms = _elapsed_ms(started_ms)
        tb = traceback.format_exc()
        error = {"code": "exception", "message": str(exc), "traceback": tb}
        _safe_print(f"❌ Vision exception job_id={job_id} error={exc}\n{tb}")

    timing = {"processing_ms": processing_ms}
    return _send_callback(
        callback,
        job_id=job_id,
        job_type=payload.get("job_type"),
        status=status,
        error=error,
        result=result,
        timing=timing,
        trace_id=trace_id,
        metadata=metadata,
    )


def _build_callback_envelope(
    job_id: str,
    job_type: Optional[str],
    status: str,
    result: Optional[Dict[str, Any]],
    error: Optional[Dict[str, Any]],
    timing: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "job_type": job_type,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "result": result,
        "error": error,
        "timing": timing,
        "metadata": metadata,
    }


def _send_callback(
    callback: Dict[str, Any],
    job_id: str,
    job_type: Optional[str],
    status: str,
    error: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    timing: Dict[str, Any],
    trace_id: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    cb_type = (callback.get("auth_type") or "").lower()
    if not cb_type:
        cb_type = "internal"
    url = callback.get("url")
    auth_type = callback.get("auth_type")
    auth_token = callback.get("token")
    headers = {"Content-Type": "application/json"}
    if trace_id:
        headers["X-Trace-Id"] = str(trace_id)

    envelope = _build_callback_envelope(
        job_id=job_id,
        job_type=job_type,
        status=status,
        result=result,
        error=error,
        timing=timing,
        metadata=metadata,
    )

    # Dispatch by callback type
    if cb_type == "internal":
        # Apply bearer auth if provided
        if auth_type == "bearer" and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        # Apply Jarvis app-to-app headers if configured
        app_id = os.getenv("JARVIS_AUTH_APP_ID")
        app_key = os.getenv("JARVIS_AUTH_APP_KEY")
        if app_id and app_key:
            headers["X-Jarvis-App-Id"] = app_id
            headers["X-Jarvis-App-Key"] = app_key
    else:
        _safe_print(f"⚠️  Callback error: unsupported type '{cb_type}' job_id={job_id}")
        envelope["callback_status"] = "error"
        envelope["callback_reason"] = f"unsupported_callback_type:{cb_type}"
        return envelope

    if not url:
        _safe_print(f"⚠️  Callback skipped: missing URL job_id={job_id}")
        envelope["callback_status"] = "skipped"
        envelope["callback_reason"] = "missing_url"
        return envelope

    try:
        _safe_print(f"📬 Posting callback for job_id={job_id} job_type={job_type} to {url}")
        _safe_print(f"📨 Callback payload job_id={job_id}: {envelope}")
        with httpx.Client(timeout=float(os.getenv("LLM_PROXY_CALLBACK_TIMEOUT", "10"))) as client:
            resp = client.post(url, headers=headers, json=envelope)
            envelope["callback_status"] = resp.status_code
            envelope["callback_body"] = resp.text[:500]
            _safe_print(f"📫 Callback response status={resp.status_code} job_id={job_id}")
            return envelope
    except Exception as exc:
        envelope["callback_error"] = str(exc)
        envelope["callback_error_detail"] = getattr(exc, "args", None)
        _safe_print(f"⚠️  Callback error job_id={job_id}: {exc} details={getattr(exc, 'args', None)}")
        return envelope

