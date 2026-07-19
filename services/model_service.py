import asyncio
import atexit
import hmac
import inspect
import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Optional, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from managers.chat_types import GenerationParams
from managers.model_manager import ModelManager
from models.api_models import ChatCompletionRequest, Message
from services.chat_runner import run_chat_completion, normalize_messages
from services.streaming import ClosingStreamingResponse
from services.date_keys import extract_date_keys_fast, build_date_hint_message
from services.date_key_matcher import extract_date_keys as extract_date_keys_regex
from api.settings_routes import router as settings_router
from services.settings_helpers import get_setting

load_dotenv()

# Pin the GPU backend to the DISCRETE GPU before any backend import triggers
# llama.cpp's Vulkan/HIP init (ModelManager() below -> gguf_backend -> llama_cpp).
# On dGPU+iGPU boxes this avoids binding the integrated GPU. Best-effort and
# guarded: a missing module or any failure must never block the model service.
# Respects an operator-set *_VISIBLE_DEVICES.
try:
    from gpu_select import select_discrete_gpu

    select_discrete_gpu()
except Exception:  # noqa: BLE001 - startup must not depend on GPU auto-select
    pass

logger = logging.getLogger("uvicorn")

app = FastAPI()

# Include settings routes
app.include_router(settings_router)


# ============================================================================
# Remote logging setup (jarvis-logs)
# ============================================================================

_jarvis_logger: Optional[logging.Logger] = None


def _setup_remote_logging() -> None:
    """Set up remote logging to jarvis-logs server for forwarded worker logs."""
    global _jarvis_logger
    try:
        from jarvis_log_client import init as init_log_client, JarvisLogHandler

        app_id = os.getenv("JARVIS_APP_ID", "llm-proxy")
        app_key = os.getenv("JARVIS_APP_KEY")
        if not app_key:
            logger.info("JARVIS_APP_KEY not set, remote logging disabled")
            return

        init_log_client(app_id=app_id, app_key=app_key)

        # Create a dedicated logger for forwarded worker logs
        _jarvis_logger = logging.getLogger("llm-proxy-worker")
        _jarvis_logger.setLevel(logging.DEBUG)

        handler = JarvisLogHandler(
            service="llm-proxy-worker",
            level=logging.DEBUG,
        )
        _jarvis_logger.addHandler(handler)
        logger.info("Remote logging enabled for worker log forwarding")
    except ImportError:
        logger.warning("jarvis-log-client not installed, remote logging disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize remote logging: {e}")


# Initialize remote logging on module load
_setup_remote_logging()


# ============================================================================
# Auth helper (defined early for use by all endpoints)
# ============================================================================

def require_internal_token(x_internal_token: str | None):
    expected = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    # Fail closed: if no token is configured, reject every request rather than
    # allowing unauthenticated access to the model service.
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={"error": {"type": "service_unavailable", "message": "Model service token not configured", "code": "token_not_configured"}},
        )
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=401,
            detail={"error": {"type": "unauthorized", "message": "Invalid internal token", "code": "unauthorized"}},
        )


# ============================================================================
# Log forwarding endpoint for workers
# ============================================================================

class LogEntry(BaseModel):
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: str
    logger_name: Optional[str] = None
    timestamp: Optional[str] = None
    extra: Optional[dict] = None


class LogBatch(BaseModel):
    entries: List[LogEntry]


@app.post("/internal/log")
def forward_log(
    entry: LogEntry,
    x_internal_token: str | None = Header(default=None),
):
    """Forward a single log entry from worker to jarvis-logs."""
    require_internal_token(x_internal_token)
    _forward_log_entry(entry)
    return {"status": "ok"}


@app.post("/internal/logs")
def forward_logs_batch(
    batch: LogBatch,
    x_internal_token: str | None = Header(default=None),
):
    """Forward multiple log entries from worker to jarvis-logs."""
    require_internal_token(x_internal_token)
    for entry in batch.entries:
        _forward_log_entry(entry)
    return {"status": "ok", "count": len(batch.entries)}


def _forward_log_entry(entry: LogEntry) -> None:
    """Forward a log entry to jarvis-logs via the remote logger."""
    level = getattr(logging, entry.level.upper(), logging.INFO)

    if _jarvis_logger:
        _jarvis_logger.log(level, entry.message, extra=entry.extra or {})
    else:
        # Fallback to local logging if remote not available
        logger.log(level, f"[worker] {entry.message}")

# auto_load=False keeps the import cheap and crash-proof: the process binds
# the port FIRST, then loads weights in a background thread (startup hook
# below). This is the fix for the 2026-07 incident where a boot-fatal
# background-model load killed the process before uvicorn could bind :7705,
# leaving a healthy-looking container that 500'd every completion for hours.
model_manager = ModelManager(auto_load=False)

# ============================================================================
# Startup: bind first, load models in the background, self-heal failures
# ============================================================================

# Guard against double-startup (TestClient re-enters the lifespan).
_startup_done = False
service_started_monotonic: float | None = None
service_started_epoch: float | None = None

_RETRY_LOOP_INTERVAL_S = 60.0


def _load_models_in_background(manager: ModelManager) -> None:
    """Initial loader. Bound to the manager captured at startup: if a reload
    retires that instance mid-flight, its load_all() no-ops (shutdown flag)
    instead of racing a second full-weight load on the replacement manager."""
    try:
        manager.load_all()
    except Exception:  # noqa: BLE001 — the loader thread must never die silently
        logger.exception("🚨 Unexpected error during initial model load")


def _retry_loop() -> None:
    """Self-healing loop: while any slot is failed, keep calling
    retry_failed_loads(). It self-gates via per-slot cooldown (60s → 600s),
    so this interval just controls how promptly we notice cooldown expiry."""
    while True:
        time.sleep(_RETRY_LOOP_INTERVAL_S)
        try:
            if any(s["status"] == "failed" for s in model_manager.model_states.values()):
                model_manager.retry_failed_loads()
        except Exception:  # noqa: BLE001
            logger.exception("Model retry loop iteration failed")


@app.on_event("startup")
def startup_event():
    global _startup_done, service_started_monotonic, service_started_epoch
    if _startup_done:
        return
    _startup_done = True
    service_started_monotonic = time.monotonic()
    service_started_epoch = time.time()
    # Mark slots loading BEFORE the loader thread spawns so /health never
    # reports "not_loaded" in the gap.
    model_manager.mark_all_loading()
    threading.Thread(
        target=_load_models_in_background,
        args=(model_manager,),
        name="model-loader",
        daemon=True,
    ).start()
    threading.Thread(target=_retry_loop, name="model-retry-loop", daemon=True).start()
    logger.info("🚀 Model service bound; loading models in a background thread")


# ============================================================================
# Slot readiness guard for inference endpoints
# ============================================================================


def _resolve_slot(model_name: str | None) -> str:
    """Map a requested model name/alias to a manager slot for state checks."""
    name = (model_name or "live").lower()
    if name in ("live", "background"):
        return name
    for slot in ("live", "background"):
        if model_manager.aliases.get(slot) == model_name:
            return slot
    return "live"


def _require_loaded_backend(model_name: str | None) -> None:
    """503 (model_not_loaded) when the resolved slot has no backend.

    Also fires an opportunistic, non-blocking retry: retry_failed_loads()
    holds its own lock + cooldown, so the daemon thread can never block or
    stampede — concurrent callers no-op fast.
    """
    slot = _resolve_slot(model_name)
    backend = model_manager.live_model if slot == "live" else model_manager.background_model
    if backend is not None:
        return
    state = dict(model_manager.model_states.get(slot, {}))
    if state.get("status") == "failed":
        threading.Thread(target=model_manager.retry_failed_loads, daemon=True).start()
    status = state.get("status", "unknown")
    error = state.get("error") or "not loaded yet"
    raise HTTPException(
        status_code=503,
        detail={
            "error": {
                "type": "model_not_loaded",
                "message": f"{slot} model is {status}: {error}",
                "code": "model_not_loaded",
                "slot": slot,
                "state": state,
            }
        },
    )


# Safety net: atexit handler runs even if FastAPI shutdown event doesn't fire
# (e.g., when process is killed before uvicorn can run shutdown hooks)
_cleanup_done = False

def _atexit_cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    logger.info("🧹 [atexit] Cleaning up model service...")
    try:
        model_manager.unload_all()
    except Exception as e:
        logger.warning(f"⚠️  [atexit] Cleanup error: {e}")

atexit.register(_atexit_cleanup)


@app.on_event("shutdown")
def shutdown_event():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    logger.info("🧹 Shutting down model service, unloading models...")
    model_manager.unload_all()


# Serialize reloads with each other; coordination with in-flight loads on the
# CURRENT manager happens via begin_shutdown() below.
_reload_lock = threading.Lock()


@app.post("/internal/model/unload")
def unload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    logger.info("🧹 Unloading models (debug pause)...")
    # The GPU is being handed to vision inference / adapter training: the
    # retry machinery must not reload weights into it mid-job (OOM / native
    # crash). Cleared by /internal/model/reload (fresh manager is unpaused).
    model_manager.pause_loads()
    model_manager.unload_all()
    logger.info("✅ Unload complete")
    return {"status": "ok"}


@app.post("/internal/model/reload")
def reload_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    global model_manager
    logger.info("🔄 Reloading models (debug resume)")

    with _reload_lock:
        # Invalidate settings cache before reloading
        try:
            from services.settings_service import get_settings_service
            settings = get_settings_service()
            settings.invalidate_cache()
            logger.info("🔄 Settings cache invalidated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to invalidate settings cache: {e}")

        # Retire the old manager FIRST: begin_shutdown() blocks until any
        # in-flight load on it finishes (the startup loader thread and retry
        # threads hold bound references to the OLD instance) and prevents new
        # ones — otherwise two full-weight loads race in one process (double
        # VRAM → native llama.cpp OOM/abort) and the old manager's weights
        # land orphaned after we swap.
        old_manager = model_manager
        old_manager.begin_shutdown()
        try:
            old_manager.unload_all()
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"unload_all during reload failed: {e}")
        # Reset the singleton so the fresh manager re-reads settings, then load
        # through the fault-isolated machinery. force=True semantics: a reload
        # always attempts every slot immediately (fresh manager → no cooldown).
        # A failed slot no longer raises — it lands in model_states and the
        # retry loop keeps working on it.
        ModelManager._instance = None
        ModelManager._initialized = False
        model_manager = ModelManager(auto_load=False)
        new_manager = model_manager
        new_manager.load_all()

    # ok-iff-loaded: callers (adapter-training resume, vision resume,
    # patch_settings) log success on 200 — a reload onto a broken config must
    # not read as success while the retry loop grinds on a dead live model.
    slots = {slot: dict(state) for slot, state in new_manager.model_states.items()}
    live_ready = new_manager.model_states["live"]["status"] == "ready"
    return JSONResponse(
        status_code=200 if live_ready else 503,
        content={"status": "ok" if live_ready else "degraded", "slots": slots},
    )


def _get_user_text(req: ChatCompletionRequest) -> str:
    """Extract the last user message text for date key extraction."""
    for msg in reversed(req.messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content
            # Handle structured content (text + images)
            if isinstance(content, list):
                for part in content:
                    if hasattr(part, "type") and part.type == "text":
                        return part.text
                    elif isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""


@app.post("/internal/model/chat")
async def model_chat(req: ChatCompletionRequest, x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    _require_loaded_backend(req.model)
    logger.info(f"▶️  /internal/model/chat model={req.model} messages={len(req.messages)}")

    # Extract date keys using deterministic regex matcher (~0ms).
    # Returns keys directly — no confidence thresholds or model hints needed.
    # Falls back to FastText if regex returns empty (belt-and-suspenders).
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            logger.debug(f"🗓️  Extracting date keys from: {user_text!r}")
            date_keys = extract_date_keys_regex(user_text)
            logger.debug(f"🗓️  Regex matcher result: keys={date_keys}")

            # Fallback: if regex found nothing, try FastText + hint
            if not date_keys:
                ft_result = extract_date_keys_fast(user_text)
                if ft_result.keys:
                    date_keys = ft_result.keys
                    logger.debug(f"🗓️  FastText fallback: keys={ft_result.keys}")
                    hint_msg = build_date_hint_message(ft_result)
                    if hint_msg:
                        req.messages.append(Message(**hint_msg))

    result = await run_chat_completion(model_manager, req, allow_images=True)
    preview = (result.content or "")[:1000]
    logger.debug(f"🧾 /internal/model/chat response_preview={preview!r}")
    logger.info(f"✅ /internal/model/chat model={req.model} done")

    response = {"content": result.content, "usage": result.usage}
    if date_keys is not None:
        response["date_keys"] = date_keys
    if result.tool_calls is not None:
        response["tool_calls"] = result.tool_calls
    if result.finish_reason is not None:
        response["finish_reason"] = result.finish_reason
    return response


# ============================================================================
# Streaming: producer-thread pump with abort + cancel
#
# The backend stream generators (gguf, mlx) are SYNC generators that hold the
# backend's threading.Lock across every yield. Before 2026-07 they were handed
# straight to StreamingResponse: an abandoned client left the generator
# suspended forever, the lock held, and llama.cpp grinding into a dead socket
# — one dropped consumer poisoned every later request (and a sync
# non-streaming request blocking the event loop on that lock could deadlock
# the whole service, /health included).
#
# Now a dedicated producer thread iterates the generator and pushes events
# into a bounded queue; the async consumer yields them as SSE. On client
# disconnect or cancel, the abort event stops the producer at the next token
# boundary and gen.close() runs IN THE PRODUCER THREAD, so GeneratorExit
# unwinds the backend's `with self._lock:` deterministically — never left to
# GC. Abort granularity is one token; a long prompt prefill inside
# llama_decode still cannot be interrupted mid-eval.
# ============================================================================

_STREAM_SENTINEL = object()
_STREAM_QUEUE_MAX = 256


class _StreamHandle:
    __slots__ = ("abort", "cancel_requested")

    def __init__(self) -> None:
        self.abort = threading.Event()
        self.cancel_requested = threading.Event()


_active_streams: dict[str, _StreamHandle] = {}
_active_streams_lock = threading.Lock()


def _pump_stream(gen, q: "queue.Queue", handle: _StreamHandle, notify) -> None:
    """Producer: iterate the sync backend generator, honoring abort.

    Runs in its own thread. The finally block closes the generator from this
    same thread (it is suspended at a yield whenever we hold an item), which
    releases any backend lock held across yields. ``notify`` wakes the async
    consumer (loop.call_soon_threadsafe) — the consumer never touches the
    shared thread-pool executor, so blocked generation threads can't starve
    stream consumption into a circular wait.
    """
    abort = handle.abort

    def put_frame(item) -> bool:
        """Deliver a frame with backpressure, bailing only on abort.

        A full queue means the consumer is SLOW, not gone — a dead consumer
        always sets abort via teardown, so waiting on Full is safe and
        terminal frames (backend errors) are never silently dropped on a
        connected-but-slow client.
        """
        while not abort.is_set():
            try:
                q.put(item, timeout=0.25)
                notify()
                return True
            except queue.Full:
                continue
        return False

    try:
        for event in gen:
            if abort.is_set():
                break
            if not put_frame(event):
                break
    except Exception as e:  # noqa: BLE001 — surface backend errors as a frame
        logger.error(f"Streaming error: {e}")
        put_frame({"error": str(e)})
    finally:
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            logger.exception("Closing stream generator failed")
        # Sentinel is best-effort and non-blocking: if it can't be delivered
        # (aborted stream / full queue) the consumer's drained-and-dead
        # fallback ends the stream instead.
        try:
            q.put_nowait(_STREAM_SENTINEL)
        except queue.Full:
            pass
        notify()


@app.post("/internal/model/cancel/{request_id}")
def cancel_stream(
    request_id: str,
    x_internal_token: str | None = Header(default=None),
):
    """Cancel an in-flight streaming generation by request id.

    The id is the X-Request-Id echoed on the stream response (supplied by the
    caller or generated). Cancellation is cooperative: it takes effect at the
    next token boundary and the stream ends with a {"cancelled": true} frame.
    A new stream reusing an active id supersedes (aborts) the stale one, so
    retries with a stable id never dead-end on a tearing-down predecessor.
    """
    require_internal_token(x_internal_token)
    with _active_streams_lock:
        handle = _active_streams.get(request_id)
    if handle is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "not_found",
                    "message": f"No active stream with request id '{request_id}'",
                    "code": "request_not_found",
                }
            },
        )
    handle.cancel_requested.set()
    handle.abort.set()
    return {"status": "cancelling", "request_id": request_id}


@app.post("/internal/model/chat/stream")
async def model_chat_stream(
    req: ChatCompletionRequest,
    x_internal_token: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
):
    """SSE streaming chat completion endpoint.

    Returns Server-Sent Events with token deltas, followed by a final
    event containing the full content and usage stats. NOT OpenAI chunk
    format: frames are {"delta": "tok"} … {"done": true, content, usage,
    tool_calls, finish_reason}, or {"error": "..."} / {"cancelled": true}.

    The response echoes X-Request-Id (caller-supplied or generated); POST
    /internal/model/cancel/{request_id} aborts the generation mid-stream.
    """
    require_internal_token(x_internal_token)
    _require_loaded_backend(req.model)
    logger.info(f"▶️  /internal/model/chat/stream model={req.model} messages={len(req.messages)}")

    # Date key extraction (same as non-streaming)
    date_keys: Optional[List[str]] = None
    if req.include_date_context:
        user_text = _get_user_text(req)
        if user_text:
            date_keys = extract_date_keys_regex(user_text)
            if not date_keys:
                ft_result = extract_date_keys_fast(user_text)
                if ft_result.keys:
                    date_keys = ft_result.keys
                    hint_msg = build_date_hint_message(ft_result)
                    if hint_msg:
                        req.messages.append(Message(**hint_msg))

    model_config = model_manager.get_model_config(req.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not found")

    backend = model_config.backend_instance
    # Capability check must be structural: base.py defines
    # generate_text_chat_stream as a plain raising function, so hasattr is
    # always true — only a real generator function marks streaming support.
    stream_fn = getattr(backend, "generate_text_chat_stream", None)
    if stream_fn is None or not inspect.isgeneratorfunction(
        getattr(stream_fn, "__func__", stream_fn)
    ):
        raise HTTPException(status_code=501, detail="Backend does not support streaming")

    normalized = normalize_messages(req.messages)
    params = GenerationParams(
        temperature=req.temperature or 0.7,
        max_tokens=req.max_tokens,
        stream=True,
        adapter_settings=(
            req.adapter_settings.model_dump()
            if req.adapter_settings and req.adapter_settings.enabled
            else None
        ),
    )

    request_id = x_request_id or uuid.uuid4().hex
    handle = _StreamHandle()
    gen = backend.generate_text_chat_stream(model_config, normalized, params)

    with _active_streams_lock:
        prior = _active_streams.get(request_id)
        if prior is not None:
            # Supersede, don't 409: a caller retrying with a stable id must
            # win over its own stale/tearing-down predecessor.
            prior.cancel_requested.set()
            prior.abort.set()
        _active_streams[request_id] = handle

    def _teardown() -> None:
        """Idempotent; safe from any thread. Guarded pop: a superseding
        stream may own this id by now — never remove someone else's handle."""
        handle.abort.set()
        with _active_streams_lock:
            if _active_streams.get(request_id) is handle:
                del _active_streams[request_id]

    loop = asyncio.get_running_loop()
    data_ready = asyncio.Event()

    def _notify() -> None:
        try:
            loop.call_soon_threadsafe(data_ready.set)
        except RuntimeError:
            pass  # loop closed (shutdown) — consumer is gone anyway

    q: queue.Queue = queue.Queue(maxsize=_STREAM_QUEUE_MAX)
    producer = threading.Thread(
        target=_pump_stream,
        args=(gen, q, handle, _notify),
        name=f"stream-pump-{request_id[:8]}",
        daemon=True,
    )
    try:
        producer.start()
    except Exception:
        _teardown()
        raise

    async def event_stream():
        # Event-driven consumer: woken by call_soon_threadsafe, drains with
        # get_nowait. Deliberately NO asyncio.to_thread here — the default
        # executor can be saturated by threads blocked on the backend lock
        # (chat_runner offloads), and a consumer queued behind them would
        # complete the circular wait this module exists to prevent.
        try:
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    data_ready.clear()
                    try:
                        item = q.get_nowait()  # re-check: lost-wakeup guard
                    except queue.Empty:
                        # Queue fully drained: only end if the producer is
                        # gone too (its terminal frames are always enqueued
                        # before it exits, so nothing can be lost here).
                        if not producer.is_alive():
                            break
                        try:
                            await asyncio.wait_for(data_ready.wait(), timeout=1.0)
                        except asyncio.TimeoutError:
                            pass
                        continue
                if item is _STREAM_SENTINEL:
                    if handle.cancel_requested.is_set():
                        yield f"data: {json.dumps({'cancelled': True})}\n\n"
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            _teardown()

    # on_teardown covers the hole the generator's own finally cannot: a
    # client that disconnects before the first __anext__ acloses a
    # never-started generator, skipping its body entirely.
    return ClosingStreamingResponse(
        event_stream(),
        on_teardown=_teardown,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


@app.get("/health")
async def health():
    """Model service health — unauthenticated, always HTTP 200 while the
    process is alive. The API server (api/health_routes.py) aggregates this
    body into a real HTTP status code for the docker healthcheck.

    Backward-compatible keys: "status" ("ok" only when the live slot is
    ready), "models", "aliases". Added: "slots", "uptime_s", "started_at".
    """
    now = time.monotonic()
    slots: dict[str, dict] = {}
    for slot, state in model_manager.model_states.items():
        view = dict(state)
        last = view.pop("last_attempt_monotonic", None)
        view["last_attempt_seconds_ago"] = (
            round(now - last, 1) if last is not None else None
        )
        slots[slot] = view

    live_status = model_manager.model_states["live"]["status"]
    if live_status == "ready":
        status = "ok"  # keep the historical literal — consumers match on it
    elif live_status == "failed":
        status = "degraded"
    else:
        status = "loading"

    return {
        "status": status,
        "models": list(model_manager.registry.keys()),
        "aliases": model_manager.aliases,
        "slots": slots,
        "uptime_s": (
            round(now - service_started_monotonic, 1)
            if service_started_monotonic is not None
            else None
        ),
        "started_at": service_started_epoch,
    }


@app.get("/internal/model/models")
async def list_models(x_internal_token: str | None = Header(default=None)):
    require_internal_token(x_internal_token)
    data = []
    for model_id, cfg in model_manager.registry.items():
        data.append(
            {
                "id": model_id,
                "backend": cfg.backend_type,
                "supports_images": cfg.supports_images,
                "context_length": cfg.context_length,
            }
        )
    return {"models": data, "aliases": model_manager.aliases}


@app.get("/internal/model/engine")
async def get_engine_info(x_internal_token: str | None = Header(default=None)):
    """Get information about the current inference engine.

    Returns engine type and capabilities like prefix caching support.
    - llama_cpp: Benefits from warmup messages (prefix caching with matched content)
    - vllm: Automatic prefix caching, no benefit from explicit warmup messages
    - transformers: No prefix caching optimization
    """
    require_internal_token(x_internal_token)

    # Get the main model's inference engine
    inference_engine = get_setting(
        "inference.general.engine", "JARVIS_INFERENCE_ENGINE", "llama_cpp"
    ).lower()
    backend_type = get_setting(
        "model.main.backend", "JARVIS_MODEL_BACKEND", "GGUF"
    ).upper()

    # Check if live_model backend instance has inference_engine attribute (more accurate)
    if model_manager.live_model and hasattr(model_manager.live_model, "inference_engine"):
        inference_engine = model_manager.live_model.inference_engine

    # llama.cpp and MLX benefit from warmup messages (prefix caching reuses KV cache for matching prefixes)
    # vLLM has automatic prefix caching but doesn't benefit from explicit warmup messages
    # transformers has no prefix caching optimization
    allows_caching = inference_engine in ("llama_cpp", "mlx")

    return {
        "inference_engine": inference_engine,
        "backend_type": backend_type,
        "allows_caching": allows_caching,
        "description": {
            "llama_cpp": "llama.cpp with prefix caching - benefits from warmup messages",
            "mlx": "MLX with KV cache prefix caching - benefits from warmup messages",
            "vllm": "vLLM with automatic prefix caching - no warmup needed",
            "transformers": "HuggingFace Transformers - no prefix caching",
        }.get(inference_engine, "Unknown engine"),
    }

