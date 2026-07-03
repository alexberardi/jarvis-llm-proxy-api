import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn")

from services.settings_helpers import (
    get_int_setting,
    get_setting,
)

# Retry backoff for failed model loads: 60s doubling per attempt, capped at 600s.
_RETRY_BACKOFF_BASE_S = 60.0
_RETRY_BACKOFF_CAP_S = 600.0


class ModelConfig:
    """Configuration for a single model"""
    def __init__(
        self,
        model_id: str,
        backend_type: str,
        backend_instance: Any,
        supports_images: bool = False,
        context_length: int = 4096,
    ):
        self.model_id = model_id
        self.backend_type = backend_type
        self.backend_instance = backend_instance
        self.supports_images = supports_images
        self.context_length = context_length

class ModelManager:
    """Singleton model manager for all LLM backends.

    Two-model system:
      - **live** — used by the chat endpoint, optimized for speed + accuracy
      - **background** — used by the Redis queue worker, optimized for accuracy (can be slower)

    Routing is implicit: chat endpoint always uses live, queue always uses background.
    If both resolve to the same model path and backend, they share a single instance.

    Use ModelManager() to get the singleton instance. The first call
    initializes the models (unless auto_load=False — then call load_all());
    subsequent calls return the same instance.

    Model loads are per-slot fault-isolated: a failed load records
    model_states[slot] = "failed" instead of raising, and retry_failed_loads()
    re-attempts it with exponential cooldown.
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, auto_load: bool = True):
        """Create (or return) the singleton.

        Args:
            auto_load: when False, models are NOT loaded during construction;
                call load_all() to run the initial load. Singleton caveat: the
                flag only matters on the FIRST construction — subsequent
                ModelManager() calls return the existing instance and ignore it.
        """
        # Only initialize once
        if ModelManager._initialized:
            return
        ModelManager._initialized = True

        self.live_model = None
        self.background_model = None

        # Model registry: maps model_id -> ModelConfig
        self.registry: Dict[str, ModelConfig] = {}

        # Aliases for stable client references
        self.aliases: Dict[str, str] = {}

        # Per-slot load state — drives the model service /health body and the
        # retry machinery. A failed load NEVER raises out of construction.
        self.model_states: dict[str, dict] = {
            "live": self._new_slot_state(),
            "background": self._new_slot_state(),
        }
        # Resolved per-slot configs, frozen by _resolve_slot_configs (see
        # comment there for why retries do not re-read settings).
        self._slot_configs: dict[str, dict] = {}
        self._slots_shared: bool = False
        # ONE load at a time per manager: load_all(), retry_failed_loads() and
        # the swap methods all serialize on this lock so two full-weight loads
        # can never run concurrently in one process (double VRAM → native
        # llama.cpp OOM/abort kills the whole model service).
        self._load_lock = threading.Lock()
        # Loads paused: /internal/model/unload freed the GPU on purpose
        # (vision/training) — retries must not reload weights mid-job.
        self._loads_paused = False
        # Retired: /internal/model/reload swapped in a fresh manager — no new
        # load may start on this instance (in-flight loader/retry threads hold
        # bound references to it).
        self._shutdown = False

        if auto_load:
            self.load_all()

    # ------------------------------------------------------------------ #
    #  Slot state tracking                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _new_slot_state() -> dict:
        return {
            "status": "not_loaded",  # not_loaded | loading | ready | failed
            "error": None,
            "attempts": 0,
            "last_attempt_monotonic": None,
            "loaded_at_epoch": None,
        }

    def _set_slot_state(
        self,
        slot: str,
        status: str,
        error: str | None = None,
        count_attempt: bool = False,
    ) -> None:
        """Single transition helper so every slot-state change stays consistent."""
        state = self.model_states[slot]
        state["status"] = status
        state["error"] = error
        if count_attempt:
            state["attempts"] += 1
            state["last_attempt_monotonic"] = time.monotonic()
        if status == "ready":
            state["loaded_at_epoch"] = time.time()

    def mark_all_loading(self) -> None:
        """Flag every slot as loading (the service calls this before spawning
        the loader thread so /health never reports not_loaded in the gap)."""
        for slot in self.model_states:
            self._set_slot_state(slot, "loading")

    def load_all(self) -> None:
        """Run the initial model load (pairs with auto_load=False).

        Serialized with retry_failed_loads() on _load_lock so the retry loop
        can never run a second full-weight load concurrently with an
        in-flight initial load / reload. An exception that escapes
        _initialize_models OUTSIDE the per-slot fault isolation (e.g. a
        malformed settings value in the resolution preamble) marks every
        still-pending slot "failed" — a slot must never wedge in "loading",
        which the retry machinery ignores.
        """
        with self._load_lock:
            if self._shutdown:
                logger.warning("load_all skipped: manager has been retired (reload)")
                return
            try:
                self._initialize_models()
            except Exception as e:  # noqa: BLE001 — a wedged "loading" slot is unretryable
                logger.exception(
                    f"🚨 Model initialization failed outside per-slot isolation: {e}"
                )
                for slot, state in self.model_states.items():
                    if state["status"] not in ("ready", "failed"):
                        self._set_slot_state(
                            slot,
                            "failed",
                            error=f"{type(e).__name__}: {e}",
                            count_attempt=True,
                        )

    def pause_loads(self) -> None:
        """Stop retry loads: /internal/model/unload freed the GPU on purpose
        (vision inference / adapter training) — a retry reloading weights
        mid-job would OOM or crash llama.cpp natively. Cleared implicitly by
        /internal/model/reload (fresh manager) or resume_loads()."""
        self._loads_paused = True

    def resume_loads(self) -> None:
        self._loads_paused = False

    def begin_shutdown(self) -> None:
        """Retire this manager (called by /internal/model/reload before the
        singleton swap). Blocks until any in-flight load_all/retry on this
        instance finishes; afterwards no new load can start on it. Without
        this, a retry thread minutes deep inside a load (bound to the OLD
        instance) races the reload's fresh load — two concurrent full-weight
        loads, and the old manager's weights land orphaned (never unloaded).
        """
        self._shutdown = True
        with self._load_lock:
            pass

    # ------------------------------------------------------------------ #
    #  Backend factory                                                     #
    # ------------------------------------------------------------------ #

    def _create_backend(
        self,
        backend_type: str,
        model_path: str,
        chat_format: str,
        stop_tokens: str,
        context_window: int,
        rest_url: str,
        model_type: str,
    ) -> Any:
        """Instantiate a backend by type. Handles all supported backends.

        Args:
            backend_type: MOCK, MLX, GGUF, TRANSFORMERS, VLLM, REST
            model_path: Model file path or HuggingFace ID
            chat_format: Chat template format
            stop_tokens: Comma-separated stop tokens
            context_window: Max context window size
            rest_url: REST endpoint URL (only for REST backend)
            model_type: Human label for logging ("live" or "background")
        """
        backend_type = backend_type.upper()

        if backend_type == "MOCK":
            from backends.mock_backend import MockBackend
            return MockBackend(model_path or f"mock-{model_type}")
        elif backend_type == "MLX":
            from backends.mlx_backend import MlxClient
            return MlxClient(model_path)
        elif backend_type == "GGUF":
            from backends.gguf_backend import GGUFClient
            return GGUFClient(model_path, chat_format, stop_tokens, context_window)
        elif backend_type == "TRANSFORMERS":
            from backends.transformers_backend import TransformersClient
            return TransformersClient(model_path, chat_format, stop_tokens, context_window)
        elif backend_type == "VLLM":
            from backends.vllm_backend import VLLMClient
            return VLLMClient(model_path, chat_format, stop_tokens, context_window)
        elif backend_type == "REST":
            from backends.rest_backend import RestClient
            if not rest_url:
                raise ValueError(
                    f"REST URL must be set when using REST backend for {model_type} model"
                )
            return RestClient(rest_url, model_path or "jarvis-llm", model_type)
        else:
            raise ValueError(
                f"Unsupported backend '{backend_type}' for {model_type} model. "
                "Use 'MOCK', 'MLX', 'GGUF', 'TRANSFORMERS', 'VLLM', or 'REST'."
            )

    # ------------------------------------------------------------------ #
    #  Model initialization                                                #
    # ------------------------------------------------------------------ #

    def _initialize_models(self):
        self._resolve_slot_configs()

        # ---- Create backend instances (per-slot fault isolation) ----
        # A failed load records model_states[slot] = failed and leaves the
        # slot None — it must NEVER raise out of here (the 2026-07 incident:
        # a boot-fatal background load killed the process before it could
        # bind the port, leaving a healthy-looking but dead container).
        self._attempt_slot_load("live")

        if self._slots_shared:
            self._mirror_live_to_background()
        else:
            self._attempt_slot_load("background")

        # Populate model registry (tolerant of None slots)
        self._populate_registry()

        # Auto-load date key adapter for successfully loaded slots
        if self.live_model is not None:
            self._load_date_key_adapter(
                self.live_model, self._slot_configs["live"]["model_path"], "live"
            )
        if not self._slots_shared and self.background_model is not None:
            self._load_date_key_adapter(
                self.background_model,
                self._slot_configs["background"]["model_path"],
                "background",
            )

    def _resolve_slot_configs(self) -> None:
        """Read settings and freeze the per-slot configs (+ sharing flag)."""
        # ---- Read live config (with fallback to legacy main settings) ----
        live_backend = get_setting(
            "model.live.backend", "JARVIS_LIVE_MODEL_BACKEND", ""
        )
        if not live_backend:
            live_backend = get_setting(
                "model.main.backend", "JARVIS_MODEL_BACKEND", "GGUF"
            )
        live_backend = live_backend.upper()

        live_model_path = get_setting(
            "model.live.name", "JARVIS_LIVE_MODEL_NAME", ""
        )
        if not live_model_path:
            live_model_path = get_setting(
                "model.main.name", "JARVIS_MODEL_NAME", None
            )

        live_chat_format = get_setting(
            "model.live.chat_format", "JARVIS_LIVE_MODEL_CHAT_FORMAT", ""
        )
        if not live_chat_format:
            live_chat_format = get_setting(
                "model.main.chat_format", "JARVIS_MODEL_CHAT_FORMAT", ""
            )

        live_stop_tokens = get_setting(
            "model.live.stop_tokens", "JARVIS_LIVE_MODEL_STOP_TOKENS", ""
        )
        if not live_stop_tokens:
            live_stop_tokens = get_setting(
                "model.main.stop_tokens", "JARVIS_MODEL_STOP_TOKENS", ""
            )

        live_context_window = get_int_setting(
            "model.live.context_window", "JARVIS_LIVE_MODEL_CONTEXT_WINDOW", 0
        )
        if not live_context_window:
            live_context_window = get_int_setting(
                "model.main.context_window", "JARVIS_MODEL_CONTEXT_WINDOW", 8192
            )

        live_rest_url = get_setting(
            "model.live.rest_url", "JARVIS_LIVE_REST_MODEL_URL", ""
        )
        if not live_rest_url:
            live_rest_url = get_setting(
                "model.main.rest_url", "JARVIS_REST_MODEL_URL", ""
            )

        # ---- Read background config (with fallback to live settings) ----
        bg_backend = get_setting(
            "model.background.backend", "JARVIS_BACKGROUND_MODEL_BACKEND", ""
        )
        if not bg_backend:
            bg_backend = live_backend
        else:
            bg_backend = bg_backend.upper()

        bg_model_path = get_setting(
            "model.background.name", "JARVIS_BACKGROUND_MODEL_NAME", ""
        )
        if not bg_model_path:
            bg_model_path = live_model_path

        bg_chat_format = get_setting(
            "model.background.chat_format", "JARVIS_BACKGROUND_MODEL_CHAT_FORMAT", ""
        )
        if not bg_chat_format:
            bg_chat_format = live_chat_format

        bg_stop_tokens = get_setting(
            "model.background.stop_tokens", "JARVIS_BACKGROUND_MODEL_STOP_TOKENS", ""
        )
        if not bg_stop_tokens:
            bg_stop_tokens = live_stop_tokens

        bg_context_window = get_int_setting(
            "model.background.context_window", "JARVIS_BACKGROUND_MODEL_CONTEXT_WINDOW", 0
        )
        if not bg_context_window:
            bg_context_window = live_context_window

        bg_rest_url = get_setting(
            "model.background.rest_url", "JARVIS_BACKGROUND_REST_MODEL_URL", ""
        )
        if not bg_rest_url:
            bg_rest_url = live_rest_url

        # ---- Sharing logic ----
        should_share = (
            bg_backend == live_backend
            and bg_model_path == live_model_path
        )

        if should_share:
            logger.info("🔄 Memory Optimization: live and background share the same model instance")
            logger.info(f"   → Backend: {live_backend}, Model: {live_model_path}")
            logger.info("   → Memory savings: ~50% (single model instance)")

        # ---- Freeze resolved configs for retries ----
        # Deliberate: retry_failed_loads re-attempts the SAME config that
        # failed rather than re-reading settings mid-flight — a half-applied
        # settings change must not make a background retry silently load a
        # different model than the operator asked for. Config changes go
        # through /internal/model/reload, which rebuilds the manager and
        # re-reads settings.
        self._slot_configs = {
            "live": {
                "backend_type": live_backend,
                "model_path": live_model_path,
                "chat_format": live_chat_format,
                "stop_tokens": live_stop_tokens,
                "context_window": live_context_window,
                "rest_url": live_rest_url,
            },
            "background": {
                "backend_type": bg_backend,
                "model_path": bg_model_path,
                "chat_format": bg_chat_format,
                "stop_tokens": bg_stop_tokens,
                "context_window": bg_context_window,
                "rest_url": bg_rest_url,
            },
        }
        self._slots_shared = should_share

    def _reresolve_configs_preserving_loaded(self) -> None:
        """Re-run settings resolution, keeping the frozen config of any slot
        whose backend is currently loaded — the frozen snapshot must keep
        describing the weights actually in memory (registry/alias labels)."""
        preserved = {
            slot: cfg
            for slot, cfg in self._slot_configs.items()
            if self.model_states.get(slot, {}).get("status") == "ready"
        }
        self._resolve_slot_configs()
        if preserved:
            self._slot_configs.update(preserved)
            # Sharing was computed from the fresh resolution; recompute it
            # against the effective (possibly preserved) configs.
            live_cfg = self._slot_configs.get("live") or {}
            bg_cfg = self._slot_configs.get("background") or {}
            self._slots_shared = (
                live_cfg.get("backend_type") == bg_cfg.get("backend_type")
                and live_cfg.get("model_path") == bg_cfg.get("model_path")
            )

    def _frozen_config_unloadable(self, slot: str) -> bool:
        """True when the frozen config can NEVER produce a backend: missing
        entirely, or missing the one input its backend requires (typically the
        settings DB was unreachable at resolve time and the env fallback was
        empty → model_path froze as None). MOCK defaults its path; REST needs
        rest_url instead of a path."""
        cfg = self._slot_configs.get(slot)
        if cfg is None:
            return True
        backend_type = (cfg.get("backend_type") or "").upper()
        if backend_type == "MOCK":
            return False
        if backend_type == "REST":
            return not cfg.get("rest_url")
        return not cfg.get("model_path")

    # ------------------------------------------------------------------ #
    #  Per-slot loading + retry                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _slot_attr(slot: str) -> str:
        return "live_model" if slot == "live" else "background_model"

    def _attempt_slot_load(self, slot: str) -> bool:
        """Load one slot's backend from its frozen config. Never raises on a
        backend/model failure — records the failure in model_states instead."""
        cfg = self._slot_configs.get(slot)
        if cfg is None:
            self._set_slot_state(slot, "failed", error="no resolved config for slot")
            return False
        self._set_slot_state(slot, "loading", count_attempt=True)
        logger.info(f"🔁 Initializing {slot} model: {cfg['model_path']} ({cfg['backend_type']})")
        try:
            backend = self._create_backend(
                backend_type=cfg["backend_type"],
                model_path=cfg["model_path"],
                chat_format=cfg["chat_format"],
                stop_tokens=cfg["stop_tokens"],
                context_window=cfg["context_window"],
                rest_url=cfg["rest_url"],
                model_type=slot,
            )
        except Exception as e:  # noqa: BLE001 — model-load failure must not kill the process
            logger.exception(
                f"🚨🚨 MODEL LOAD FAILED for '{slot}' slot "
                f"({cfg['backend_type']} {cfg['model_path']}) — continuing WITHOUT it; "
                f"requests for this slot will 503 (model_not_loaded) until a retry succeeds: {e}"
            )
            setattr(self, self._slot_attr(slot), None)
            self._set_slot_state(slot, "failed", error=f"{type(e).__name__}: {e}")
            return False
        setattr(self, self._slot_attr(slot), backend)
        # Register BEFORE flipping to "ready": a slot observed ready must be
        # addressable via get_model_config, or /internal/model/chat 404s
        # ("Model not found" — treated as non-retryable by callers, unlike the
        # 503 model_not_loaded contract) while a sibling slot is still loading.
        self._populate_registry()
        self._set_slot_state(slot, "ready")
        logger.info(f"✅ {slot} model ready: {cfg['model_path']} ({cfg['backend_type']})")
        return True

    def _mirror_live_to_background(self) -> None:
        """Shared-instance case (bg config == live config): background mirrors
        live's outcome — instance AND state — without a second load attempt."""
        self.background_model = self.live_model
        self.model_states["background"] = dict(self.model_states["live"])
        if self.model_states["live"]["status"] == "ready":
            logger.info("✅ Background model sharing live instance")

    def _cooldown_remaining(self, slot: str, now: float) -> float:
        """Seconds until this slot may be retried (exponential 60s → 600s cap)."""
        state = self.model_states[slot]
        last = state["last_attempt_monotonic"]
        if last is None:
            return 0.0
        attempts = max(state["attempts"], 1)
        delay = min(_RETRY_BACKOFF_BASE_S * (2 ** (attempts - 1)), _RETRY_BACKOFF_CAP_S)
        return max(0.0, delay - (now - last))

    def retry_failed_loads(self, force: bool = False) -> bool:
        """Re-attempt loading ONLY failed slots.

        Concurrency-safe: serialized with load_all() on _load_lock; while ANY
        load is in flight, other callers no-op fast (return False). No-ops
        while paused (vision/training unloaded the GPU on purpose) or after
        begin_shutdown() (manager retired by /internal/model/reload). Each
        slot has an exponential cooldown (60s doubling to a 600s cap) unless
        force=True.

        Returns True if anything was (re)loaded.
        """
        if self._shutdown or self._loads_paused:
            return False
        if not self._load_lock.acquire(blocking=False):
            return False
        try:
            reloaded = False
            for slot in ("live", "background"):
                if self._shutdown or self._loads_paused:
                    break
                if slot == "background" and self._slots_shared:
                    continue  # mirrored from live below
                if self.model_states[slot]["status"] != "failed":
                    continue
                if not force and self._cooldown_remaining(slot, time.monotonic()) > 0:
                    continue
                if self._frozen_config_unloadable(slot):
                    # The frozen snapshot can never load — replaying it
                    # verbatim would fail forever with no operator recourse
                    # short of a manual reload. Re-resolve settings first
                    # (valid-but-failing configs still retry frozen — see the
                    # freeze comment in _resolve_slot_configs).
                    try:
                        self._reresolve_configs_preserving_loaded()
                    except Exception as e:  # noqa: BLE001 — settings DB may still be down
                        self._set_slot_state(
                            slot,
                            "failed",
                            error=f"settings re-resolution failed: {type(e).__name__}: {e}",
                            count_attempt=True,
                        )
                        continue
                if self._attempt_slot_load(slot):
                    reloaded = True
                    cfg = self._slot_configs.get(slot, {})
                    self._load_date_key_adapter(
                        getattr(self, self._slot_attr(slot)), cfg.get("model_path", ""), slot
                    )
            if reloaded:
                if self._slots_shared:
                    self._mirror_live_to_background()
                self._populate_registry()
            return reloaded
        finally:
            self._load_lock.release()

    # ------------------------------------------------------------------ #
    #  Date key adapter                                                    #
    # ------------------------------------------------------------------ #

    def _load_date_key_adapter(self, backend: Any, model_path: str, model_type: str) -> None:
        """Try to auto-load a date key adapter for the given backend."""
        if backend is None:
            return
        try:
            from services.date_key_adapter import try_load_adapter
            if try_load_adapter(backend, model_path):
                logger.info(f"📅 Date key adapter active for {model_type} model")
            else:
                logger.debug(f"No date key adapter loaded for {model_type} model")
        except Exception as e:
            logger.warning(f"Date key adapter load failed for {model_type}: {e}")

    # ------------------------------------------------------------------ #
    #  Swap methods                                                        #
    # ------------------------------------------------------------------ #

    def swap_live_model(
        self,
        new_model: str,
        new_model_backend: str,
        new_model_chat_format: str,
        new_model_context_window: int = None,
    ):
        # Serialized on _load_lock: a swap racing an in-flight retry/load
        # would run two full-weight loads concurrently (double VRAM).
        with self._load_lock:
            try:
                if hasattr(self.live_model, 'unload'):
                    self.live_model.unload()

                rest_url = ""
                if new_model_backend.upper() == "REST":
                    rest_url = get_setting(
                        "model.live.rest_url", "JARVIS_REST_MODEL_URL", ""
                    )

                self.live_model = self._create_backend(
                    backend_type=new_model_backend,
                    model_path=new_model,
                    chat_format=new_model_chat_format,
                    stop_tokens="",
                    context_window=new_model_context_window or 8192,
                    rest_url=rest_url,
                    model_type="live",
                )
                # Keep the frozen snapshot honest — retries and registry
                # populates must describe the weights actually loaded.
                self._slot_configs["live"] = {
                    "backend_type": new_model_backend.upper(),
                    "model_path": new_model,
                    "chat_format": new_model_chat_format,
                    "stop_tokens": "",
                    "context_window": new_model_context_window or 8192,
                    "rest_url": rest_url,
                }
                self._slots_shared = self.background_model is self.live_model
                # Keep slot state honest — a swap onto a previously failed slot
                # must clear the failed status so /health recovers.
                self._set_slot_state("live", "ready", count_attempt=True)
                return {"status": "success", "message": f"Live model swapped to {new_model}"}
            except Exception as e:
                if self.live_model is None:
                    self._set_slot_state("live", "failed", error=f"{type(e).__name__}: {e}")
                return {"status": "error", "message": str(e)}

    def swap_background_model(
        self,
        new_model: str,
        new_model_backend: str,
        new_model_chat_format: str,
        new_model_context_window: int = None,
    ):
        # Serialized on _load_lock (see swap_live_model).
        with self._load_lock:
            try:
                if hasattr(self.background_model, 'unload'):
                    self.background_model.unload()

                rest_url = ""
                if new_model_backend.upper() == "REST":
                    rest_url = get_setting(
                        "model.background.rest_url", "JARVIS_BACKGROUND_REST_MODEL_URL", ""
                    )

                self.background_model = self._create_backend(
                    backend_type=new_model_backend,
                    model_path=new_model,
                    chat_format=new_model_chat_format,
                    stop_tokens="",
                    context_window=new_model_context_window or 8192,
                    rest_url=rest_url,
                    model_type="background",
                )
                self._slot_configs["background"] = {
                    "backend_type": new_model_backend.upper(),
                    "model_path": new_model,
                    "chat_format": new_model_chat_format,
                    "stop_tokens": "",
                    "context_window": new_model_context_window or 8192,
                    "rest_url": rest_url,
                }
                self._slots_shared = self.background_model is self.live_model
                self._set_slot_state("background", "ready", count_attempt=True)
                return {"status": "success", "message": f"Background model swapped to {new_model}"}
            except Exception as e:
                if self.background_model is None:
                    self._set_slot_state("background", "failed", error=f"{type(e).__name__}: {e}")
                return {"status": "error", "message": str(e)}

    def unload_all(self):
        """Unload all model backends (best-effort)."""
        import asyncio
        import inspect

        seen = set()
        for name in ("live_model", "background_model"):
            model = getattr(self, name, None)
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            if hasattr(model, "unload"):
                try:
                    result = model.unload()
                    # Handle async unload methods (e.g., RestClient)
                    if inspect.iscoroutine(result):
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.ensure_future(result)
                                logger.warning(f"⚠️  Async unload for {name} scheduled (may complete later)")
                            else:
                                loop.run_until_complete(result)
                        except RuntimeError:
                            asyncio.run(result)
                    logger.info(f"🔄 Unloaded model: {name}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to unload {name}: {e}")

    # ------------------------------------------------------------------ #
    #  Registry                                                            #
    # ------------------------------------------------------------------ #

    def _populate_registry(self):
        """Populate the model registry and aliases after models are initialized.

        Reads ONLY the frozen _slot_configs — never live settings. The
        registry must describe the weights that are actually loaded;
        re-reading settings here would let a mid-flight settings change
        relabel the old weights with a new model id / context window (the
        exact half-applied-settings hazard the freeze in
        _resolve_slot_configs defends against).
        """
        live_cfg = self._slot_configs.get("live") or {}
        bg_cfg = self._slot_configs.get("background") or {}

        live_model_id = live_cfg.get("model_path") or "jarvis-text-8b"
        live_backend = live_cfg.get("backend_type") or "GGUF"
        live_context = live_cfg.get("context_window") or 4096

        bg_model_id = bg_cfg.get("model_path") or live_model_id
        bg_backend = bg_cfg.get("backend_type") or live_backend
        bg_context = bg_cfg.get("context_window") or live_context

        # Register live model
        if self.live_model:
            self.registry[live_model_id] = ModelConfig(
                model_id=live_model_id,
                backend_type=live_backend.upper(),
                backend_instance=self.live_model,
                supports_images=False,
                context_length=live_context,
            )
            self.aliases["live"] = live_model_id
            logger.info(f"📋 Registered live model: {live_model_id} (alias: 'live')")

        # Register background model
        if self.background_model and self.background_model is not self.live_model:
            self.registry[bg_model_id] = ModelConfig(
                model_id=bg_model_id,
                backend_type=bg_backend.upper(),
                backend_instance=self.background_model,
                supports_images=False,
                context_length=bg_context,
            )
            self.aliases["background"] = bg_model_id
            logger.info(f"📋 Registered background model: {bg_model_id} (alias: 'background')")
        elif self.background_model:
            self.aliases["background"] = live_model_id
            logger.info(f"📋 Background alias points to shared live model: {live_model_id}")

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name (supports both direct model IDs and aliases).

        Args:
            model_name: Either a direct model ID or an alias ('live' or 'background')

        Returns:
            ModelConfig if found, None otherwise
        """
        lower_name = model_name.lower()

        # First check if it's an alias
        if lower_name in self.aliases:
            resolved_id = self.aliases[lower_name]
            return self.registry.get(resolved_id)

        # Otherwise treat it as a direct model ID
        return self.registry.get(model_name)

    def list_models(self) -> list:
        """List all available models in the registry"""
        models = []
        for model_id, config in self.registry.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "jarvis",
                "supports_images": config.supports_images,
                "context_length": config.context_length,
            })
        return models
