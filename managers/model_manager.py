import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn")

from services.settings_helpers import (
    get_int_setting,
    get_setting,
)

# Old alias names that trigger deprecation warnings
_DEPRECATED_ALIASES = {"full", "lightweight", "cloud", "vision"}


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
    initializes the models; subsequent calls return the same instance.
    """

    _instance: Optional["ModelManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
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

        self._initialize_models()

    # ---- Backwards-compat property aliases ----

    @property
    def main_model(self) -> Any:
        return self.live_model

    @main_model.setter
    def main_model(self, value: Any) -> None:
        self.live_model = value

    @property
    def lightweight_model(self) -> Any:
        return self.live_model

    @lightweight_model.setter
    def lightweight_model(self, value: Any) -> None:
        self.live_model = value

    @property
    def cloud_model(self) -> Any:
        return self.background_model

    @property
    def vision_model(self) -> Any:
        return self.background_model

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

        # ---- Create backend instances ----
        logger.info(f"🔁 Initializing live model: {live_model_path} ({live_backend})")
        self.live_model = self._create_backend(
            backend_type=live_backend,
            model_path=live_model_path,
            chat_format=live_chat_format,
            stop_tokens=live_stop_tokens,
            context_window=live_context_window,
            rest_url=live_rest_url,
            model_type="live",
        )

        if should_share:
            self.background_model = self.live_model
            logger.info("✅ Background model sharing live instance")
        else:
            logger.info(f"🔁 Initializing background model: {bg_model_path} ({bg_backend})")
            self.background_model = self._create_backend(
                backend_type=bg_backend,
                model_path=bg_model_path,
                chat_format=bg_chat_format,
                stop_tokens=bg_stop_tokens,
                context_window=bg_context_window,
                rest_url=bg_rest_url,
                model_type="background",
            )

        # Populate model registry
        self._populate_registry()

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
            return {"status": "success", "message": f"Live model swapped to {new_model}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def swap_background_model(
        self,
        new_model: str,
        new_model_backend: str,
        new_model_chat_format: str,
        new_model_context_window: int = None,
    ):
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
            return {"status": "success", "message": f"Background model swapped to {new_model}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Backwards-compat aliases
    def swap_main_model(self, *args, **kwargs):
        return self.swap_live_model(*args, **kwargs)

    def swap_lightweight_model(self, *args, **kwargs):
        return self.swap_background_model(*args, **kwargs)

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
        """Populate the model registry and aliases after models are initialized."""
        live_model_id = get_setting(
            "model.live.name", "JARVIS_LIVE_MODEL_NAME", ""
        ) or get_setting(
            "model.main.name", "JARVIS_MODEL_NAME", "jarvis-text-8b"
        )
        live_backend = get_setting(
            "model.live.backend", "JARVIS_LIVE_MODEL_BACKEND", ""
        ) or get_setting(
            "model.main.backend", "JARVIS_MODEL_BACKEND", "GGUF"
        )
        live_context = get_int_setting(
            "model.live.context_window", "JARVIS_LIVE_MODEL_CONTEXT_WINDOW", 0
        ) or get_int_setting(
            "model.main.context_window", "JARVIS_MODEL_CONTEXT_WINDOW", 4096
        )

        bg_model_id = get_setting(
            "model.background.name", "JARVIS_BACKGROUND_MODEL_NAME", ""
        ) or live_model_id
        bg_backend = get_setting(
            "model.background.backend", "JARVIS_BACKGROUND_MODEL_BACKEND", ""
        ) or live_backend
        bg_context = get_int_setting(
            "model.background.context_window", "JARVIS_BACKGROUND_MODEL_CONTEXT_WINDOW", 0
        ) or live_context

        # Register live model
        if self.live_model:
            self.registry[live_model_id] = ModelConfig(
                model_id=live_model_id,
                backend_type=live_backend.upper(),
                backend_instance=self.live_model,
                supports_images=False,
                context_length=live_context,
            )
            # Canonical alias
            self.aliases["live"] = live_model_id
            # Deprecated aliases → live
            self.aliases["full"] = live_model_id
            self.aliases["lightweight"] = live_model_id
            logger.info(f"📋 Registered live model: {live_model_id} (aliases: 'live', 'full', 'lightweight')")

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
            # Deprecated aliases → background
            self.aliases["cloud"] = bg_model_id
            self.aliases["vision"] = bg_model_id
            logger.info(f"📋 Registered background model: {bg_model_id} (aliases: 'background', 'cloud', 'vision')")
        elif self.background_model:
            # Shared instance — all aliases point to the same model
            self.aliases["background"] = live_model_id
            self.aliases["cloud"] = live_model_id
            self.aliases["vision"] = live_model_id
            logger.info(f"📋 Background aliases point to shared live model: {live_model_id}")

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name (supports both direct model IDs and aliases).

        Args:
            model_name: Either a direct model ID or an alias
                        (live, background, full, lightweight, vision, cloud)

        Returns:
            ModelConfig if found, None otherwise
        """
        lower_name = model_name.lower()

        # Log deprecation warning for old aliases
        if lower_name in _DEPRECATED_ALIASES:
            logger.warning(
                "⚠️  Deprecated model alias '%s' used. "
                "Migrate to 'live' or 'background'.",
                model_name,
            )

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
