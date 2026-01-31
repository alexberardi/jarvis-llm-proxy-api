"""Settings service with caching and environment variable fallback.

This service provides runtime configuration that can be modified without
restarting the application. Settings are stored in the database with
fallback to environment variables for backward compatibility.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger("uvicorn")


@dataclass
class SettingValue:
    """A cached setting value with metadata."""

    value: Any
    value_type: str
    requires_reload: bool
    is_secret: bool
    env_fallback: str | None
    from_db: bool  # True if from database, False if from env fallback
    cached_at: float


@dataclass
class SettingDefinition:
    """Definition for a setting with its default value and metadata."""

    key: str
    category: str
    value_type: str  # string, int, float, bool, json
    env_fallback: str | None
    default: Any
    description: str
    requires_reload: bool = False
    is_secret: bool = False


# All settings definitions with their categories, types, and env fallbacks
SETTINGS_DEFINITIONS: list[SettingDefinition] = [
    # ==================== model.main ====================
    SettingDefinition(
        key="model.main.name",
        category="model.main",
        value_type="string",
        env_fallback="JARVIS_MODEL_NAME",
        default=".models/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf",
        description="Main model path or HuggingFace ID",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.main.backend",
        category="model.main",
        value_type="string",
        env_fallback="JARVIS_MODEL_BACKEND",
        default="GGUF",
        description="Main model backend: GGUF, VLLM, TRANSFORMERS, REST, MOCK",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.main.chat_format",
        category="model.main",
        value_type="string",
        env_fallback="JARVIS_MODEL_CHAT_FORMAT",
        default="llama3",
        description="Chat template format: llama3, chatml, mistral, etc.",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.main.context_window",
        category="model.main",
        value_type="int",
        env_fallback="JARVIS_MODEL_CONTEXT_WINDOW",
        default=8192,
        description="Maximum context window size in tokens",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.main.stop_tokens",
        category="model.main",
        value_type="string",
        env_fallback="JARVIS_MODEL_STOP_TOKENS",
        default="",
        description="Comma-separated stop tokens",
        requires_reload=True,
    ),
    # ==================== model.lightweight ====================
    SettingDefinition(
        key="model.lightweight.name",
        category="model.lightweight",
        value_type="string",
        env_fallback="JARVIS_LIGHTWEIGHT_MODEL_NAME",
        default="",
        description="Lightweight model path (empty to share main model)",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.lightweight.backend",
        category="model.lightweight",
        value_type="string",
        env_fallback="JARVIS_LIGHTWEIGHT_MODEL_BACKEND",
        default="",
        description="Lightweight model backend",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.lightweight.chat_format",
        category="model.lightweight",
        value_type="string",
        env_fallback="JARVIS_LIGHTWEIGHT_MODEL_CHAT_FORMAT",
        default="",
        description="Lightweight model chat format",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.lightweight.context_window",
        category="model.lightweight",
        value_type="int",
        env_fallback="JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW",
        default=8192,
        description="Lightweight model context window",
        requires_reload=True,
    ),
    # ==================== model.vision ====================
    SettingDefinition(
        key="model.vision.name",
        category="model.vision",
        value_type="string",
        env_fallback="JARVIS_VISION_MODEL_NAME",
        default="",
        description="Vision model path or HuggingFace ID",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.vision.backend",
        category="model.vision",
        value_type="string",
        env_fallback="JARVIS_VISION_MODEL_BACKEND",
        default="",
        description="Vision model backend",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.vision.context_window",
        category="model.vision",
        value_type="int",
        env_fallback="JARVIS_VISION_MODEL_CONTEXT_WINDOW",
        default=131072,
        description="Vision model context window",
        requires_reload=True,
    ),
    # ==================== model.cloud ====================
    SettingDefinition(
        key="model.cloud.name",
        category="model.cloud",
        value_type="string",
        env_fallback="JARVIS_CLOUD_MODEL_NAME",
        default="",
        description="Cloud model identifier",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.cloud.backend",
        category="model.cloud",
        value_type="string",
        env_fallback="JARVIS_CLOUD_MODEL_BACKEND",
        default="REST",
        description="Cloud model backend (typically REST)",
        requires_reload=True,
    ),
    SettingDefinition(
        key="model.cloud.context_window",
        category="model.cloud",
        value_type="int",
        env_fallback="JARVIS_CLOUD_MODEL_CONTEXT_WINDOW",
        default=4096,
        description="Cloud model context window",
        requires_reload=True,
    ),
    # ==================== inference.vllm ====================
    SettingDefinition(
        key="inference.vllm.gpu_memory_utilization",
        category="inference.vllm",
        value_type="float",
        env_fallback="JARVIS_VLLM_GPU_MEMORY_UTILIZATION",
        default=0.9,
        description="GPU memory utilization (0.0-1.0)",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.tensor_parallel_size",
        category="inference.vllm",
        value_type="int",
        env_fallback="JARVIS_VLLM_TENSOR_PARALLEL_SIZE",
        default=1,
        description="Number of GPUs for tensor parallelism",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.max_batched_tokens",
        category="inference.vllm",
        value_type="int",
        env_fallback="JARVIS_VLLM_MAX_BATCHED_TOKENS",
        default=8192,
        description="Maximum batched tokens for vLLM",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.max_num_seqs",
        category="inference.vllm",
        value_type="int",
        env_fallback="JARVIS_VLLM_MAX_NUM_SEQS",
        default=256,
        description="Maximum number of sequences",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.quantization",
        category="inference.vllm",
        value_type="string",
        env_fallback="JARVIS_VLLM_QUANTIZATION",
        default="",
        description="vLLM quantization: awq, gptq, fp8, or empty",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.max_lora_rank",
        category="inference.vllm",
        value_type="int",
        env_fallback="JARVIS_VLLM_MAX_LORA_RANK",
        default=64,
        description="Maximum LoRA rank for adapters",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.vllm.max_loras",
        category="inference.vllm",
        value_type="int",
        env_fallback="JARVIS_VLLM_MAX_LORAS",
        default=1,
        description="Maximum concurrent LoRA adapters",
        requires_reload=True,
    ),
    # ==================== inference.gguf ====================
    SettingDefinition(
        key="inference.gguf.n_gpu_layers",
        category="inference.gguf",
        value_type="int",
        env_fallback="JARVIS_N_GPU_LAYERS",
        default=-1,
        description="GPU layers (-1=all, 0=CPU only)",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.n_batch",
        category="inference.gguf",
        value_type="int",
        env_fallback="JARVIS_N_BATCH",
        default=512,
        description="Batch size for llama.cpp",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.n_ubatch",
        category="inference.gguf",
        value_type="int",
        env_fallback="JARVIS_N_UBATCH",
        default=512,
        description="Micro-batch size",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.n_threads",
        category="inference.gguf",
        value_type="int",
        env_fallback="JARVIS_N_THREADS",
        default=10,
        description="Number of CPU threads",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.flash_attn",
        category="inference.gguf",
        value_type="bool",
        env_fallback="JARVIS_FLASH_ATTN",
        default=True,
        description="Enable flash attention",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.f16_kv",
        category="inference.gguf",
        value_type="bool",
        env_fallback="JARVIS_F16_KV",
        default=True,
        description="Use FP16 for KV cache",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.gguf.mul_mat_q",
        category="inference.gguf",
        value_type="bool",
        env_fallback="JARVIS_MUL_MAT_Q",
        default=True,
        description="Enable quantized matrix multiplication",
        requires_reload=True,
    ),
    # ==================== inference.transformers ====================
    SettingDefinition(
        key="inference.transformers.device",
        category="inference.transformers",
        value_type="string",
        env_fallback="JARVIS_DEVICE",
        default="auto",
        description="Device: auto, cuda, mps, cpu",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.transformers.torch_dtype",
        category="inference.transformers",
        value_type="string",
        env_fallback="JARVIS_TORCH_DTYPE",
        default="auto",
        description="Torch dtype: auto, float16, float32, bfloat16",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.transformers.use_quantization",
        category="inference.transformers",
        value_type="bool",
        env_fallback="JARVIS_USE_QUANTIZATION",
        default=False,
        description="Enable bitsandbytes quantization",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.transformers.quantization_type",
        category="inference.transformers",
        value_type="string",
        env_fallback="JARVIS_QUANTIZATION_TYPE",
        default="4bit",
        description="Quantization type: 4bit, 8bit",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.transformers.device_map",
        category="inference.transformers",
        value_type="string",
        env_fallback="JARVIS_TRANSFORMERS_DEVICE_MAP",
        default="auto",
        description="Device map for transformers: auto, none",
        requires_reload=True,
    ),
    # ==================== inference.general ====================
    SettingDefinition(
        key="inference.general.engine",
        category="inference.general",
        value_type="string",
        env_fallback="JARVIS_INFERENCE_ENGINE",
        default="llama_cpp",
        description="Default inference engine: llama_cpp, vllm, transformers",
        requires_reload=True,
    ),
    SettingDefinition(
        key="inference.general.max_tokens",
        category="inference.general",
        value_type="int",
        env_fallback="JARVIS_MAX_TOKENS",
        default=7000,
        description="Default max generation tokens",
        requires_reload=False,
    ),
    SettingDefinition(
        key="inference.general.top_p",
        category="inference.general",
        value_type="float",
        env_fallback="JARVIS_TOP_P",
        default=0.95,
        description="Top-P sampling value",
        requires_reload=False,
    ),
    SettingDefinition(
        key="inference.general.top_k",
        category="inference.general",
        value_type="int",
        env_fallback="JARVIS_TOP_K",
        default=40,
        description="Top-K sampling value",
        requires_reload=False,
    ),
    SettingDefinition(
        key="inference.general.repeat_penalty",
        category="inference.general",
        value_type="float",
        env_fallback="JARVIS_REPEAT_PENALTY",
        default=1.1,
        description="Repetition penalty",
        requires_reload=False,
    ),
    # ==================== training ====================
    SettingDefinition(
        key="training.adapter_dir",
        category="training",
        value_type="string",
        env_fallback="LLM_PROXY_ADAPTER_DIR",
        default="/tmp/jarvis-adapters",
        description="Local adapter storage directory",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.batch_size",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_BATCH_SIZE",
        default=1,
        description="Training batch size",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.grad_accum",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_GRAD_ACCUM",
        default=4,
        description="Gradient accumulation steps",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.epochs",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_EPOCHS",
        default=1,
        description="Training epochs",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.learning_rate",
        category="training",
        value_type="float",
        env_fallback="JARVIS_ADAPTER_LEARNING_RATE",
        default=2e-4,
        description="Training learning rate",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.lora_r",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_LORA_R",
        default=16,
        description="LoRA rank",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.lora_alpha",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_LORA_ALPHA",
        default=32,
        description="LoRA alpha scaling",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.lora_dropout",
        category="training",
        value_type="float",
        env_fallback="JARVIS_ADAPTER_LORA_DROPOUT",
        default=0.05,
        description="LoRA dropout rate",
        requires_reload=False,
    ),
    SettingDefinition(
        key="training.max_seq_len",
        category="training",
        value_type="int",
        env_fallback="JARVIS_ADAPTER_MAX_SEQ_LEN",
        default=2048,
        description="Maximum sequence length for training",
        requires_reload=False,
    ),
    # ==================== storage ====================
    SettingDefinition(
        key="storage.s3_endpoint_url",
        category="storage",
        value_type="string",
        env_fallback="S3_ENDPOINT_URL",
        default="",
        description="S3 endpoint URL (for MinIO)",
        requires_reload=False,
    ),
    SettingDefinition(
        key="storage.s3_region",
        category="storage",
        value_type="string",
        env_fallback="S3_REGION",
        default="us-east-1",
        description="S3 region",
        requires_reload=False,
    ),
    SettingDefinition(
        key="storage.adapter_bucket",
        category="storage",
        value_type="string",
        env_fallback="LLM_PROXY_ADAPTER_BUCKET",
        default="jarvis-llm-proxy",
        description="S3 bucket for adapters",
        requires_reload=False,
    ),
    SettingDefinition(
        key="storage.adapter_prefix",
        category="storage",
        value_type="string",
        env_fallback="LLM_PROXY_ADAPTER_PREFIX",
        default="adapters",
        description="S3 prefix for adapters",
        requires_reload=False,
    ),
    # ==================== logging ====================
    SettingDefinition(
        key="logging.console_level",
        category="logging",
        value_type="string",
        env_fallback="JARVIS_LOG_CONSOLE_LEVEL",
        default="WARNING",
        description="Console log level",
        requires_reload=False,
    ),
    SettingDefinition(
        key="logging.remote_level",
        category="logging",
        value_type="string",
        env_fallback="JARVIS_LOG_REMOTE_LEVEL",
        default="DEBUG",
        description="Remote (jarvis-logs) log level",
        requires_reload=False,
    ),
]


def _coerce_value(raw: str | None, value_type: str, default: Any) -> Any:
    """Coerce a string value to the appropriate type."""
    if raw is None or raw == "":
        return default

    try:
        if value_type == "string":
            return raw
        elif value_type == "int":
            return int(raw)
        elif value_type == "float":
            return float(raw)
        elif value_type == "bool":
            return raw.lower() in ("true", "1", "yes", "on")
        elif value_type == "json":
            return json.loads(raw)
        else:
            return raw
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to coerce value '{raw}' to {value_type}: {e}")
        return default


def _serialize_value(value: Any, value_type: str) -> str | None:
    """Serialize a value to string for database storage."""
    if value is None:
        return None

    if value_type == "json":
        return json.dumps(value)
    elif value_type == "bool":
        return "true" if value else "false"
    else:
        return str(value)


class SettingsService:
    """Singleton settings service with caching and env fallback.

    Usage:
        from services.settings_service import get_settings_service

        settings = get_settings_service()
        model_name = settings.get("model.main.name")
        settings.set("model.main.name", "new_model_path")
    """

    CACHE_TTL_SECONDS = 60

    _instance: "SettingsService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "SettingsService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._cache: dict[str, SettingValue] = {}
        self._cache_lock = threading.Lock()
        self._definitions: dict[str, SettingDefinition] = {
            d.key: d for d in SETTINGS_DEFINITIONS
        }
        self._initialized = True
        logger.info(f"SettingsService initialized with {len(self._definitions)} settings definitions")

    def _get_db_session(self) -> Session:
        """Get a database session."""
        from db.session import SessionLocal
        return SessionLocal()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cached value is still valid."""
        if key not in self._cache:
            return False
        cached = self._cache[key]
        return (time.time() - cached.cached_at) < self.CACHE_TTL_SECONDS

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value with caching and env fallback.

        Order of precedence:
        1. Cached value (if not expired)
        2. Database value
        3. Environment variable fallback
        4. Definition default
        5. Provided default
        """
        # 1. Check cache
        with self._cache_lock:
            if self._is_cache_valid(key):
                return self._cache[key].value

        # Get definition for this key
        definition = self._definitions.get(key)
        if definition is None:
            logger.debug(f"Unknown setting key: {key}")
            return default

        # 2. Query database
        db_value = None
        from_db = False
        try:
            from db.models import Setting
            db = self._get_db_session()
            try:
                setting = db.query(Setting).filter(Setting.key == key).first()
                if setting:
                    db_value = _coerce_value(setting.value, setting.value_type, definition.default)
                    from_db = True
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Database unavailable for setting {key}: {e}")

        # 3. Fallback to env var
        if db_value is None and definition.env_fallback:
            env_value = os.getenv(definition.env_fallback)
            if env_value is not None:
                db_value = _coerce_value(env_value, definition.value_type, definition.default)

        # 4. Use definition default
        if db_value is None:
            db_value = definition.default

        # Cache the result
        with self._cache_lock:
            self._cache[key] = SettingValue(
                value=db_value,
                value_type=definition.value_type,
                requires_reload=definition.requires_reload,
                is_secret=definition.is_secret,
                env_fallback=definition.env_fallback,
                from_db=from_db,
                cached_at=time.time(),
            )

        return db_value

    def get_typed(self, key: str, value_type: type, default: Any = None) -> Any:
        """Get a setting with explicit type checking."""
        value = self.get(key, default)
        if not isinstance(value, value_type):
            return default
        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer setting."""
        return self.get_typed(key, int, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float setting."""
        value = self.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean setting."""
        return self.get_typed(key, bool, default)

    def get_str(self, key: str, default: str = "") -> str:
        """Get a string setting."""
        return self.get_typed(key, str, default)

    def set(self, key: str, value: Any) -> bool:
        """Set a setting value in the database.

        Returns True if successful, False otherwise.
        """
        definition = self._definitions.get(key)
        if definition is None:
            logger.warning(f"Cannot set unknown setting key: {key}")
            return False

        try:
            from db.models import Setting
            db = self._get_db_session()
            try:
                setting = db.query(Setting).filter(Setting.key == key).first()
                serialized = _serialize_value(value, definition.value_type)

                if setting:
                    setting.value = serialized
                    setting.updated_at = datetime.utcnow()
                else:
                    setting = Setting(
                        key=key,
                        value=serialized,
                        value_type=definition.value_type,
                        category=definition.category,
                        description=definition.description,
                        requires_reload=definition.requires_reload,
                        is_secret=definition.is_secret,
                        env_fallback=definition.env_fallback,
                    )
                    db.add(setting)

                db.commit()

                # Invalidate cache for this key
                with self._cache_lock:
                    if key in self._cache:
                        del self._cache[key]

                logger.info(f"Setting updated: {key}")
                return True
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to set setting {key}: {e}")
            return False

    def set_bulk(self, settings: dict[str, Any]) -> dict[str, bool]:
        """Set multiple settings at once.

        Returns a dict mapping keys to success status.
        """
        results = {}
        for key, value in settings.items():
            results[key] = self.set(key, value)
        return results

    def invalidate_cache(self, key: str | None = None) -> None:
        """Invalidate cache for a specific key or all keys."""
        with self._cache_lock:
            if key:
                if key in self._cache:
                    del self._cache[key]
            else:
                self._cache.clear()
        logger.info(f"Cache invalidated: {key or 'all'}")

    def list_all(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all settings with their current values.

        If category is provided, filters to that category.
        """
        result = []

        # Get all settings from DB
        db_settings: dict[str, Any] = {}
        try:
            from db.models import Setting
            db = self._get_db_session()
            try:
                query = db.query(Setting)
                if category:
                    query = query.filter(Setting.category == category)
                for setting in query.all():
                    db_settings[setting.key] = {
                        "value": setting.value,
                        "value_type": setting.value_type,
                        "from_db": True,
                    }
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Database unavailable for listing settings: {e}")

        # Merge with definitions
        for key, definition in self._definitions.items():
            if category and definition.category != category:
                continue

            # Determine current value
            if key in db_settings:
                raw_value = db_settings[key]["value"]
                current_value = _coerce_value(raw_value, definition.value_type, definition.default)
                from_db = True
            elif definition.env_fallback:
                env_value = os.getenv(definition.env_fallback)
                if env_value is not None:
                    current_value = _coerce_value(env_value, definition.value_type, definition.default)
                    from_db = False
                else:
                    current_value = definition.default
                    from_db = False
            else:
                current_value = definition.default
                from_db = False

            # Mask secrets
            display_value = current_value
            if definition.is_secret and current_value:
                display_value = "********"

            result.append({
                "key": key,
                "value": display_value,
                "value_type": definition.value_type,
                "category": definition.category,
                "description": definition.description,
                "requires_reload": definition.requires_reload,
                "is_secret": definition.is_secret,
                "env_fallback": definition.env_fallback,
                "from_db": from_db,
            })

        return sorted(result, key=lambda x: (x["category"], x["key"]))

    def list_categories(self) -> list[str]:
        """List all unique categories."""
        categories = set(d.category for d in self._definitions.values())
        return sorted(categories)

    def get_category(self, category: str) -> dict[str, Any]:
        """Get all settings for a category as a dict."""
        result = {}
        for definition in self._definitions.values():
            if definition.category == category:
                result[definition.key] = self.get(definition.key)
        return result

    def sync_from_env(self) -> dict[str, bool]:
        """Sync all settings from environment variables to database.

        This is a one-time migration helper. Only syncs settings that
        have an env_fallback defined and where the env var is set.

        Returns a dict mapping keys to whether they were synced.
        """
        results = {}
        for key, definition in self._definitions.items():
            if not definition.env_fallback:
                results[key] = False
                continue

            env_value = os.getenv(definition.env_fallback)
            if env_value is None:
                results[key] = False
                continue

            # Coerce and set
            value = _coerce_value(env_value, definition.value_type, definition.default)
            results[key] = self.set(key, value)

        synced_count = sum(1 for v in results.values() if v)
        logger.info(f"Synced {synced_count} settings from environment variables")
        return results

    def get_model_config(self, model_type: str = "main") -> dict[str, Any]:
        """Get complete configuration for a model type.

        Args:
            model_type: One of "main", "lightweight", "vision", "cloud"

        Returns:
            Dict with all relevant settings for that model.
        """
        prefix = f"model.{model_type}"
        config = {}

        for key, definition in self._definitions.items():
            if key.startswith(prefix):
                short_key = key.replace(f"{prefix}.", "")
                config[short_key] = self.get(key)

        return config

    def get_inference_config(self, backend: str) -> dict[str, Any]:
        """Get inference configuration for a specific backend.

        Args:
            backend: One of "vllm", "gguf", "transformers", "general"

        Returns:
            Dict with all relevant inference settings.
        """
        prefix = f"inference.{backend}"
        config = {}

        for key, definition in self._definitions.items():
            if key.startswith(prefix):
                short_key = key.replace(f"{prefix}.", "")
                config[short_key] = self.get(key)

        return config


# Global singleton accessor
_settings_service: SettingsService | None = None


def get_settings_service() -> SettingsService:
    """Get the global SettingsService instance."""
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service
