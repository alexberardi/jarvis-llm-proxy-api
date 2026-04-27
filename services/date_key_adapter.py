"""
Date key adapter auto-loading service.

Matches the loaded model to the correct date key LoRA adapter from
adapters/date-keys/ and signals the backend to merge it at startup.

The adapter enhances the model's understanding of date expressions so it
naturally outputs correct date key values during tool routing.
"""

import fnmatch
import json
import logging
from pathlib import Path
from typing import Optional

from services.settings_helpers import get_setting

logger = logging.getLogger("uvicorn")

# Adapters directory relative to project root
_ADAPTERS_DIR = Path(__file__).parent.parent / "adapters" / "date-keys"
_MANIFEST_PATH = _ADAPTERS_DIR / "manifest.json"


def _load_manifest() -> dict:
    """Load the adapter manifest."""
    if not _MANIFEST_PATH.is_file():
        logger.debug("No date key adapter manifest found")
        return {}
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read date key adapter manifest: {e}")
        return {}


def _match_model_to_slug(model_name: str, manifest: dict) -> Optional[str]:
    """Match a model name/path to an adapter slug.

    Tries in order:
    1. Exact HF model ID match
    2. GGUF filename pattern match
    """
    adapters = manifest.get("adapters", {})
    model_basename = Path(model_name).name if "/" not in model_name else model_name

    for slug, entry in adapters.items():
        # Check HF model IDs
        for hf_id in entry.get("hf_model_ids", []):
            if model_name == hf_id or model_basename == hf_id.split("/")[-1]:
                return slug

        # Check GGUF filename patterns
        for pattern in entry.get("gguf_patterns", []):
            if fnmatch.fnmatch(model_basename, pattern):
                return slug

    return None


def resolve_adapter(model_name: str) -> Optional[Path]:
    """Resolve the date key adapter path for the given model.

    Respects the adapter.date_keys setting:
    - "auto" (default): auto-detect from model name
    - "none": disabled, return None
    - "<slug>": use the explicitly selected adapter

    Returns the path to the adapter directory, or None if no adapter found/disabled.
    """
    setting = get_setting("adapter.date_keys", "JARVIS_DATE_KEY_ADAPTER", "auto")

    if setting == "none":
        logger.info("Date key adapter disabled via setting")
        return None

    manifest = _load_manifest()

    if setting != "auto":
        # User explicitly selected an adapter slug
        adapter_dir = _ADAPTERS_DIR / setting
        if adapter_dir.is_dir():
            logger.info(f"Using explicitly selected date key adapter: {setting}")
            return adapter_dir
        logger.warning(f"Selected date key adapter '{setting}' not found at {adapter_dir}")
        return None

    # Auto-detect from model name
    slug = _match_model_to_slug(model_name, manifest)
    if not slug:
        logger.info(f"No date key adapter found for model: {model_name}")
        return None

    adapter_dir = _ADAPTERS_DIR / slug
    if not adapter_dir.is_dir():
        logger.warning(f"Date key adapter directory missing: {adapter_dir}")
        return None

    logger.info(f"Auto-detected date key adapter: {slug} (for model: {model_name})")
    return adapter_dir


def _find_gguf_adapter(adapter_dir: Path) -> Optional[Path]:
    """Find the GGUF adapter file in the adapter directory."""
    gguf_file = adapter_dir / "gguf" / "adapter.gguf"
    if gguf_file.is_file():
        return gguf_file

    # Fallback: any .gguf file in the directory
    for f in adapter_dir.rglob("*.gguf"):
        return f

    return None


def _find_peft_adapter(adapter_dir: Path) -> Optional[Path]:
    """Find the PEFT adapter directory (for MLX/Transformers backends)."""
    if (adapter_dir / "adapter_config.json").is_file():
        return adapter_dir
    return None


def try_load_adapter(backend: object, model_name: str) -> bool:
    """Try to load the date key adapter onto the given backend.

    Call this after the model is loaded. Returns True if adapter was loaded.
    """
    adapter_dir = resolve_adapter(model_name)
    if not adapter_dir:
        return False

    backend_type = type(backend).__name__

    try:
        if backend_type == "GGUFClient":
            gguf_path = _find_gguf_adapter(adapter_dir)
            if gguf_path:
                scale = float(get_setting("adapter.date_keys_scale", "JARVIS_DATE_KEY_ADAPTER_SCALE", "1.0"))
                backend.load_adapter(str(gguf_path), scale=scale)
                logger.info(f"✅ Date key GGUF adapter loaded: {gguf_path} (scale={scale})")
                return True
            else:
                logger.warning(f"No GGUF adapter found in {adapter_dir}")

        elif backend_type == "MlxClient":
            peft_dir = _find_peft_adapter(adapter_dir)
            if peft_dir:
                backend.load_adapter(str(peft_dir))
                logger.info(f"✅ Date key MLX adapter loaded: {peft_dir}")
                return True
            else:
                logger.warning(f"No PEFT adapter found in {adapter_dir}")

        elif backend_type == "TransformersClient":
            peft_dir = _find_peft_adapter(adapter_dir)
            if peft_dir:
                backend.load_adapter(str(peft_dir))
                logger.info(f"✅ Date key Transformers adapter loaded: {peft_dir}")
                return True
            else:
                logger.warning(f"No PEFT adapter found in {adapter_dir}")

        elif backend_type == "VLLMClient":
            # vLLM uses per-request LoRARequest — adapter path stored for later
            peft_dir = _find_peft_adapter(adapter_dir)
            if peft_dir:
                logger.info(f"Date key adapter path registered for vLLM: {peft_dir}")
                # Store for per-request use
                backend._date_key_adapter_path = str(peft_dir)
                return True

        else:
            logger.info(f"Date key adapter not supported for backend: {backend_type}")

    except Exception as e:
        logger.error(f"Failed to load date key adapter: {e}")

    return False


def list_available_adapters() -> list[dict]:
    """List all installed date key adapters with metadata.

    Used by the settings API to populate the adapter selector dropdown.
    """
    manifest = _load_manifest()
    adapters = []

    for slug, entry in manifest.get("adapters", {}).items():
        adapter_dir = _ADAPTERS_DIR / slug
        installed = adapter_dir.is_dir()

        info = {
            "slug": slug,
            "hf_model_ids": entry.get("hf_model_ids", []),
            "installed": installed,
        }

        # Read metadata if installed
        metadata_path = adapter_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                info["eval_accuracy"] = meta.get("eval", {}).get("accuracy")
                info["train_duration_seconds"] = meta.get("train_duration_seconds")
                info["epochs"] = meta.get("epochs")
                info["lora_r"] = meta.get("lora_r")
            except (json.JSONDecodeError, OSError):
                pass

        # Check for GGUF adapter
        gguf_file = adapter_dir / "gguf" / "adapter.gguf"
        if gguf_file.is_file():
            info["gguf_size_mb"] = round(gguf_file.stat().st_size / (1024 * 1024), 1)

        adapters.append(info)

    return adapters
