"""Vision inference service with model swap pattern.

Handles vision/OCR requests by:
1. Unloading the main model to free GPU memory
2. Loading the vision model (e.g., Qwen2.5-VL-AWQ)
3. Processing the vision request
4. Unloading vision model
5. Reloading main model

This swap pattern is necessary when both models cannot fit in GPU memory simultaneously.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from base64 import b64decode
from typing import Any, Dict, List

import httpx

from services.settings_helpers import get_int_setting, get_setting

logger = logging.getLogger("uvicorn")


class VisionInferenceError(Exception):
    """Error during vision inference."""
    pass


def _wait_for_gpu_memory(min_free_gb: float = 8.0, timeout_s: float = 30) -> bool:
    """Wait for GPU memory to be freed."""
    try:
        import torch
        if not torch.cuda.is_available():
            return True

        start = time.time()
        while time.time() - start < timeout_s:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            if free_gb >= min_free_gb:
                logger.info(f"✅ GPU memory available: {free_gb:.2f} GiB free")
                return True
            logger.debug(f"⏳ Waiting for GPU memory: {free_gb:.2f}/{min_free_gb:.2f} GiB free")
            time.sleep(1)

        free_bytes, _ = torch.cuda.mem_get_info()
        logger.warning(f"⚠️  GPU memory wait timeout: {free_bytes / (1024**3):.2f} GiB free (need {min_free_gb})")
        return False
    except ImportError:
        logger.warning("⚠️  torch not available, skipping GPU memory wait")
        return True
    except Exception as e:
        logger.warning(f"⚠️  GPU memory check failed: {e}")
        return True


def _pause_model_service() -> bool:
    """Unload main model from model service to free GPU memory."""
    model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
    if not model_service_url:
        logger.warning("⚠️  MODEL_SERVICE_URL not set, cannot pause model service")
        return False

    token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    headers = {}
    if token:
        headers["X-Internal-Token"] = token

    unload_url = model_service_url.rstrip("/") + "/internal/model/unload"
    logger.info(f"🔌 Vision: Requesting model unload from {unload_url}...")

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(unload_url, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"⚠️  Vision: unload failed {resp.status_code}: {resp.text}")
                return False
            logger.info("🧊 Vision: model service unloaded successfully")
            # Wait for GPU memory to be freed
            _wait_for_gpu_memory(min_free_gb=8.0, timeout_s=30)
            return True
    except Exception as exc:
        logger.warning(f"⚠️  Vision: unload request failed: {exc}")
        return False


def _resume_model_service() -> bool:
    """Reload main model in model service."""
    model_service_url = get_setting("model_service.url", "MODEL_SERVICE_URL", "")
    if not model_service_url:
        logger.warning("⚠️  MODEL_SERVICE_URL not set, cannot resume model service")
        return False

    # Force garbage collection to release vision model memory
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 Vision: Forced CUDA cache clear")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"⚠️  Vision: CUDA cleanup warning: {e}")

    # Wait for vision model memory to be released
    _wait_for_gpu_memory(min_free_gb=9.0, timeout_s=60)

    token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    headers = {}
    if token:
        headers["X-Internal-Token"] = token

    reload_url = model_service_url.rstrip("/") + "/internal/model/reload"
    logger.info(f"🔄 Vision: Requesting model reload from {reload_url}...")

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(reload_url, headers=headers)
            if resp.status_code != 200:
                logger.warning(f"⚠️  Vision: reload failed {resp.status_code}: {resp.text}")
                return False
            logger.info("🔥 Vision: model service reloaded successfully")
            return True
    except Exception as exc:
        logger.warning(f"⚠️  Vision: reload request failed: {exc}")
        return False


def _load_vision_model():
    """Load the vision model for inference."""
    from backends.vllm_vision_backend import VLLMVisionClient

    vision_model_name = get_setting("model.vision.name", "JARVIS_VISION_MODEL_NAME", "")
    vision_chat_format = get_setting(
        "model.vision.chat_format", "JARVIS_VISION_MODEL_CHAT_FORMAT", "qwen"
    )
    vision_context_window = get_int_setting(
        "model.vision.context_window", "JARVIS_VISION_MODEL_CONTEXT_WINDOW", 8192
    )

    if not vision_model_name:
        raise VisionInferenceError("JARVIS_VISION_MODEL_NAME not configured")

    logger.info(f"🔭 Loading vision model: {vision_model_name}")
    return VLLMVisionClient(
        model_path=vision_model_name,
        chat_format=vision_chat_format,
        context_window=vision_context_window,
    )


def _unload_vision_model(vision_client) -> None:
    """Unload the vision model."""
    if vision_client and hasattr(vision_client, "unload"):
        try:
            vision_client.unload()
            logger.info("🧹 Vision model unloaded")
        except Exception as e:
            logger.warning(f"⚠️  Vision model unload error: {e}")

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def _parse_messages(messages: List[Dict[str, Any]]) -> List:
    """Parse messages into NormalizedMessage format with images."""
    from managers.chat_types import NormalizedMessage, TextPart, ImagePart

    normalized = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts = []

        if isinstance(content, str):
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(TextPart(text=item))
                elif isinstance(item, dict):
                    item_type = item.get("type", "text")
                    if item_type == "text":
                        parts.append(TextPart(text=item.get("text", "")))
                    elif item_type == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)

                        # Handle base64 data URLs
                        if url.startswith("data:image"):
                            # Format: data:image/png;base64,<data>
                            try:
                                header, b64_data = url.split(",", 1)
                                image_bytes = b64decode(b64_data)
                                parts.append(ImagePart(data=image_bytes, media_type="image/png"))
                            except Exception as e:
                                logger.warning(f"⚠️  Failed to parse base64 image: {e}")
                        else:
                            logger.warning(f"⚠️  Non-base64 image URLs not yet supported: {url[:50]}...")
                    elif item_type == "image":
                        # Direct image data
                        if "data" in item:
                            image_bytes = b64decode(item["data"]) if isinstance(item["data"], str) else item["data"]
                            parts.append(ImagePart(data=image_bytes, media_type=item.get("media_type", "image/png")))

        if not parts:
            parts.append(TextPart(text=""))

        normalized.append(NormalizedMessage(role=role, content=parts))

    return normalized


def run_vision_inference(request: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Run vision inference with model swap pattern.

    Args:
        request: Vision request containing messages with images
        job_id: Job identifier for logging

    Returns:
        Dict with inference result
    """
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 1024)
    temperature = request.get("temperature", 0.7)
    response_format = request.get("response_format")

    if not messages:
        raise VisionInferenceError("No messages provided")

    vision_client = None
    started = time.time()

    try:
        # Step 1: Unload main model
        logger.info(f"🔭 Vision job {job_id}: Starting model swap")
        if not _pause_model_service():
            raise VisionInferenceError("Failed to unload main model")

        # Step 2: Load vision model
        vision_client = _load_vision_model()
        load_time = time.time() - started
        logger.info(f"🔭 Vision model loaded in {load_time:.1f}s")

        # Step 3: Parse and process request
        from managers.chat_types import GenerationParams

        normalized_messages = _parse_messages(messages)
        params = GenerationParams(
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

        # Run inference
        import asyncio
        result = asyncio.run(vision_client.generate_vision_chat(None, normalized_messages, params))

        inference_time = time.time() - started - load_time
        logger.info(f"✅ Vision inference completed in {inference_time:.1f}s (total: {time.time() - started:.1f}s)")

        return {
            "content": result.content,
            "usage": result.usage,
            "timing": {
                "load_time_s": round(load_time, 2),
                "inference_time_s": round(inference_time, 2),
                "total_time_s": round(time.time() - started, 2),
            },
        }

    finally:
        # Step 4: Unload vision model
        if vision_client:
            _unload_vision_model(vision_client)
            vision_client = None

        # Step 5: Reload main model
        _resume_model_service()
        total_time = time.time() - started
        logger.info(f"🔄 Vision job {job_id}: Model swap complete ({total_time:.1f}s total)")
