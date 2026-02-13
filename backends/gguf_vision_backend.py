"""GGUF Vision Backend using llama-cpp-python's multimodal API.

This backend supports LLaVA-style GGUF models with separate CLIP vision encoders.
It allows running quantized vision models on CPU while the main model uses GPU,
saving GPU memory.

Example usage:
    client = GGUFVisionClient(
        model_path="/path/to/llava-v1.5-7b.Q4_K_M.gguf",
        clip_model_path="/path/to/mmproj-model-f16.gguf",
        context_window=4096,
    )
    result = client.generate_vision_chat(None, messages, params)
"""

import base64
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from backends.base import LLMBackendBase

if TYPE_CHECKING:
    pass
from managers.chat_types import (
    ChatResult,
    GenerationParams,
    ImagePart,
    NormalizedMessage,
    TextPart,
)
from services.settings_helpers import get_bool_setting, get_int_setting

logger = logging.getLogger("uvicorn")


class GGUFVisionClient(LLMBackendBase):
    """Vision-only GGUF client using llama-cpp-python's LLaVA support.

    This client handles vision/multimodal inference using quantized GGUF models
    with CLIP vision encoders. It only implements generate_vision_chat;
    text-only calls should use the standard GGUF backend.

    Attributes:
        model_name: Path to the GGUF model file
        model_path: Path to the GGUF model file
        clip_model_path: Path to the CLIP/mmproj GGUF file
        inference_engine: Always "gguf_vision"
        model: The loaded Llama model instance
        chat_handler: The Llava15ChatHandler instance
        last_usage: Token usage from last generation
    """

    def __init__(
        self,
        model_path: str,
        clip_model_path: str,
        chat_format: Optional[str] = None,
        context_window: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
    ):
        """Initialize the GGUF Vision client.

        Args:
            model_path: Path to the main GGUF model file (e.g., llava.gguf)
            clip_model_path: Path to the CLIP/mmproj GGUF file
            chat_format: Optional chat template format (usually auto-detected)
            context_window: Maximum context window size (default: from env or 4096)
            n_gpu_layers: Number of layers to offload to GPU (0=CPU only, -1=all).
                         Default: from env JARVIS_VISION_N_GPU_LAYERS or 0

        Raises:
            ValueError: If model_path or clip_model_path is empty
        """
        if not model_path:
            raise ValueError("Model path is required for GGUFVisionClient")
        if not clip_model_path:
            raise ValueError("CLIP model path is required for GGUFVisionClient")

        self.model_name = model_path
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.chat_format = chat_format
        self.last_usage: Optional[Dict[str, int]] = None
        self.inference_engine = "gguf_vision"
        self._lock = threading.Lock()

        # Get context window from parameter or environment
        if context_window is None:
            context_window = get_int_setting(
                "model.vision.context_window", "JARVIS_VISION_MODEL_CONTEXT_WINDOW", 4096
            )
        self.context_window = context_window

        # Get hardware settings from parameter or environment
        if n_gpu_layers is None:
            n_gpu_layers = get_int_setting(
                "model.vision.n_gpu_layers", "JARVIS_VISION_N_GPU_LAYERS", 0
            )
        n_threads = get_int_setting(
            "inference.gguf.n_threads", "JARVIS_N_THREADS", min(10, os.cpu_count() or 4)
        )
        verbose = get_bool_setting("inference.gguf.verbose", "JARVIS_VERBOSE", False)

        logger.info(f"🔁 Loading GGUF Vision model: {model_path}")
        logger.info(f"   CLIP encoder: {clip_model_path}")
        logger.info(f"   Context window: {context_window}")
        logger.info(f"   GPU layers: {n_gpu_layers}")

        # Import llama_cpp at runtime (allows mocking in tests)
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        # Initialize the CLIP/vision chat handler
        self.chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path)

        # Initialize the Llama model with the chat handler
        self.model = Llama(
            model_path=model_path,
            chat_handler=self.chat_handler,
            n_ctx=context_window,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

        logger.info(f"✅ GGUF Vision model loaded: {model_path}")

    def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Generate vision response using GGUF LLaVA model.

        Converts NormalizedMessage format to llama.cpp's expected format,
        converting ImagePart data to base64 data URLs.

        Args:
            model_cfg: Model configuration (unused, for interface compatibility)
            messages: List of normalized messages with text and image parts
            params: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            ChatResult with generated content and token usage
        """
        with self._lock:
            start_time = time.time()

            # Convert messages to llama.cpp format
            llama_messages = self._convert_messages(messages)

            # Build generation kwargs
            gen_kwargs: Dict[str, Any] = {
                "messages": llama_messages,
                "temperature": params.temperature if params.temperature is not None else 0.7,
                "max_tokens": params.max_tokens or 256,
                "stream": False,
            }

            # Call the model
            response = self.model.create_chat_completion(**gen_kwargs)

            # Extract response content
            content = response["choices"][0]["message"]["content"] or ""

            # Extract usage
            usage = response.get("usage", {})
            self.last_usage = usage

            # Log performance metrics
            end_time = time.time()
            completion_tokens = usage.get("completion_tokens", 0)
            tokens_per_second = completion_tokens / (end_time - start_time) if end_time > start_time else 0
            logger.info(
                f"🚀 [GGUF-VISION] {completion_tokens} tokens in {end_time - start_time:.2f}s "
                f"({tokens_per_second:.1f} tok/s)"
            )

            return ChatResult(content=content, usage=usage)

    def _convert_messages(
        self, messages: List[NormalizedMessage]
    ) -> List[Dict[str, Any]]:
        """Convert NormalizedMessage list to llama.cpp message format.

        Converts ImagePart instances to base64 data URLs in the format
        expected by llama-cpp-python's multimodal API.

        Args:
            messages: List of normalized messages with text and image parts

        Returns:
            List of message dicts in llama.cpp format
        """
        llama_messages: List[Dict[str, Any]] = []

        for message in messages:
            content_parts: List[Dict[str, Any]] = []

            for part in message.content:
                if isinstance(part, TextPart):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    # Convert image data to base64 data URL
                    b64_data = base64.b64encode(part.data).decode("utf-8")
                    data_url = f"data:{part.mime_type};base64,{b64_data}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

            # If no content parts, add empty text
            if not content_parts:
                content_parts.append({"type": "text", "text": ""})

            llama_messages.append({
                "role": message.role,
                "content": content_parts,
            })

        return llama_messages

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Text-only generation is not supported by this vision-only backend.

        Raises:
            NotImplementedError: Always, as this is a vision-only backend
        """
        raise NotImplementedError(
            "GGUFVisionClient is vision-only. Use GGUFClient for text generation."
        )

    def unload(self) -> None:
        """Unload the model and free resources."""
        logger.info(f"🔄 Unloading GGUF Vision model: {self.model_path}")

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, "chat_handler") and self.chat_handler is not None:
            del self.chat_handler
            self.chat_handler = None

        logger.info(f"✅ GGUF Vision model unloaded")
