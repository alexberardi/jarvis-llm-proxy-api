"""vLLM Vision Backend for multimodal inference.

Supports vision-language models like Qwen2.5-VL with AWQ quantization.
Uses vLLM's native multimodal capabilities for efficient GPU inference.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional

# Fix for vLLM CUDA multiprocessing issue
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from backends.base import LLMBackendBase
from managers.chat_types import (
    ChatResult,
    GenerationParams,
    ImagePart,
    NormalizedMessage,
    TextPart,
)
from services.settings_helpers import get_float_setting, get_int_setting, get_setting

logger = logging.getLogger("uvicorn")


class VLLMVisionClient(LLMBackendBase):
    """
    vLLM-based vision backend for multimodal models (e.g., Qwen2.5-VL-AWQ).

    Supports:
    - Single and multi-image inputs
    - AWQ/GPTQ quantization
    - JSON structured outputs
    - Efficient GPU memory management
    """

    def __init__(
        self,
        model_path: str,
        chat_format: Optional[str] = None,
        stop_tokens: Optional[str] = None,
        context_window: Optional[int] = None,
    ):
        if not model_path:
            raise ValueError("Model path is required")

        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = (chat_format or "qwen").lower()
        self.model = None
        self.processor = None
        self.last_usage = None
        self.inference_engine = "vllm_vision"

        # Get context window from parameter or environment variable
        if context_window is None:
            context_window = get_int_setting(
                "model.vision.context_window", "JARVIS_VISION_MODEL_CONTEXT_WINDOW", 8192
            )
        self.context_window = context_window

        # vLLM configuration
        tensor_parallel_size = get_int_setting(
            "inference.vllm.tensor_parallel_size", "JARVIS_VLLM_TENSOR_PARALLEL_SIZE", 1
        )
        gpu_memory_utilization = get_float_setting(
            "inference.vllm.gpu_memory_utilization", "JARVIS_VLLM_GPU_MEMORY_UTILIZATION", 0.9
        )

        # Use vision-specific quantization setting if available, else fallback to main
        vllm_quantization = get_setting(
            "model.vision.vllm_quantization", "JARVIS_VISION_VLLM_QUANTIZATION", ""
        ).strip().lower()
        if not vllm_quantization:
            vllm_quantization = get_setting(
                "inference.vllm.quantization", "JARVIS_VLLM_QUANTIZATION", ""
            ).strip().lower()
        vllm_quantization = vllm_quantization.replace("-", "_")
        if vllm_quantization in {"", "none", "false", "off", "auto"}:
            vllm_quantization = None

        # Batching parameters - more conservative for vision models
        max_num_batched_tokens = get_int_setting(
            "model.vision.vllm_max_batched_tokens",
            "JARVIS_VISION_VLLM_MAX_BATCHED_TOKENS",
            4096,
        )
        max_num_seqs = get_int_setting(
            "model.vision.vllm_max_num_seqs",
            "JARVIS_VISION_VLLM_MAX_NUM_SEQS",
            8,
        )

        logger.info(f"🔭 vLLM Vision: Loading {model_path}")
        logger.debug(f"🔭 vLLM Vision: Context window: {context_window}")
        logger.debug(f"🔭 vLLM Vision: GPU memory utilization: {gpu_memory_utilization}")
        logger.debug(f"🔭 vLLM Vision: Quantization: {vllm_quantization or 'none'}")

        # Check for GGUF (not supported by vLLM for vision)
        if model_path.endswith('.gguf'):
            raise ValueError(
                f"vLLM Vision does not support GGUF files. Use HuggingFace model ID: {model_path}"
            )

        # Load processor for chat template
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logger.debug(f"🔭 vLLM Vision: Processor loaded")
        except Exception as e:
            logger.warning(f"⚠️  vLLM Vision: Could not load processor: {e}")
            self.processor = None

        # Initialize vLLM engine
        try:
            llm_kwargs: Dict[str, Any] = {
                "model": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": context_window,
                "trust_remote_code": True,
                "enforce_eager": True,  # Safer for multimodal
                "disable_log_stats": True,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
                "enable_prefix_caching": False,  # Disable for vision
                "swap_space": 0,
                # Limit multimodal data per prompt
                "limit_mm_per_prompt": {"image": 4},
            }

            if vllm_quantization is not None:
                llm_kwargs["quantization"] = vllm_quantization

            self.model = LLM(**llm_kwargs)
            logger.info(f"✅ vLLM Vision model loaded: {model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load vLLM Vision model: {e}")
            raise

        # Stop tokens
        if isinstance(stop_tokens, str):
            stop_tokens = [t.strip() for t in stop_tokens.split(",") if t.strip()]
        self.stop_tokens = stop_tokens or []
        if not self.stop_tokens:
            if self.chat_format in {"chatml", "qwen"}:
                self.stop_tokens = ["<|im_end|>"]

    def _prepare_multimodal_inputs(
        self,
        messages: List[NormalizedMessage],
    ) -> tuple[str, Dict[str, Any]]:
        """Convert normalized messages to vLLM multimodal format.

        Returns:
            Tuple of (prompt_text, multi_modal_data)
        """
        images: List[Image.Image] = []
        processor_messages: List[Dict[str, Any]] = []

        for message in messages:
            parts: List[Dict[str, Any]] = []

            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    # Load image from bytes
                    img = Image.open(io.BytesIO(part.data)).convert("RGB")
                    images.append(img)
                    # Add image placeholder for vLLM
                    parts.append({"type": "image"})

            if not parts:
                parts.append({"type": "text", "text": ""})

            processor_messages.append({"role": message.role, "content": parts})

        # Apply chat template to get prompt
        if self.processor and hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                processor_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Fallback: manual ChatML format
            prompt = self._manual_chat_template(processor_messages)

        # Prepare multimodal data for vLLM
        multi_modal_data: Dict[str, Any] = {}
        if images:
            multi_modal_data["image"] = images if len(images) > 1 else images[0]

        return prompt, multi_modal_data

    def _manual_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        """Manual ChatML template for when processor is unavailable."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content_parts = msg.get("content", [])

            # Extract text from content parts
            text = ""
            for part in content_parts:
                if part.get("type") == "text":
                    text += part.get("text", "")
                elif part.get("type") == "image":
                    text += "<image>"

            prompt += f"<|im_start|>{role}\n{text}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Generate vision chat response using vLLM multimodal inference."""

        # Prepare inputs
        prompt, multi_modal_data = self._prepare_multimodal_inputs(messages)

        # Configure sampling parameters
        max_tokens = params.max_tokens or 1024
        temperature = params.temperature if params.temperature is not None else 0.7
        do_sample = temperature > 0

        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature if do_sample else 0,
            "max_tokens": max_tokens,
            "stop": self.stop_tokens,
        }

        # Handle JSON structured output
        if params.response_format and params.response_format.get("type") == "json_object":
            if "json_schema" in params.response_format:
                json_schema = params.response_format["json_schema"]
                sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=json_schema)
                logger.debug(f"🔒 Vision: Enforcing JSON schema")
            else:
                sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json_object=True)
                logger.debug(f"🔒 Vision: Enforcing valid JSON object")

        if params.top_p is not None:
            sampling_kwargs["top_p"] = params.top_p

        sampling_params = SamplingParams(**sampling_kwargs)

        logger.debug(f"🔭 vLLM Vision generating: max_tokens={max_tokens}, images={len(multi_modal_data.get('image', []))}")

        try:
            # Generate with multimodal data
            if multi_modal_data:
                outputs = self.model.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": multi_modal_data,
                    },
                    sampling_params=sampling_params,
                )
            else:
                outputs = self.model.generate([prompt], sampling_params=sampling_params)

            if not outputs:
                raise ValueError("No output generated")

            output = outputs[0]
            generated_text = output.outputs[0].text.strip()

            # Calculate usage
            prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
            completion_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
            total_tokens = prompt_tokens + completion_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.last_usage = usage

            return ChatResult(content=generated_text, usage=usage)

        except Exception as e:
            logger.error(f"❌ vLLM Vision generation error: {e}")
            raise

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Vision-only backend does not support text-only chat."""
        raise NotImplementedError(
            "VLLMVisionClient is vision-only. Use VLLMClient for text generation."
        )

    def unload(self) -> None:
        """Unload the vision model to free GPU memory."""
        import gc
        import os
        import signal
        import time

        if self.model:
            current_pid = os.getpid()
            child_pids = []

            # Find vLLM child processes
            try:
                import psutil
                current_proc = psutil.Process(current_pid)
                for child in current_proc.children(recursive=True):
                    try:
                        if 'python' in child.name().lower() or 'vllm' in ' '.join(child.cmdline()).lower():
                            child_pids.append(child.pid)
                            logger.debug(f"🔍 Found vLLM Vision child process: PID {child.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                logger.warning("⚠️  psutil not available for cleanup")
            except Exception as e:
                logger.warning(f"⚠️  Error finding child processes: {e}")

            # Attempt graceful shutdown
            try:
                engine = getattr(self.model, "llm_engine", None)
                if engine:
                    engine_core = getattr(engine, "engine_core", None)
                    if engine_core:
                        if hasattr(engine_core, "shutdown"):
                            engine_core.shutdown()
                        if hasattr(engine_core, "close"):
                            engine_core.close()
                    if hasattr(engine, "shutdown"):
                        engine.shutdown()
                logger.info("🧹 vLLM Vision engine shutdown complete")
            except Exception as e:
                logger.warning(f"⚠️  vLLM Vision engine shutdown failed: {e}")

            # Delete model reference
            model_ref = self.model
            self.model = None
            del model_ref
            logger.info(f"🗑️  vLLM Vision model reference deleted: {self.model_name}")

            # Clean up processor
            if self.processor:
                del self.processor
                self.processor = None

            gc.collect()
            time.sleep(1)

            # Force kill child processes
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass

            time.sleep(2)

            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            gc.collect()

            # Clear CUDA cache
            try:
                import torch
                import torch.distributed as dist

                if dist.is_initialized():
                    try:
                        dist.destroy_process_group()
                        logger.info("🧹 Destroyed torch distributed process group")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to destroy process group: {e}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                    total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
                    logger.info(f"🧹 CUDA cache cleared. Free: {free_mem:.2f}/{total_mem:.2f} GiB")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"⚠️  CUDA cleanup error: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "backend": "vLLM Vision",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "chat_format": self.chat_format,
            "context_window": self.context_window,
            "last_usage": self.last_usage,
            "engine": "vLLM Multimodal Inference",
        }
