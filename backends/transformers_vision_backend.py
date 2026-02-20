from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    AutoProcessor,
)
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None

from managers.chat_types import ChatResult, GenerationParams, ImagePart, NormalizedMessage, TextPart
from backends.base import LLMBackendBase
from services.settings_helpers import get_setting


class TransformersVisionClient(LLMBackendBase):
    """
    Vision backend using HuggingFace transformers (e.g., Qwen2-VL).

    Supports multiple images per request when the underlying model/processor supports it.
    """

    def __init__(self, model_path: str, chat_format: Optional[str] = None, stop_tokens: Optional[str] = None, context_window: Optional[int] = None):
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.stop_tokens = stop_tokens
        self.context_window = context_window

        self.device = self._get_device()
        self.torch_dtype = self._get_dtype()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model_type = getattr(config, "model_type", "").lower()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        attn_impl = get_setting(
            "model.vision.attn_impl", "JARVIS_VISION_ATTN_IMPL", ""
        )

        if self.model_type in {"smolvlm", "smolvlm2"}:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl if attn_impl else None,
            )
        elif self.model_type in {"qwen3vl", "qwen2vl", "qwen2_5_vl"} and Qwen3VLForConditionalGeneration:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl if attn_impl else None,
            )
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                attn_implementation=attn_impl if attn_impl else None,
            )

        self.model.to(self.device)
        self.model.eval()
        self.last_usage = None
        self.inference_engine = "transformers_vision"  # HuggingFace Transformers vision backend

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_dtype(self) -> Optional[torch.dtype]:
        override = get_setting(
            "model.vision.torch_dtype", "JARVIS_VISION_TORCH_DTYPE", ""
        ).lower()
        if override == "float16" or override == "fp16":
            return torch.float16
        if override == "bfloat16" or override == "bf16":
            return torch.bfloat16
        if override == "float32" or override == "fp32":
            return torch.float32
        # Defaults: fp16 on mps/cuda, else fp32
        if self.device in ("mps", "cuda"):
            return torch.float16
        return torch.float32

    @staticmethod
    def _to_processor_messages(messages: List[NormalizedMessage]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        processor_messages: List[Dict[str, Any]] = []
        images: List[Image.Image] = []

        for message in messages:
            parts: List[Dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    img = Image.open(io.BytesIO(part.data)).convert("RGB")
                    images.append(img)
                    parts.append({"type": "image", "image": img})
            # If legacy string-only, ensure at least one text part
            if not parts:
                parts.append({"type": "text", "text": ""})
            processor_messages.append({"role": message.role, "content": parts})
        return processor_messages, images

    @staticmethod
    def _to_processor_messages_smol(messages: List[NormalizedMessage]) -> List[Dict[str, Any]]:
        proc_messages: List[Dict[str, Any]] = []
        for message in messages:
            parts: List[Dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImagePart):
                    img = Image.open(io.BytesIO(part.data)).convert("RGB")
                    parts.append({"type": "image", "image": img})
            if not parts:
                parts.append({"type": "text", "text": ""})
            proc_messages.append({"role": message.role, "content": parts})
        return proc_messages

    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        if self.model_type in {"smolvlm", "smolvlm2"}:
            proc_messages = self._to_processor_messages_smol(messages)
            inputs = self.processor.apply_chat_template(
                proc_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": params.max_tokens or 256,
                "do_sample": params.temperature is not None and params.temperature > 0,
                "temperature": params.temperature if params.temperature is not None else 0.7,
            }
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            prompt_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            completion_tokens = len(output_ids[0])
            total_tokens = prompt_tokens + completion_tokens
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            return ChatResult(content=text, usage=self.last_usage, tool_calls=None, finish_reason="stop")

        if self.model_type in {"qwen3vl", "qwen2vl", "qwen2_5_vl"} and Qwen3VLForConditionalGeneration:
            proc_messages = self._to_processor_messages_smol(messages)
            inputs = self.processor.apply_chat_template(
                proc_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": params.max_tokens or 256,
                "do_sample": params.temperature is not None and params.temperature > 0,
                "temperature": params.temperature if params.temperature is not None else 0.7,
            }
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)

            if "input_ids" in inputs:
                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            else:
                generated_ids = output_ids

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            prompt_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            completion_tokens = len(generated_ids[0])
            total_tokens = prompt_tokens + completion_tokens
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            return ChatResult(content=text, usage=self.last_usage, tool_calls=None, finish_reason="stop")

        proc_messages, images = self._to_processor_messages(messages)

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                proc_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=prompt,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=proc_messages,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": params.max_tokens or 256,
            "do_sample": params.temperature is not None and params.temperature > 0,
            "temperature": params.temperature if params.temperature is not None else 0.7,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        if "input_ids" in inputs:
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        else:
            generated_ids = output_ids

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        prompt_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        completion_tokens = len(generated_ids[0])
        total_tokens = prompt_tokens + completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        return ChatResult(content=text, usage=self.last_usage, tool_calls=None, finish_reason="stop")

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Vision-only backend does not support text-only chat."""
        raise NotImplementedError(
            "TransformersVisionClient is vision-only. Use TransformersClient for text generation."
        )

    def unload(self) -> None:
        """Unload vision model from memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "processor") and self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

