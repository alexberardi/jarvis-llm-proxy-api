from __future__ import annotations

import io
import os
import tempfile
import time
from typing import Any, Dict, List, Tuple

from PIL import Image

from managers.chat_types import ChatResult, GenerationParams, ImagePart, NormalizedMessage, TextPart


class MlxVisionClient:
    """
    Vision-only MLX client that relies on mlx-vlm (vision build).

    This client expects mlx_vlm to be installed. It only implements generate_vision_chat;
    text-only calls should use the standard MLX backend.
    """

    def __init__(self, model_path: str):
        print("🔁 Loading MLX-VLM vision model...")
        from mlx_vlm.generate import generate  # type: ignore
        from mlx_vlm.utils import load, load_config  # type: ignore

        self._generate = generate
        self.model_name = model_path
        self.model_path = model_path
        loaded = load(model_path)

        # Expect (model, processor); tolerate (model, processor, tokenizer)
        self.model = None
        self.processor = None
        if isinstance(loaded, tuple):
            if len(loaded) >= 2:
                self.model, self.processor = loaded[0], loaded[1]
            else:
                self.model = loaded[0]
        else:
            self.model = loaded

        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if not hasattr(self.processor, "image_processor"):
            raise ValueError("MLX-VISION processor lacks image_processor.")

        self.config = load_config(model_path)
        self.last_usage = None

    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        prompt_text, images = self._build_prompt_and_images(messages)

        if self.processor is None:
            raise ValueError("MLX-VISION backend requires a processor; upgrade mlx-vlm or verify model assets.")
        if self.config is None:
            raise ValueError("MLX-VISION backend missing config.")

        # Apply chat template per mlx-vlm docs
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore

        formatted_prompt = apply_chat_template(
            self.processor,
            self.config,
            prompt_text,
            num_images=len(images),
        )

        # mlx_vlm.generate signature: (model, processor, prompt, image=None, audio=None, verbose=False, **kwargs)
        # It expects image paths or strings; persist to temp files when needed.
        tmp_files: List[str] = []
        try:
            image_paths: List[str] = []
            for idx, img in enumerate(images):
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                img.save(path, format="PNG")
                tmp_files.append(path)
                image_paths.append(path)

            image_arg: Any = None
            if image_paths:
                image_arg = image_paths[0] if len(image_paths) == 1 else image_paths

            generate_kwargs: Dict[str, Any] = {
                "model": self.model,
                "processor": self.processor,
                "prompt": formatted_prompt,
                "image": image_arg,
                "verbose": False,
            }
            if params.temperature is not None:
                generate_kwargs["temperature"] = params.temperature
            if params.max_tokens is not None:
                generate_kwargs["max_tokens"] = params.max_tokens

            start_time = time.time()
            result = self._generate(**generate_kwargs)
            end_time = time.time()
        finally:
            for path in tmp_files:
                try:
                    os.remove(path)
                except OSError:
                    pass

        content = None
        # Heuristic extraction from GenerationResult
        if hasattr(result, "generations"):
            gens = getattr(result, "generations")
            if isinstance(gens, list) and gens:
                first = gens[0]
                if isinstance(first, str):
                    content = first
                elif hasattr(first, "text"):
                    content = getattr(first, "text")
        if content is None and hasattr(result, "text"):
            content = getattr(result, "text")
        if content is None:
            content = str(result)

        completion_tokens = len(content.split())
        prompt_tokens = len(formatted_prompt.split())
        total_tokens = prompt_tokens + completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        tokens_per_second = completion_tokens / (end_time - start_time) if end_time > start_time else 0
        print(f"🚀 [MLX-VISION] ~{completion_tokens} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tok/s)")

        return ChatResult(content=content.strip(), usage=self.last_usage)

    @staticmethod
    def _build_prompt_and_images(
        messages: List[NormalizedMessage],
    ) -> Tuple[str, List[Image.Image]]:
        images: List[Image.Image] = []
        templated_messages: List[Dict[str, str]] = []

        for message in messages:
            content_parts: List[str] = []
            for part in message.content:
                if isinstance(part, TextPart):
                    content_parts.append(part.text)
                elif isinstance(part, ImagePart):
                    try:
                        img = Image.open(io.BytesIO(part.data))
                        img = img.convert("RGB")
                    except Exception as exc:  # noqa: BLE001
                        raise ValueError(f"Invalid image data: {exc}") from exc
                    images.append(img)
                    content_parts.append("<image>")
            templated_messages.append({"role": message.role, "content": "\n".join(content_parts).strip()})

        # Build a simple text prompt from the collected text parts (roles included)
        prompt_parts: List[str] = []
        for msg in templated_messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}" if content else role)
        prompt_text = "\n".join(prompt_parts).strip()

        return prompt_text, images

