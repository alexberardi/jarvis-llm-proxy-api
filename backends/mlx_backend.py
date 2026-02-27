import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from mlx_lm.generate import generate, stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from managers.chat_types import ChatResult, GenerationParams, ImagePart, NormalizedMessage, TextPart
from backends.base import LLMBackendBase

logger = logging.getLogger("uvicorn")


@dataclass
class _PromptCacheState:
    """Persistent KV cache state for prompt prefix reuse."""
    cache: list                # per-layer KVCache objects (mutated in-place by generate)
    cached_tokens: list[int]   # token IDs processed into the cache
    is_trimmable: bool         # whether trim_prompt_cache works for this model


class MlxClient(LLMBackendBase):
    """MLX backend with optional per-request LoRA adapter support.

    Adapter Loading:
        MLX-LM supports LoRA adapters via the load() function's adapter_path parameter.
        Adapters can be swapped dynamically using load_adapters() and removed with
        remove_lora_layers().

    Per-Request Adapter Swapping:
        Unlike vLLM which has native per-request LoRARequest support, MLX requires
        calling load_adapters(model, adapter_path) to switch adapters. This modifies
        the model in-place by loading new weights.

        The workflow is:
        1. Load base model once at init
        2. For each request with adapter_settings.hash:
           - If hash differs from current, call load_adapters()
           - If no adapter requested, call remove_lora_layers()
        3. Run generation

    References:
        - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
        - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py
    """

    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        logger.info("🔁 Loading MLX model...")
        self.model_name = model_path
        self.model_path = model_path
        self._current_adapter_path: Optional[str] = None
        self._current_adapter_hash: Optional[str] = None
        self._lora_layers_applied: bool = False
        self.last_usage = None
        self.inference_engine = "mlx"  # Apple Silicon MLX backend
        self._cache_state: _PromptCacheState | None = None

        # Load model with optional initial adapter
        if adapter_path:
            logger.info(f"🧩 Loading MLX model with adapter: {adapter_path}")
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            self._current_adapter_path = adapter_path
            self._lora_layers_applied = True
        else:
            self.model, self.tokenizer = load(model_path)

    def load_adapter(self, adapter_path: str) -> None:
        """Load or swap a LoRA adapter dynamically.

        This method supports per-request adapter swapping for MLX models.
        It handles three cases:
        1. Same adapter already loaded: no-op
        2. Different adapter requested: load new adapter weights
        3. First adapter on base model: apply LoRA layers + load weights

        Args:
            adapter_path: Path to adapter directory containing adapters.safetensors
                         and adapter_config.json

        Note:
            The adapter directory must contain:
            - adapters.safetensors: The adapter weights
            - adapter_config.json: LoRA configuration (num_layers, rank, etc.)

            If adapter_config.json is missing, you may need to create it:
            {
                "num_layers": 16,
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "scale": 2.0
            }
        """
        if self._current_adapter_path == adapter_path:
            logger.debug(f"🧩 MLX adapter already loaded: {adapter_path}")
            return

        try:
            from mlx_lm.tuner.utils import load_adapters

            logger.info(f"🧩 MLX loading adapter: {adapter_path}")
            start_time = time.time()

            # load_adapters modifies model in-place and returns it
            self.model = load_adapters(self.model, adapter_path)
            self._current_adapter_path = adapter_path
            self._lora_layers_applied = True
            self._invalidate_cache()

            elapsed = time.time() - start_time
            logger.info(f"✅ MLX adapter loaded in {elapsed:.2f}s")

        except ImportError as e:
            logger.warning(f"⚠️  MLX LoRA tuner not available: {e}")
            logger.info("    Install with: pip install mlx-lm[lora]")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  MLX adapter files not found: {e}")
            logger.info(f"    Expected: {adapter_path}/adapters.safetensors")
            logger.info(f"    And:      {adapter_path}/adapter_config.json")
        except Exception as e:
            logger.warning(f"⚠️  MLX adapter load failed: {e}")

    def remove_adapter(self) -> None:
        """Remove LoRA adapter and revert to base model.

        This is useful when a request explicitly wants the base model
        without any adapter applied.
        """
        if not self._lora_layers_applied:
            logger.debug("🧩 MLX no adapter to remove (base model)")
            return

        try:
            from mlx_lm.tuner.utils import remove_lora_layers

            logger.info("🧩 MLX removing adapter, reverting to base model...")
            start_time = time.time()

            self.model = remove_lora_layers(self.model)
            self._current_adapter_path = None
            self._current_adapter_hash = None
            self._lora_layers_applied = False
            self._invalidate_cache()

            elapsed = time.time() - start_time
            logger.info(f"✅ MLX adapter removed in {elapsed:.2f}s")

        except ImportError as e:
            logger.warning(f"⚠️  MLX LoRA tuner not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️  MLX adapter removal failed: {e}")

    def get_current_adapter(self) -> Optional[str]:
        """Return the currently loaded adapter path, or None if base model."""
        return self._current_adapter_path

    def _resolve_mlx_adapter(self, adapter_hash: str) -> Optional[str]:
        """Resolve an adapter hash to a local PEFT adapter directory path.

        MLX uses PEFT-format adapters (adapters.safetensors + adapter_config.json)
        which live at the root of the cached adapter directory.

        Args:
            adapter_hash: Unique adapter identifier from adapter_settings.

        Returns:
            Path to adapter directory, or None if not found.
        """
        from services import adapter_cache  # lazy import to avoid import chain

        adapter_dir = adapter_cache.get_adapter_path(adapter_hash)
        if adapter_dir is None:
            logger.warning(f"🧩 [MLX] Adapter {adapter_hash} not found in cache")
            return None

        # Check for PEFT adapter files at root
        safetensors = adapter_dir / "adapters.safetensors"
        config = adapter_dir / "adapter_config.json"
        if safetensors.is_file() and config.is_file():
            return str(adapter_dir)

        # Also check for HF-style naming (adapter_model.safetensors)
        hf_safetensors = adapter_dir / "adapter_model.safetensors"
        if hf_safetensors.is_file() and config.is_file():
            return str(adapter_dir)

        logger.warning(
            f"🧩 [MLX] Adapter dir {adapter_dir} missing expected files "
            f"(adapters.safetensors + adapter_config.json)"
        )
        return None

    def _handle_adapter_switch(self, adapter_settings: dict) -> None:
        """Handle per-request adapter switching based on adapter_settings.

        Sticky behavior: when adapter_settings is absent or disabled,
        keep the current adapter loaded (avoids unnecessary reloads).

        Args:
            adapter_settings: Dict with 'hash', 'scale', and 'enabled' keys.
        """
        adapter_hash = adapter_settings.get("hash")
        adapter_enabled = adapter_settings.get("enabled", True)

        if not adapter_hash or not adapter_enabled:
            return  # sticky — keep current adapter

        if adapter_hash == self._current_adapter_hash:
            return  # same adapter, skip reload

        adapter_path = self._resolve_mlx_adapter(adapter_hash)
        if adapter_path is None:
            return

        self.load_adapter(adapter_path)
        self._current_adapter_hash = adapter_hash

    # ── KV cache helpers ──────────────────────────────────────────────

    def _invalidate_cache(self) -> None:
        """Invalidate the KV cache (e.g., after adapter switch or model unload)."""
        self._cache_state = None

    def _build_prompt_string(self, messages: list[dict]) -> str:
        """Build a prompt string from messages using the tokenizer's chat template."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            return self.tokenizer.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        # Fallback to simple format if no chat template
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[SYSTEM] {content}")
            elif role == "user":
                parts.append(f"[USER] {content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT] {content}")
        return "\n".join(parts).strip()

    def _tokenize_prompt(self, prompt_str: str) -> list[int]:
        """Tokenize a prompt string matching stream_generate's exact logic."""
        # Match stream_generate: skip BOS if the prompt already starts with it
        raw_tokenizer = getattr(self.tokenizer, '_tokenizer', self.tokenizer)
        bos = getattr(raw_tokenizer, 'bos_token', None)
        add_special = bos is None or not prompt_str.startswith(bos)
        return self.tokenizer.encode(prompt_str, add_special_tokens=add_special)

    @staticmethod
    def _find_common_prefix_length(a: list[int], b: list[int]) -> int:
        """Return length of longest common prefix of two token lists."""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return i
        return min_len

    # ── Core generation with KV cache ─────────────────────────────────

    def _generate_with_cache(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> tuple[str, int, int, float]:
        """Generate with KV cache reuse for prompt prefix caching.

        Returns: (response_text, prompt_tokens, completion_tokens, generation_tps)
        """
        prompt_str = self._build_prompt_string(messages)
        new_tokens = self._tokenize_prompt(prompt_str)

        # Determine cache + suffix tokens to process
        if self._cache_state is not None and self._cache_state.is_trimmable:
            common_len = self._find_common_prefix_length(
                self._cache_state.cached_tokens, new_tokens
            )
            if common_len > 0:
                # Trim cache to common prefix point
                tokens_to_trim = len(self._cache_state.cached_tokens) - common_len
                if tokens_to_trim > 0:
                    trim_prompt_cache(self._cache_state.cache, tokens_to_trim)
                suffix_tokens = new_tokens[common_len:]
                cache = self._cache_state.cache
                logger.info(
                    f"🔄 KV cache hit: {common_len}/{len(new_tokens)} tokens reused, "
                    f"processing {len(suffix_tokens)} new tokens"
                )
            else:
                logger.info("🔄 No KV cache prefix match, creating new cache")
                cache = make_prompt_cache(self.model)
                suffix_tokens = new_tokens
        else:
            if self._cache_state is not None:
                logger.info("🔄 KV cache not trimmable, creating new cache")
            else:
                logger.info("🔄 No KV cache, creating new one")
            cache = make_prompt_cache(self.model)
            suffix_tokens = new_tokens

        # Edge case: identical prompt — pop 1 token so stream_generate has input
        if len(suffix_tokens) == 0:
            trim_prompt_cache(cache, 1)
            suffix_tokens = new_tokens[-1:]

        sampler = make_sampler(temp=temperature)
        response_text = ""
        gen_tokens = 0
        generation_tps = 0.0

        for response in stream_generate(
            self.model,
            self.tokenizer,
            suffix_tokens,  # List[int] skips re-tokenization
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=cache,
        ):
            response_text += response.text
            gen_tokens = response.generation_tokens
            generation_tps = response.generation_tps

        # Trim generated tokens so cache contains only prompt tokens
        if gen_tokens > 0:
            trim_prompt_cache(cache, gen_tokens)

        # Update cache state
        self._cache_state = _PromptCacheState(
            cache=cache,
            cached_tokens=list(new_tokens),
            is_trimmable=can_trim_prompt_cache(cache),
        )

        return response_text.strip(), len(new_tokens), gen_tokens, generation_tps

    # ── Public generation methods ─────────────────────────────────────

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Chat method with temperature support."""
        return self.chat_with_temperature(messages, temperature)

    def chat_with_temperature(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 256) -> str:
        start_time = time.time()

        response_text, prompt_tokens, completion_tokens, generation_tps = self._generate_with_cache(
            messages, temperature, max_tokens
        )

        total_time = time.time() - start_time

        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        logger.debug(
            f"🚀 Generated {completion_tokens} tokens in {total_time:.2f}s "
            f"({generation_tps:.1f} tok/s)"
        )
        logger.debug(
            f"📊 Prompt: {prompt_tokens} tokens | Completion: {completion_tokens} tokens"
        )

        return response_text

    async def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        # Per-request adapter switching via adapter cache
        if params.adapter_settings:
            self._handle_adapter_switch(params.adapter_settings)

        formatted = self._to_dict_messages(messages)
        max_tokens = params.max_tokens or 256
        content = self.chat_with_temperature(formatted, params.temperature, max_tokens=max_tokens)
        return ChatResult(content=content, usage=self.last_usage, tool_calls=None, finish_reason="stop")

    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        prompt, images = self._build_prompt_and_images(messages, self.tokenizer)

        sampler = make_sampler(temp=params.temperature)
        generate_kwargs: Dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "images": images,
            "verbose": False,
            "sampler": sampler,
        }
        if params.max_tokens is not None:
            generate_kwargs["max_tokens"] = params.max_tokens

        start_time = time.time()
        response = generate(**generate_kwargs)
        end_time = time.time()

        completion_tokens = len(response.split())
        prompt_tokens = len(prompt.split())
        total_tokens = prompt_tokens + completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        tokens_per_second = completion_tokens / (end_time - start_time) if end_time > start_time else 0
        logger.debug(f"🚀 [MLX vision] ~{completion_tokens} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tok/s)")

        return ChatResult(content=response.strip(), usage=self.last_usage, tool_calls=None, finish_reason="stop")

    def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context for warm-up — populates the KV cache with the system prompt."""
        try:
            self._generate_with_cache(messages, temperature=0.0, max_tokens=1)

            logger.debug(f"🔥 Context processed for {len(messages)} messages")
            return {
                "messages": messages,
                "context_processed": True,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.warning(f"⚠️  Error processing context: {e}")
            return {
                "messages": messages,
                "context_processed": False,
                "timestamp": time.time(),
            }
    
    def unload(self):
        """Unload the model and clean up resources"""
        self._invalidate_cache()
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        logger.info(f"🔄 Unloaded model: {self.model_name}")

    @staticmethod
    def _to_dict_messages(messages: List[NormalizedMessage]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        for message in messages:
            text_segments = [part.text for part in message.content if isinstance(part, TextPart)]
            formatted.append({"role": message.role, "content": "\n\n".join(text_segments).strip()})
        return formatted

    @staticmethod
    def _build_prompt_and_images(
        messages: List[NormalizedMessage],
        tokenizer: Any = None,
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

        if tokenizer is not None and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                templated_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Simple fallback formatting
            prompt_parts: List[str] = []
            for msg in templated_messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt_parts.append(f"[SYSTEM] {content}")
                elif role == "user":
                    prompt_parts.append(f"[USER] {content}")
                elif role == "assistant":
                    prompt_parts.append(f"[ASSISTANT] {content}")
            prompt = "\n".join(prompt_parts).strip()

        return prompt, images