import logging
import os
import threading
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)

from backends.base import LLMBackendBase

logger = logging.getLogger("uvicorn")


class TransformersClient(LLMBackendBase):
    def __init__(self, model_path: str, chat_format: str = None, stop_tokens: List[str] = None, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")
        
        # Store model info
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.stop_tokens = stop_tokens or []
        self.last_usage = None
        self.last_used_at = None
        self._lock = threading.Lock()  # Add thread safety
        
        # Get context window from parameter or environment variable
        if context_window is None:
            context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "4096"))
        self.context_window = context_window
        self._actual_context_window = None  # Will be set after model loading
        
        # Get device configuration
        self.device = self._get_device()
        
        # Get quantization configuration
        self.use_quantization = os.getenv("JARVIS_USE_QUANTIZATION", "false").lower() == "true"
        self.quantization_type = os.getenv("JARVIS_QUANTIZATION_TYPE", "4bit").lower()
        
        # Get generation parameters
        self.max_tokens = int(os.getenv("JARVIS_MAX_TOKENS", "2048"))
        self.top_p = float(os.getenv("JARVIS_TOP_P", "0.95"))
        self.top_k = int(os.getenv("JARVIS_TOP_K", "50"))
        self.repetition_penalty = float(os.getenv("JARVIS_REPETITION_PENALTY", "1.1"))
        self.do_sample = os.getenv("JARVIS_DO_SAMPLE", "true").lower() == "true"
        
        # Memory optimization settings
        self.use_cache = os.getenv("JARVIS_USE_CACHE", "true").lower() == "true"
        self.torch_dtype = self._get_torch_dtype()
        self.trust_remote_code = os.getenv("JARVIS_TRUST_REMOTE_CODE", "false").lower() == "true"
        
        # Check if vLLM inference should be used
        self.inference_engine = os.getenv("JARVIS_INFERENCE_ENGINE", "transformers").lower()
        self.vllm_backend = None
        # Note: Dynamic per-request adapters only supported on vLLM backend
        # Transformers backend runs base model only
        
        logger.debug(f"🔍 Debug: Model path: {model_path}")
        logger.debug(f"🔍 Debug: Chat format: {chat_format}")
        logger.debug(f"🔍 Debug: Context window: {context_window}")
        logger.debug(f"🔍 Debug: Device: {self.device}")
        logger.debug(f"🔍 Debug: Quantization: {'enabled' if self.use_quantization else 'disabled'}")
        logger.debug(f"🔍 Debug: Torch dtype: {self.torch_dtype}")
        logger.debug(f"🔍 Debug: Trust remote code: {self.trust_remote_code}")
        logger.debug(f"🔍 Debug: Inference engine: {self.inference_engine}")
        
        # Initialize model and tokenizer based on inference engine
        if self.inference_engine == "vllm":
            self._init_vllm()
        else:
            self._load_model()
        
        logger.info(f"✅ Transformers model loaded successfully!")

        # Warm up the model
        self._warmup_model()

        logger.debug(f"🔍 Debug: Model initialization complete")
    
    def _get_device(self) -> str:
        """Determine the best device for inference"""
        device_preference = os.getenv("JARVIS_DEVICE", "auto").lower()
        
        if device_preference == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_preference
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get the appropriate torch dtype"""
        dtype_str = os.getenv("JARVIS_TORCH_DTYPE", "auto").lower()
        
        if dtype_str == "auto":
            if self.device == "cuda":
                return torch.float16
            elif self.device == "mps":
                return torch.float32  # MPS doesn't support float16 well yet
            else:
                return torch.float32
        elif dtype_str == "float16":
            return torch.float16
        elif dtype_str == "float32":
            return torch.float32
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        else:
            return torch.float32
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled"""
        if not self.use_quantization:
            return None
        
        try:
            if self.quantization_type == "4bit":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization_type == "8bit":
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                logger.warning(f"⚠️  Unknown quantization type: {self.quantization_type}")
                return None
        except ImportError:
            logger.warning("⚠️  bitsandbytes not installed, disabling quantization")
            return None
    
    def _init_vllm(self):
        """Initialize vLLM backend for transformers"""
        try:
            from .vllm_backend import VLLMClient
            logger.info("🚀 Initializing vLLM backend for Transformers model")

            self.vllm_backend = VLLMClient(
                self.model_path,
                self.chat_format,
                self.stop_tokens,
                self.context_window
            )

            # Set a dummy tokenizer for compatibility
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            logger.info("✅ vLLM backend initialized successfully")

        except ImportError:
            logger.error("❌ vLLM not installed. Install with: pip install vllm")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize vLLM backend: {e}")
            raise
    
    def _load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"🔁 Loading Transformers model: {self.model_path}")

        # Load tokenizer
        logger.debug("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True  # Use fast tokenizer when available
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get quantization config
        quantization_config = self._get_quantization_config()
        
        # Load model
        logger.debug("🤖 Loading model...")
        device_map_env = os.getenv("JARVIS_TRANSFORMERS_DEVICE_MAP", "").strip().lower()
        if device_map_env in {"", "auto"}:
            device_map = "auto" if self.device == "cuda" else None
        elif device_map_env in {"none", "off", "false", "no"}:
            device_map = None
        else:
            device_map = device_map_env
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "device_map": device_map,
        }
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # Only set low_cpu_mem_usage if not using quantization
            model_kwargs["low_cpu_mem_usage"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        
        # Move model to device only when not using device_map/offload
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()

        logger.debug(f"🔍 Debug: Model loaded on device: {next(self.model.parameters()).device}")
        logger.debug(f"🔍 Debug: Model dtype: {next(self.model.parameters()).dtype}")
        logger.debug(f"🔍 Debug: Tokenizer vocab size: {len(self.tokenizer)}")

        # Detect actual model context window
        self._detect_model_context_window()
        logger.debug(f"🔍 Debug: Model context window: {self._actual_context_window or 'unknown'}")
        logger.debug(f"🔍 Debug: Using context window: {self.context_window}")

    def _get_input_device(self) -> str:
        """Pick a safe input device when using accelerate device_map."""
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            first_device = next(iter(device_map.values()))
            if isinstance(first_device, int):
                return f"cuda:{first_device}"
            if isinstance(first_device, str) and first_device.isdigit():
                return f"cuda:{first_device}"
            return str(first_device)
        return self.device
    
    def _detect_model_context_window(self):
        """Detect the model's actual context window from config"""
        try:
            config = self.model.config
            # Common attribute names for context window
            context_attrs = ['max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length', 'context_length']
            
            for attr in context_attrs:
                if hasattr(config, attr):
                    self._actual_context_window = getattr(config, attr)
                    # Use the detected context window if it's smaller than our setting
                    if self._actual_context_window < self.context_window:
                        logger.warning(f"⚠️  Model's max context ({self._actual_context_window}) is smaller than configured ({self.context_window})")
                        logger.info(f"🔧 Adjusting context window to model's maximum: {self._actual_context_window}")
                        self.context_window = self._actual_context_window
                    break
        except Exception as e:
            logger.warning(f"⚠️  Could not detect model context window: {e}")
    
    def _warmup_model(self):
        """Warm up the model with a small inference"""
        try:
            logger.debug(f"🔍 Debug: Warming up model...")
            warmup_messages = [{"role": "user", "content": "Hello"}]
            self._generate_response(warmup_messages, temperature=0.0, max_new_tokens=1)
            logger.debug(f"🔍 Debug: Model warmup completed successfully")
        except Exception as e:
            logger.warning(f"⚠️  Debug: Model warmup failed: {e}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the tokenizer's chat template or fallback"""
        try:
            # Try to use the tokenizer's chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                return formatted
        except Exception as e:
            logger.warning(f"⚠️  Chat template failed: {e}, using fallback formatting")
        
        # Fallback formatting based on chat_format or generic
        if self.chat_format == "chatml":
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
            return formatted
        elif self.chat_format == "llama":
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n"
                elif role == "user":
                    if not formatted.endswith("[INST] "):
                        formatted += f"[INST] {content} [/INST]"
                    else:
                        formatted += f"{content} [/INST]"
                elif role == "assistant":
                    formatted += f"{content}</s><s>[INST] "
            return formatted
        else:
            # Generic format
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"System: {content}\n"
                elif role == "user":
                    formatted += f"Human: {content}\n"
                elif role == "assistant":
                    formatted += f"Assistant: {content}\n"
            formatted += "Assistant:"
            return formatted
    
    def _generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_new_tokens: int = None) -> str:
        """Generate response using the model"""
        # Format messages
        formatted_prompt = self._format_messages(messages)
        
        # Calculate safe input length
        max_new_tokens_to_use = max_new_tokens or self.max_tokens
        # Reserve space for generation, with a safety margin
        max_input_length = max(self.context_window - max_new_tokens_to_use - 50, 512)
        
        # Tokenize input with proper length handling
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=False
        ).to(self._get_input_device())
        
        # Check if input is too long and warn
        input_length = inputs.input_ids.shape[1]
        if input_length > max_input_length * 0.9:  # Warn if using >90% of available space
            logger.warning(f"⚠️  Input length ({input_length}) is close to maximum ({max_input_length}). Consider reducing context or increasing context window.")
        
        # Set up generation config with conditional parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        
        # Only add sampling parameters if do_sample is True
        if self.do_sample and temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "repetition_penalty": self.repetition_penalty,
            })
        else:
            generation_kwargs["do_sample"] = False
        
        generation_config = GenerationConfig(**generation_kwargs)
        
        # Add stop tokens if provided
        if self.stop_tokens:
            # Convert stop tokens to token IDs
            stop_token_ids = []
            for stop_token in self.stop_tokens:
                token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
                stop_token_ids.extend(token_ids)
            if stop_token_ids:
                generation_config.eos_token_id = stop_token_ids
        
        # Generate with error handling for unsupported parameters
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,  # Penalize repetition
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except Exception as e:
                if "not supported" in str(e).lower() or "invalid" in str(e).lower():
                    logger.warning(f"⚠️  Generation parameter issue: {e}")
                    logger.info("🔧 Retrying with basic generation config...")
                    # Fallback to basic generation
                    basic_config = GenerationConfig(
                        max_new_tokens=max_new_tokens or self.max_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,  # Use greedy decoding as fallback
                    )
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=basic_config,
                    )
                else:
                    raise e
        
        # Decode response
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = response.strip()
        
        # Remove any remaining stop tokens
        for stop_token in self.stop_tokens:
            if response.endswith(stop_token):
                response = response[:-len(stop_token)].strip()

        return response

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List["NormalizedMessage"],
        params: "GenerationParams",
    ) -> "ChatResult":
        """Generate text response using the Transformers backend.

        Converts NormalizedMessage to dict format and calls chat_with_temperature.
        """
        from managers.chat_types import ChatResult, TextPart

        # Convert NormalizedMessage to simple dict format
        dict_messages = []
        for msg in messages:
            text_parts = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    text_parts.append(part.text)
            dict_messages.append({
                "role": msg.role,
                "content": " ".join(text_parts)
            })

        content = self.chat_with_temperature(dict_messages, params.temperature)
        return ChatResult(content=content, usage=self.last_usage)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat method with temperature support"""
        return self.chat_with_temperature(messages, temperature)
    
    def chat_with_temperature(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat with temperature control and thread safety"""
        with self._lock:
            if self.inference_engine == "vllm" and self.vllm_backend:
                return self._chat_vllm(messages, temperature)
            else:
                return self._chat_internal(messages, temperature)
    
    def _chat_vllm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat using vLLM backend"""
        logger.debug(f"🚀 vLLM Transformers chat with {len(messages)} messages, temperature: {temperature}")
        try:
            response_text, usage = self.vllm_backend.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            
            # Store usage information for compatibility
            self.last_usage = usage
            
            # Update last usage time
            self.last_used_at = time.time()
            
            return response_text
            
        except Exception as e:
            logger.error(f"❌ vLLM Transformers chat error: {e}")
            return ""
    
    def _chat_internal(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Internal chat implementation"""
        start_time = time.time()

        logger.debug(f"🔍 DEBUG: Starting chat with {len(messages)} messages")
        logger.debug(f"🔍 DEBUG: Temperature: {temperature}")

        try:
            # Generate response
            response = self._generate_response(messages, temperature)
            
            # Calculate timing and token usage
            end_time = time.time()
            total_time = end_time - start_time
            
            # Estimate token usage (rough approximation)
            prompt_text = self._format_messages(messages)
            prompt_tokens = len(self.tokenizer.encode(prompt_text))
            completion_tokens = len(self.tokenizer.encode(response))
            total_tokens = prompt_tokens + completion_tokens
            
            # Store usage information
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            # Calculate performance metrics
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            # Print performance stats
            logger.debug(f"🚀 Generated {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
            logger.debug(f"📊 Prompt: {prompt_tokens} tokens | Completion: {completion_tokens} tokens | Total: {total_tokens} tokens")
            logger.debug(f"🌡️  Temperature: {temperature}")
            logger.debug(f"🔧 Device: {self.device}")

            return response

        except Exception as e:
            logger.error(f"❌ Error during inference: {e}")
            return ""
    
    def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context without generating a response - for warm-up purposes"""
        try:
            # Format messages and tokenize to process context
            formatted_prompt = self._format_messages(messages)
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.context_window - 1,  # Leave room for at least 1 token generation
                padding=False
            ).to(self._get_input_device())
            
            # Run a forward pass to process the context
            with torch.no_grad():
                _ = self.model(**inputs)
            
            processed_context = {
                "messages": messages,
                "context_processed": True,
                "timestamp": time.time(),
                "backend": "transformers",
                "model_name": self.model_name
            }
            
            logger.debug(f"🔥 Context processed for {len(messages)} messages")
            return processed_context

        except Exception as e:
            logger.warning(f"⚠️  Error processing context: {e}")
            return {
                "messages": messages,
                "context_processed": False,
                "timestamp": time.time(),
                "backend": "transformers",
                "model_name": self.model_name
            }
    
    def unload(self):
        """Unload the model and clean up resources"""
        # IMPORTANT: Unload vLLM backend first - this kills the EngineCore subprocess
        if hasattr(self, 'vllm_backend') and self.vllm_backend is not None:
            logger.info(f"🔌 Unloading vLLM backend for {self.model_name}...")
            try:
                self.vllm_backend.unload()
            except Exception as e:
                logger.warning(f"⚠️  vLLM backend unload error: {e}")
            self.vllm_backend = None

        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"🔄 Unloaded model: {self.model_name}")
    
    def __del__(self):
        """Clean up on destruction"""
        try:
            self.unload()
        except (RuntimeError, AttributeError):
            pass
