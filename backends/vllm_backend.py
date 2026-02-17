import logging
import multiprocessing

# Fix for vLLM CUDA multiprocessing issue
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams
import inspect
import os
import time
from typing import List, Dict, Any, Union, Optional
from .power_metrics import PowerMetrics
from services import adapter_cache
from managers.chat_types import NormalizedMessage, TextPart, GenerationParams, ChatResult
from backends.base import LLMBackendBase
from services.settings_helpers import (
    get_bool_setting,
    get_float_setting,
    get_int_setting,
    get_setting,
)

logger = logging.getLogger("uvicorn")


class VLLMClient(LLMBackendBase):
    def __init__(self, model_path: str, chat_format: str, stop_tokens: List[str] = None, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")
        
        # Store model name for unload functionality
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = (chat_format or "").lower()
        self.model = None
        self.last_usage = None
        self._lora_enabled = False
        self.inference_engine = "vllm"  # For /v1/engine endpoint

        # Initialize power monitoring (optional)
        self.power_metrics = PowerMetrics()
        self.power_metrics.start_monitoring()
        
        # Get context window from parameter or environment variable, default to 4096 (vLLM typical)
        if context_window is None:
            context_window = get_int_setting(
                "model.main.context_window", "JARVIS_MODEL_CONTEXT_WINDOW", 4096
            )
        
        # Get vLLM specific configuration
        tensor_parallel_size = get_int_setting(
            "inference.vllm.tensor_parallel_size", "JARVIS_VLLM_TENSOR_PARALLEL_SIZE", 1
        )
        gpu_memory_utilization = get_float_setting(
            "inference.vllm.gpu_memory_utilization", "JARVIS_VLLM_GPU_MEMORY_UTILIZATION", 0.9
        )
        max_model_len = context_window
        vllm_quantization = get_setting(
            "inference.vllm.quantization", "JARVIS_VLLM_QUANTIZATION", ""
        ).strip().lower()
        # Normalize common variants (e.g., awq-marlin -> awq_marlin)
        vllm_quantization = vllm_quantization.replace("-", "_")
        if vllm_quantization in {"", "none", "false", "off", "auto"}:
            vllm_quantization = None
        
        # Batching and scheduling parameters to reduce latency spikes
        max_num_batched_tokens = get_int_setting(
            "inference.vllm.max_batched_tokens", "JARVIS_VLLM_MAX_BATCHED_TOKENS", 8192
        )
        max_num_seqs = get_int_setting(
            "inference.vllm.max_num_seqs", "JARVIS_VLLM_MAX_NUM_SEQS", 256
        )
        enforce_eager = get_bool_setting(
            "inference.vllm.enforce_eager", "JARVIS_VLLM_ENFORCE_EAGER", True
        )
        
        logger.debug(f"🚀 vLLM Debug: Model path: {model_path}")
        logger.debug(f"🚀 vLLM Debug: Chat format: {chat_format}")
        logger.debug(f"🚀 vLLM Debug: Context window: {context_window}")
        logger.debug(f"🚀 vLLM Debug: Tensor parallel size: {tensor_parallel_size}")
        logger.debug(f"🚀 vLLM Debug: GPU memory utilization: {gpu_memory_utilization}")
        logger.debug(f"🚀 vLLM Debug: Max model length: {max_model_len}")
        logger.debug(f"🚀 vLLM Debug: Max batched tokens: {max_num_batched_tokens}")
        logger.debug(f"🚀 vLLM Debug: Max sequences: {max_num_seqs}")
        logger.debug(f"🚀 vLLM Debug: Quantization: {vllm_quantization or 'none'}")
        
        # Check if model_path is a local GGUF file
        if model_path.endswith('.gguf') and ('/' in model_path or '\\' in model_path):
            logger.warning(f"⚠️  vLLM does not support local GGUF files directly")
            logger.info(f"💡 For GGUF files, use JARVIS_INFERENCE_ENGINE=llama_cpp instead")
            logger.info(f"💡 For vLLM, use HuggingFace model names like: microsoft/Phi-3-mini-4k-instruct")
            raise ValueError(f"vLLM requires HuggingFace model names or converted models, not local GGUF files: {model_path}")

        # Initialize vLLM engine
        try:
            logger.info(f"🚀 Loading vLLM model: {model_path}")
            llm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "trust_remote_code": True,  # For custom model architectures
                "enforce_eager": enforce_eager,  # True = skip CUDA graphs (saves VRAM, slower); False = compile graphs (faster, needs more VRAM)
                "disable_log_stats": True,  # Reduce log noise
                # Batching parameters to reduce latency spikes
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
                # Performance optimizations
                "enable_prefix_caching": True,
                "swap_space": 0,  # Disable CPU swap to avoid latency spikes
            }
            # Enable LoRA for dynamic per-request adapter loading
            max_lora_rank = get_int_setting(
                "inference.vllm.max_lora_rank", "JARVIS_VLLM_MAX_LORA_RANK", 64
            )
            max_loras = get_int_setting(
                "inference.vllm.max_loras", "JARVIS_VLLM_MAX_LORAS", 1
            )
            if max_loras > 0:
                # vLLM 0.14+ passes enable_lora through kwargs to EngineArgs
                llm_kwargs["enable_lora"] = True
                self._lora_enabled = True
                llm_kwargs["max_lora_rank"] = max_lora_rank
                llm_kwargs["max_loras"] = max_loras
                logger.info(f"🧩 vLLM LoRA enabled: max_lora_rank={max_lora_rank} max_loras={max_loras}")
            if vllm_quantization is not None:
                llm_kwargs["quantization"] = vllm_quantization

            self.model = LLM(
                **llm_kwargs,
            )
            
            logger.info(f"✅ vLLM model loaded successfully!")
            try:
                engine = getattr(self.model, "llm_engine", None)
                model_config = getattr(engine, "model_config", None) if engine else None
                if model_config:
                    logger.debug(f"🚀 vLLM Debug: model_config.quantization={getattr(model_config, 'quantization', None)}")
                    logger.debug(f"🚀 vLLM Debug: model_config.dtype={getattr(model_config, 'dtype', None)}")
                    hf_config = getattr(model_config, "hf_config", None)
                    if hf_config and getattr(hf_config, "quantization_config", None):
                        logger.debug(f"🚀 vLLM Debug: hf_config.quantization_config={hf_config.quantization_config}")
            except Exception as e:
                logger.warning(f"⚠️  vLLM Debug: Failed to read model config: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load vLLM model: {e}")
            logger.info(f"💡 Make sure the model name is a valid HuggingFace model")
            logger.info(f"💡 Examples: microsoft/Phi-3-mini-4k-instruct, meta-llama/Llama-2-7b-chat-hf")
            raise
        
        # Store stop tokens for sampling
        if isinstance(stop_tokens, str):
            stop_tokens = [t.strip() for t in stop_tokens.split(",") if t.strip()]
        self.stop_tokens = stop_tokens or []
        if not self.stop_tokens:
            # Default stop tokens for common chat formats to prevent run-on output
            if self.chat_format in {"chatml", "qwen"}:
                self.stop_tokens = ["<|im_end|>"]
            elif self.chat_format in {"llama3"}:
                self.stop_tokens = ["<|eot_id|>"]
        
        # Update last usage
        self.last_usage = time.time()

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = 0.7,
                 top_p: float = 0.9, stop: List[str] = None, stream: bool = False,
                 response_format: Optional[Dict[str, Any]] = None,
                 lora_request: Optional[LoRARequest] = None) -> Union[str, Any]:
        """Generate response using vLLM

        Args:
            lora_request: Optional per-request LoRA adapter. If None, uses static adapter if configured.
        """
        
        # Update last usage
        self.last_usage = time.time()
        
        # Convert messages to prompt based on chat format
        prompt = self._messages_to_prompt(messages)
        
        # Prepare sampling parameters
        max_tokens = max_tokens or get_int_setting(
            "inference.general.max_tokens", "JARVIS_MAX_TOKENS", 2048
        )
        stop_tokens = stop or self.stop_tokens or []
        
        # Handle JSON structured output if requested
        structured_outputs = None
        if response_format and response_format.get("type") == "json_object":
            # vLLM uses StructuredOutputsParams for JSON schema enforcement
            if "json_schema" in response_format:
                json_schema = response_format["json_schema"]
                structured_outputs = StructuredOutputsParams(json=json_schema)
                logger.info(f"🔒 Enforcing JSON schema with {len(json_schema.get('required', []))} required fields")
            else:
                # Default: just require valid JSON object (no specific schema)
                structured_outputs = StructuredOutputsParams(json_object=True)
                logger.info("🔒 Enforcing valid JSON object (no schema)")

        sampling_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop_tokens,
        }
        if structured_outputs is not None:
            sampling_kwargs["structured_outputs"] = structured_outputs
        sampling_params = SamplingParams(**sampling_kwargs)
        
        logger.debug(f"🚀 vLLM generating with max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        
        try:
            # Generate response
            generate_kwargs = {}
            # Use per-request lora_request if provided
            if lora_request:
                if "lora_request" in inspect.signature(self.model.generate).parameters:
                    generate_kwargs["lora_request"] = lora_request
                    logger.info(f"🧩 Adapter active (vLLM): {lora_request.lora_path}")
                else:
                    logger.warning("⚠️  vLLM generate() does not support lora_request; ignoring adapter")

            outputs = self.model.generate([prompt], sampling_params, **generate_kwargs)
            
            if not outputs:
                raise ValueError("No output generated")
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            # Calculate token usage (approximate)
            prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
            completion_tokens = len(output.outputs[0].token_ids) if output.outputs[0].token_ids else 0
            total_tokens = prompt_tokens + completion_tokens
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            
            return generated_text, usage
            
        except Exception as e:
            logger.error(f"❌ vLLM generation error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = None) -> str:
        """Chat method for compatibility with other backends"""
        generated_text, usage = self.generate(messages, max_tokens=max_tokens)
        self.last_usage = usage  # Store usage info
        return generated_text
    
    def chat_with_temperature(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = None) -> str:
        """Chat method with temperature for compatibility with other backends"""
        generated_text, usage = self.generate(messages, max_tokens=max_tokens, temperature=temperature)
        self.last_usage = usage  # Store usage info
        return generated_text

    def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """Generate text chat response using vLLM with GenerationParams.

        This method follows the modern backend interface pattern and supports
        per-request adapter settings via params.adapter_settings.
        """
        # Convert NormalizedMessage to legacy dict format
        legacy_messages = []
        for msg in messages:
            text_content = " ".join(
                part.text for part in msg.content if isinstance(part, TextPart)
            )
            legacy_messages.append({"role": msg.role, "content": text_content})

        # Resolve per-request adapter if specified
        per_request_lora: Optional[LoRARequest] = None
        if params.adapter_settings and self._lora_enabled:
            adapter_hash = params.adapter_settings.get("hash")
            adapter_scale = params.adapter_settings.get("scale", 1.0)
            adapter_enabled = params.adapter_settings.get("enabled", True)

            if adapter_hash and adapter_enabled:
                # Resolve adapter path from cache (downloads from S3 if needed)
                adapter_path = adapter_cache.get_adapter_path(adapter_hash)
                if adapter_path:
                    # Create unique LoRARequest for this adapter
                    # Use hash as both name and part of ID for uniqueness
                    lora_id = hash(adapter_hash) % (2**31)  # Positive int32
                    per_request_lora = LoRARequest(
                        lora_name=f"adapter-{adapter_hash[:8]}",
                        lora_int_id=lora_id,
                        lora_path=str(adapter_path),
                    )
                    logger.info(f"🧩 [vLLM] Per-request adapter: hash={adapter_hash[:8]}, path={adapter_path}")
                else:
                    logger.warning(f"⚠️  [vLLM] Adapter not found: hash={adapter_hash}")
            elif adapter_hash and not adapter_enabled:
                logger.debug(f"🧩 [vLLM] Adapter disabled by request: hash={adapter_hash}")
        elif params.adapter_settings and not self._lora_enabled:
            logger.warning(f"⚠️  [vLLM] Adapter requested but LoRA not enabled (max_loras=0?)")

        # Generate using existing method
        response_text, usage = self.generate(
            messages=legacy_messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            response_format=params.response_format,
            lora_request=per_request_lora,
        )

        return ChatResult(content=response_text, usage=usage)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt based on chat format"""
        
        if self.chat_format in {"chatml", "qwen"}:
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            
        elif self.chat_format == "llama3":
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt += f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>\n"
                elif role == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>\n"
                elif role == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>\n"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            
        elif self.chat_format == "mistral":
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    prompt += f"[INST] {content} [/INST]"
                elif role == "assistant":
                    prompt += f" {content}</s>"
                elif role == "system":
                    prompt = f"[INST] {content}\n\n" + prompt
            
        else:
            # Default: simple concatenation
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant:"
        
        return prompt

    def unload(self):
        """Unload the model to free memory - kills vLLM child processes"""
        import gc
        import time
        import os
        import signal
        
        if self.model:
            # Get the current process to find child processes
            current_pid = os.getpid()
            child_pids = []
            
            # Try to find vLLM child processes before shutting down
            try:
                import psutil
                current_proc = psutil.Process(current_pid)
                # Get all descendant processes (children and their children)
                for child in current_proc.children(recursive=True):
                    try:
                        # Check if it's a vLLM EngineCore process
                        if 'python' in child.name().lower() or 'vllm' in ' '.join(child.cmdline()).lower():
                            child_pids.append(child.pid)
                            logger.debug(f"🔍 Found vLLM child process: PID {child.pid}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                logger.warning("⚠️  psutil not available, will use basic cleanup")
            except Exception as e:
                logger.warning(f"⚠️  Error finding child processes: {e}")
            
            # Attempt to gracefully shutdown vLLM engine if available
            try:
                engine = getattr(self.model, "llm_engine", None)
                if engine:
                    # Try engine_core shutdown first (V1 API)
                    engine_core = getattr(engine, "engine_core", None)
                    if engine_core:
                        # Try shutdown method
                        if hasattr(engine_core, "shutdown"):
                            engine_core.shutdown()
                            logger.info("🧹 vLLM engine_core shutdown complete")
                        # Try close method
                        if hasattr(engine_core, "close"):
                            engine_core.close()
                            logger.info("🧹 vLLM engine_core close complete")
                    # Then try engine shutdown
                    if hasattr(engine, "shutdown"):
                        engine.shutdown()
                        logger.info("🧹 vLLM engine shutdown complete")
            except Exception as e:
                logger.warning(f"⚠️  vLLM engine shutdown failed: {e}")

            # Delete model reference
            model_ref = self.model
            self.model = None
            del model_ref
            logger.info(f"🗑️  vLLM model reference deleted: {self.model_name}")
            
            # Force garbage collection to trigger __del__ on vLLM objects
            gc.collect()
            
            # Wait briefly for processes to start dying
            time.sleep(1)
            
            # Force kill any remaining vLLM child processes
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.debug(f"🔪 Sent SIGTERM to vLLM child PID {pid}")
                except OSError:
                    pass  # Process already dead

            # Give processes time to terminate gracefully
            time.sleep(2)

            # SIGKILL any stubborn processes
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.debug(f"💀 Sent SIGKILL to vLLM child PID {pid}")
                except OSError:
                    pass  # Process already dead
            
            # Final GC and cache clear
            gc.collect()

            # Clear CUDA cache multiple times with delay
            try:
                import torch
                import torch.distributed as dist

                # Destroy any distributed process groups (NCCL cleanup)
                # This is critical - vLLM creates process groups that must be destroyed
                # before reinitializing, otherwise "Engine core initialization failed"
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

                    # Check memory
                    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                    total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
                    logger.info(f"🧹 CUDA cache cleared. Free: {free_mem:.2f}/{total_mem:.2f} GiB")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"⚠️  CUDA cleanup error: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "backend": "vLLM",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "chat_format": self.chat_format,
            "last_usage": self.last_usage,
            "engine": "vLLM High-Performance Inference"
        }
