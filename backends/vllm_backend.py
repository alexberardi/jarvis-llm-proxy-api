import multiprocessing
# Fix for vLLM CUDA multiprocessing issue
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from vllm import LLM, SamplingParams
import os
import time
from typing import List, Dict, Any, Union, Optional
from .power_metrics import PowerMetrics

class VLLMClient:
    def __init__(self, model_path: str, chat_format: str, stop_tokens: List[str] = None, context_window: int = None):
        if not model_path:
            raise ValueError("Model path is required")
        
        # Store model name for unload functionality
        self.model_name = model_path
        self.model_path = model_path
        self.chat_format = chat_format
        self.model = None
        self.last_usage = None

        # Initialize power monitoring (optional)
        self.power_metrics = PowerMetrics()
        self.power_metrics.start_monitoring()
        
        # Get context window from parameter or environment variable, default to 4096 (vLLM typical)
        if context_window is None:
            context_window = int(os.getenv("JARVIS_MODEL_CONTEXT_WINDOW", "4096"))
        
        # Get vLLM specific configuration
        tensor_parallel_size = int(os.getenv("JARVIS_VLLM_TENSOR_PARALLEL_SIZE", "1"))
        gpu_memory_utilization = float(os.getenv("JARVIS_VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        max_model_len = context_window
        
        # Batching and scheduling parameters to reduce latency spikes
        max_num_batched_tokens = int(os.getenv("JARVIS_VLLM_MAX_BATCHED_TOKENS", "8192"))
        max_num_seqs = int(os.getenv("JARVIS_VLLM_MAX_NUM_SEQS", "256"))
        
        print(f"🚀 vLLM Debug: Model path: {model_path}")
        print(f"🚀 vLLM Debug: Chat format: {chat_format}")
        print(f"🚀 vLLM Debug: Context window: {context_window}")
        print(f"🚀 vLLM Debug: Tensor parallel size: {tensor_parallel_size}")
        print(f"🚀 vLLM Debug: GPU memory utilization: {gpu_memory_utilization}")
        print(f"🚀 vLLM Debug: Max model length: {max_model_len}")
        print(f"🚀 vLLM Debug: Max batched tokens: {max_num_batched_tokens}")
        print(f"🚀 vLLM Debug: Max sequences: {max_num_seqs}")
        
        # Check if model_path is a local GGUF file
        if model_path.endswith('.gguf') and ('/' in model_path or '\\' in model_path):
            print(f"⚠️  vLLM does not support local GGUF files directly")
            print(f"💡 For GGUF files, use JARVIS_INFERENCE_ENGINE=llama_cpp instead")
            print(f"💡 For vLLM, use HuggingFace model names like: microsoft/Phi-3-mini-4k-instruct")
            raise ValueError(f"vLLM requires HuggingFace model names or converted models, not local GGUF files: {model_path}")

        # Initialize vLLM engine
        try:
            print(f"🚀 Loading vLLM model: {model_path}")
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,  # For custom model architectures
                enforce_eager=False,  # Use CUDA graphs for better performance
                disable_log_stats=True,  # Reduce log noise
                # Batching parameters to reduce latency spikes
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                # Performance optimizations
                enable_prefix_caching=True,
                swap_space=0,  # Disable CPU swap to avoid latency spikes
            )
            
            print(f"✅ vLLM model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load vLLM model: {e}")
            print(f"💡 Make sure the model name is a valid HuggingFace model")
            print(f"💡 Examples: microsoft/Phi-3-mini-4k-instruct, meta-llama/Llama-2-7b-chat-hf")
            raise
        
        # Store stop tokens for sampling
        self.stop_tokens = stop_tokens or []
        
        # Update last usage
        self.last_usage = time.time()

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = 0.7, 
                 top_p: float = 0.9, stop: List[str] = None, stream: bool = False, 
                 response_format: Optional[Dict[str, Any]] = None) -> Union[str, Any]:
        """Generate response using vLLM"""
        
        # Update last usage
        self.last_usage = time.time()
        
        # Convert messages to prompt based on chat format
        prompt = self._messages_to_prompt(messages)
        
        # Prepare sampling parameters
        max_tokens = max_tokens or int(os.getenv("JARVIS_MAX_TOKENS", "2048"))
        stop_tokens = stop or self.stop_tokens or []
        
        # Handle JSON structured output if requested
        guided_json = None
        if response_format and response_format.get("type") == "json_object":
            # vLLM supports guided_json for JSON schema enforcement
            # For basic JSON object, we can use a simple schema
            # More complex schemas can be passed via json_schema in response_format
            if "json_schema" in response_format:
                guided_json = response_format["json_schema"]
            else:
                # Default: allow any JSON object
                guided_json = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                }
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens,
            guided_json=guided_json,  # Add guided JSON if requested
        )
        
        print(f"🚀 vLLM generating with max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        
        try:
            # Generate response
            outputs = self.model.generate([prompt], sampling_params)
            
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
            print(f"❌ vLLM generation error: {e}")
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
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt based on chat format"""
        
        if self.chat_format == "chatml":
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
        """Unload the model to free memory"""
        if self.model:
            # vLLM doesn't have explicit unload, but we can delete the reference
            del self.model
            self.model = None
            print(f"🗑️  vLLM model unloaded: {self.model_name}")
            
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("🧹 CUDA cache cleared")
            except ImportError:
                pass

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
