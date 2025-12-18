import io
import time
from typing import Any, Dict, List, Tuple

from PIL import Image
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from managers.chat_types import ChatResult, GenerationParams, ImagePart, NormalizedMessage, TextPart

class MlxClient:
    def __init__(self, model_path: str):
        print("🔁 Loading MLX model...")
        self.model_name = model_path
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        self.last_usage = None

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Chat method with temperature support"""
        return self.chat_with_temperature(messages, temperature)
    
    def chat_with_temperature(self, messages: list[dict], temperature: float = 0.7) -> str:
        # Start timing
        start_time = time.time()
        
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            # Convert messages to the format expected by apply_chat_template
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Apply chat template
            full_prompt = self.tokenizer.apply_chat_template(
                formatted_messages, 
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            # Fallback to simple format if no chat template
            full_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    full_prompt += f"[SYSTEM] {content}\n"
                elif role == "user":
                    full_prompt += f"[USER] {content}\n"
                elif role == "assistant":
                    full_prompt += f"[ASSISTANT] {content}\n"
            full_prompt = full_prompt.strip()

        print("⚙️  Generating with MLX...")
        print(f"🌡️  Temperature: {temperature}")
        print(f"🔍 Debug: Using chat template: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}")
        
        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)
        
        # Generate response with sampler
        response = generate(
            self.model,
            self.tokenizer,
            full_prompt,
            verbose=False,
            sampler=sampler,
        )
        
        # Calculate timing
        end_time = time.time()
        total_time = end_time - start_time
        
        # Estimate token usage (rough approximation)
        # This is a simplified token count - MLX doesn't provide exact token counts
        prompt_tokens = len(full_prompt.split())  # Rough word count
        completion_tokens = len(response.split())  # Rough word count
        total_tokens = prompt_tokens + completion_tokens
        
        # Store usage information for OpenAI-style response
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        
        # Print performance metrics
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        print(f"🚀 Generated ~{completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        print(f"📊 Prompt: ~{prompt_tokens} tokens | Completion: ~{completion_tokens} tokens | Total: ~{total_tokens} tokens")
        
        return response.strip()

    async def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        formatted = self._to_dict_messages(messages)
        content = self.chat_with_temperature(formatted, params.temperature)
        return ChatResult(content=content, usage=self.last_usage)

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
        print(f"🚀 [MLX vision] ~{completion_tokens} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tok/s)")

        return ChatResult(content=response.strip(), usage=self.last_usage)
    
    def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context without generating a response - for warm-up purposes"""
        try:
            # Use the tokenizer's chat template if available
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                # Convert messages to the format expected by apply_chat_template
                formatted_messages = []
                for msg in messages:
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Apply chat template
                full_prompt = self.tokenizer.apply_chat_template(
                    formatted_messages, 
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Fallback to simple format if no chat template
                full_prompt = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        full_prompt += f"[SYSTEM] {content}\n"
                    elif role == "user":
                        full_prompt += f"[USER] {content}\n"
                    elif role == "assistant":
                        full_prompt += f"[ASSISTANT] {content}\n"
                full_prompt = full_prompt.strip()
            
            # Create deterministic sampler for context processing
            sampler = make_sampler(temp=0.0)
            
            # Use minimal generation to process context
            response = generate(
                self.model, 
                self.tokenizer, 
                full_prompt, 
                verbose=False,
                sampler=sampler,
                max_tokens=1  # Minimal tokens
            )
            
            # Extract the internal context representation
            processed_context = {
                "messages": messages,
                "context_processed": True,
                "timestamp": time.time()
            }
            
            print(f"🔥 Context processed for {len(messages)} messages")
            return processed_context
            
        except Exception as e:
            print(f"⚠️  Error processing context: {e}")
            # Fallback to storing raw messages
            return {
                "messages": messages,
                "context_processed": False,
                "timestamp": time.time()
            }
    
    def unload(self):
        """Unload the model and clean up resources"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        print(f"🔄 Unloaded model: {self.model_name}")

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