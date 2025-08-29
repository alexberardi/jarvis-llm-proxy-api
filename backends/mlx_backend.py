from mlx_lm.utils import load
from mlx_lm.generate import generate
import os
import time
from typing import List, Dict, Any

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
        
        # Join message content in order, like ChatML
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

        print("⚙️  Generating with MLX...")
        print(f"🌡️  Temperature: {temperature}")
        
        # Generate response with temperature
        response = generate(
            self.model, 
            self.tokenizer, 
            full_prompt.strip(), 
            verbose=False,
            temp=temperature
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
    
    def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context without generating a response - for warm-up purposes"""
        try:
            # Join message content in order, like ChatML
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
            
            # Use minimal generation to process context
            response = generate(
                self.model, 
                self.tokenizer, 
                full_prompt.strip(), 
                verbose=False,
                temp=0.0,  # Deterministic
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