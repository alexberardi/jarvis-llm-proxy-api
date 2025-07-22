from mlx_lm.utils import load
from mlx_lm.generate import generate
import os

class MlxClient:
    def __init__(self, model_path: str):
        print("🔁 Loading MLX model...")
        self.model_name = model_path
        self.model, self.tokenizer = load(model_path)

    def chat(self, messages: list[dict]) -> str:
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
        response = generate(self.model, self.tokenizer, full_prompt.strip(), verbose=False)
        return response.strip()
    
    def unload(self):
        """Unload the model and clean up resources"""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        print(f"🔄 Unloaded model: {self.model_name}")