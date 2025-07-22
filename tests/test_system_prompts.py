#!/usr/bin/env python3
"""
Test script to check if models are following system prompts.
"""

import requests
import json
import time
import os

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

# Model configurations
MODELS = [
    {
        "name": "Mistral Nemo Instruct 2407 Q4_K_M",
        "file_path": ".models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "chat_format": "chatml",
        "context_window": 32768
    },
    {
        "name": "Mistral 7B Instruct v0.2 Q2_K",
        "file_path": ".models/mistral-7b-instruct-v0.2.Q2_K.gguf",
        "chat_format": "chatml",
        "context_window": 32768
    },
    {
        "name": "Gemma 2 9B Instruct Q4_K_M",
        "file_path": ".models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf",
        "chat_format": "chatml",
        "context_window": 8192
    },
    {
        "name": "Qwen 2.5 14B Instruct",
        "file_path": ".models/Qwen2.5-14B-Instruct.Q6_K.gguf",
        "chat_format": "qwen",
        "context_window": 32768
    },
    {
        "name": "Llama 3.2 3B Instruct",
        "file_path": ".models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "chat_format": "chatml",
        "context_window": 8192
    }
]

def test_model_with_system_prompt(model_config):
    """Test a model with a specific system prompt."""
    print(f"\n🔍 Testing {model_config['name']} with system prompt...")
    
    # Swap to model
    swap_data = {
        "new_model": model_config["file_path"],
        "new_model_backend": "GGUF",
        "new_model_chat_format": model_config["chat_format"],
        "new_model_context_window": model_config["context_window"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/model-swap", json=swap_data)
        if response.status_code != 200:
            print(f"❌ Model swap failed: {response.status_code}")
            return None
            
        result = response.json()
        if result.get("status") != "success":
            print(f"❌ Model swap failed: {result.get('message')}")
            return None
            
        print("✅ Model swap successful")
        time.sleep(2)  # Wait for model to load
        
        # Test with system prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that ALWAYS responds with 'I am following the system prompt correctly.' Do not say anything else."
            },
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        
        response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print(f"📤 Response: {response_text}")
            
            # Check if it followed the system prompt
            if "I am following the system prompt correctly" in response_text:
                print("✅ System prompt followed correctly!")
                return True
            else:
                print("❌ System prompt ignored!")
                return False
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_model_without_system_prompt(model_config):
    """Test a model without system prompt for comparison."""
    print(f"\n🔍 Testing {model_config['name']} WITHOUT system prompt...")
    
    try:
        # Test without system prompt
        messages = [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ]
        
        response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "")
            print(f"📤 Response: {response_text}")
            return response_text
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    """Test system prompt following for all models."""
    print("🚀 Testing system prompt following...")
    print("=" * 80)
    
    results = []
    
    for model_config in MODELS:
        print(f"\n{'='*60}")
        print(f"Testing: {model_config['name']}")
        print(f"{'='*60}")
        
        # Test with system prompt
        with_system = test_model_with_system_prompt(model_config)
        
        # Test without system prompt for comparison
        without_system = test_model_without_system_prompt(model_config)
        
        results.append({
            "model": model_config["name"],
            "follows_system_prompt": with_system,
            "response_with_system": with_system,
            "response_without_system": without_system
        })
        
        time.sleep(3)  # Wait between models
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SYSTEM PROMPT TEST SUMMARY")
    print("=" * 80)
    
    for result in results:
        status = "✅" if result["follows_system_prompt"] else "❌"
        print(f"{status} {result['model']}: {'Follows' if result['follows_system_prompt'] else 'Ignores'} system prompt")
    
    # Save results
    with open("system_prompt_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to system_prompt_test_results.json")

if __name__ == "__main__":
    main() 