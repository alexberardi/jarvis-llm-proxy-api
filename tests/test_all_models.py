#!/usr/bin/env python3
"""
Test script that loops through all models and tests them with basic requests.
"""

import requests
import json
import time
import sys
import os
from typing import Dict, List, Any

# Model configurations from model_reference.md
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
        "name": "Yi 1.5 9B Chat",
        "file_path": ".models/yi-1.5-9b/Yi-1.5-9B-Chat-Q4_K_M.gguf",
        "chat_format": "chatml",
        "context_window": 8192
    },
    {
        "name": "Llama 3.2 3B Instruct",
        "file_path": ".models/Llama-3.2-3B-Instruct-Q4_K_M_old.gguf",
        "chat_format": "chatml",
        "context_window": 8192
    }
]

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")

def test_model_swap(model_config: Dict[str, Any]) -> bool:
    """Test swapping to a specific model."""
    try:
        print(f"🔄 Swapping to {model_config['name']}...")
        
        swap_data = {
            "new_model": model_config["file_path"],
            "new_model_backend": "GGUF",
            "new_model_chat_format": model_config["chat_format"],
            "new_model_context_window": model_config["context_window"]
        }
        
        response = requests.post(f"{BASE_URL}/model-swap", json=swap_data)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"✅ Successfully swapped to {model_config['name']}")
                return True
            else:
                print(f"❌ Model swap failed: {result.get('message', 'Unknown error')}")
                return False
        else:
            print(f"❌ Model swap HTTP error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model swap exception: {e}")
        return False

def test_chat_request(messages: List[Dict[str, str]], test_name: str) -> Dict[str, Any]:
    """Test a chat request and return the response."""
    try:
        print(f"📤 Sending {test_name} request...")
        
        response = requests.post(f"{BASE_URL}/chat", json={"messages": messages})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {test_name} response received")
            return {"success": True, "response": result.get("response", ""), "error": None}
        else:
            print(f"❌ {test_name} HTTP error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {error_detail}")
            except:
                print(f"   Error text: {response.text[:200]}...")
            return {"success": False, "response": None, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ {test_name} exception: {e}")
        return {"success": False, "response": None, "error": str(e)}

def test_json_extraction(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Test JSON extraction with a specific prompt."""
    json_prompt = [
        {
            "role": "user",
            "content": "Return ONLY a valid JSON object with name and age fields. Do not include any markdown formatting, code blocks, or explanatory text. Just the raw JSON object."
        }
    ]
    
    return test_chat_request(json_prompt, "JSON extraction")

def test_hello_request() -> Dict[str, Any]:
    """Test a simple hello request."""
    hello_prompt = [
        {
            "role": "user", 
            "content": "Say hello in a friendly way."
        }
    ]
    
    return test_chat_request(hello_prompt, "Hello")

def validate_json_response(response_text: str) -> bool:
    """Validate if the response is valid JSON."""
    try:
        # Remove any leading/trailing whitespace
        cleaned = response_text.strip()
        
        # Try to parse as JSON
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def main():
    """Main test function."""
    print("🚀 Starting model test loop...")
    print(f"📋 Testing {len(MODELS)} models")
    print("=" * 80)
    
    results = []
    
    for i, model_config in enumerate(MODELS, 1):
        print(f"\n🔍 Model {i}/{len(MODELS)}: {model_config['name']}")
        print("-" * 60)
        
        # Test model swap
        swap_success = test_model_swap(model_config)
        
        if not swap_success:
            print(f"⏭️  Skipping {model_config['name']} due to swap failure")
            results.append({
                "model": model_config["name"],
                "swap_success": False,
                "hello_success": False,
                "json_success": False,
                "json_valid": False,
                "hello_response": None,
                "json_response": None,
                "errors": ["Model swap failed"]
            })
            continue
        
        # Wait a moment for model to load
        time.sleep(2)
        
        # Test hello request
        hello_result = test_hello_request()
        
        # Test JSON extraction
        json_result = test_json_extraction(model_config)
        
        # Validate JSON response
        json_valid = False
        if json_result["success"] and json_result["response"]:
            json_valid = validate_json_response(json_result["response"])
            if json_valid:
                print("✅ JSON response is valid")
            else:
                print("❌ JSON response is invalid")
        
        # Store results
        results.append({
            "model": model_config["name"],
            "swap_success": True,
            "hello_success": hello_result["success"],
            "json_success": json_result["success"],
            "json_valid": json_valid,
            "hello_response": hello_result["response"],
            "json_response": json_result["response"],
            "errors": []
        })
        
        # Add errors if any
        if hello_result["error"]:
            results[-1]["errors"].append(f"Hello error: {hello_result['error']}")
        if json_result["error"]:
            results[-1]["errors"].append(f"JSON error: {json_result['error']}")
        
        # Wait before next model
        if i < len(MODELS):
            print("⏳ Waiting 3 seconds before next model...")
            time.sleep(3)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    successful_swaps = sum(1 for r in results if r["swap_success"])
    successful_hello = sum(1 for r in results if r["hello_success"])
    successful_json = sum(1 for r in results if r["json_success"])
    valid_json = sum(1 for r in results if r["json_valid"])
    
    print(f"✅ Successful model swaps: {successful_swaps}/{len(MODELS)}")
    print(f"✅ Successful hello requests: {successful_hello}/{len(MODELS)}")
    print(f"✅ Successful JSON requests: {successful_json}/{len(MODELS)}")
    print(f"✅ Valid JSON responses: {valid_json}/{len(MODELS)}")
    
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        status_emoji = "✅" if result["swap_success"] else "❌"
        print(f"{status_emoji} {result['model']}")
        
        if result["swap_success"]:
            hello_status = "✅" if result["hello_success"] else "❌"
            json_status = "✅" if result["json_success"] else "❌"
            json_valid_status = "✅" if result["json_valid"] else "❌"
            
            print(f"   Hello: {hello_status} | JSON: {json_status} | Valid JSON: {json_valid_status}")
            
            if result["json_response"]:
                print(f"   JSON Response: {result['json_response'][:100]}...")
        else:
            print(f"   Errors: {', '.join(result['errors'])}")
    
    # Save results to file
    with open("model_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to model_test_results.json")

if __name__ == "__main__":
    main() 