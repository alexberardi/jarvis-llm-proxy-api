#!/usr/bin/env python3
"""
Test script for the Transformers backend
Run this after installing the transformers dependencies to validate the backend works.
"""

import os
import sys

def test_transformers_backend():
    """Test the transformers backend with a small model"""
    print("🧪 Testing Transformers Backend...")
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from backends.transformers_backend import TransformersClient
        print("✅ TransformersClient imported successfully")
        
        # Set environment variables for testing
        os.environ["JARVIS_DEVICE"] = "auto"  # Use best available device
        os.environ["JARVIS_TORCH_DTYPE"] = "auto"
        os.environ["JARVIS_MAX_TOKENS"] = "100"
        os.environ["JARVIS_USE_QUANTIZATION"] = "false"
        os.environ["JARVIS_DO_SAMPLE"] = "false"  # Use greedy decoding for testing
        os.environ["JARVIS_MODEL_CONTEXT_WINDOW"] = "1024"  # Conservative context window
        
        # Test with jarvis-diablo model or fallback to small model
        model_name = "alexberardi/jarvis-diablo"  # The model you want to test
        print(f"🤖 Testing with model: {model_name}")
        print("⚠️  Note: This will download the model if not cached locally")
        
        # Initialize client
        print("🔄 Initializing TransformersClient...")
        client = TransformersClient(
            model_path=model_name,
            chat_format="generic",
            context_window=512
        )
        
        # Test chat
        print("💬 Testing chat functionality...")
        messages = [
            {"role": "user", "content": "Hello! How are you?"}
        ]
        
        response = client.chat(messages, temperature=0.7)
        print(f"🤖 Response: {response}")
        
        # Test unload
        print("🔄 Testing unload...")
        client.unload()
        print("✅ Unload successful")
        
        print("🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install dependencies with: pip install transformers torch accelerate")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_transformers_backend()
    sys.exit(0 if success else 1)
