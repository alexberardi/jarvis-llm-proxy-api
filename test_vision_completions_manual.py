#!/usr/bin/env python3
"""
Manual test script for the vision completions implementation.

Usage:
    python test_vision_completions_manual.py

This script tests the new /v1/chat/completions endpoint with:
1. Text-only messages (string content)
2. Text-only messages (structured content)
3. Multimodal messages (text + images)
4. Model alias resolution
"""

import base64
import os
import sys

# Set up mock environment for testing
os.environ["JARVIS_MODEL_NAME"] = "jarvis-text-8b"
os.environ["JARVIS_LIGHTWEIGHT_MODEL_NAME"] = "jarvis-text-1b"
os.environ["JARVIS_VISION_MODEL_NAME"] = "jarvis-vision-11b"
os.environ["JARVIS_MODEL_BACKEND"] = "MOCK"
os.environ["JARVIS_LIGHTWEIGHT_MODEL_BACKEND"] = "MOCK"
os.environ["JARVIS_VISION_MODEL_BACKEND"] = "MOCK"

# Import after setting env vars
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = client.get("/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    print("✅ PASS: Health check works")


def test_list_models():
    """Test /v1/models endpoint"""
    print("\n" + "="*60)
    print("TEST: List Models")
    print("="*60)
    
    response = client.get("/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    print("✅ PASS: List models works")


def test_text_chat_with_string():
    """Test text-only chat with string content"""
    print("\n" + "="*60)
    print("TEST: Text Chat (String Content)")
    print("="*60)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "full",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "temperature": 0.7,
        },
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "jarvis-text-8b"
    assert len(data["choices"]) == 1
    assert "mock-text" in data["choices"][0]["message"]["content"]
    print("✅ PASS: Text chat with string content works")


def test_text_chat_with_structured():
    """Test text-only chat with structured content"""
    print("\n" + "="*60)
    print("TEST: Text Chat (Structured Content)")
    print("="*60)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lightweight",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Structured hello"}
                    ],
                }
            ],
        },
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "jarvis-text-1b"
    assert "Structured hello" in data["choices"][0]["message"]["content"]
    print("✅ PASS: Text chat with structured content works")


def test_vision_chat_fails_on_text_model():
    """Test that sending images to text-only model fails"""
    print("\n" + "="*60)
    print("TEST: Images Rejected by Text-Only Model")
    print("="*60)
    
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "full",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 400
    data = response.json()
    # FastAPI wraps errors in 'detail'
    error = data.get("detail", {}).get("error") or data.get("error")
    assert error is not None
    assert error["type"] == "invalid_request_error"
    assert "does not support images" in error["message"]
    print("✅ PASS: Text-only model correctly rejects images")


def test_vision_chat_works():
    """Test vision chat with images"""
    print("\n" + "="*60)
    print("TEST: Vision Chat with Images")
    print("="*60)
    
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "jarvis-vision-11b"
    assert "mock-vision" in data["choices"][0]["message"]["content"]
    assert "images=1" in data["choices"][0]["message"]["content"]
    print("✅ PASS: Vision chat with images works")


def test_model_aliases():
    """Test model alias resolution"""
    print("\n" + "="*60)
    print("TEST: Model Alias Resolution")
    print("="*60)
    
    # Test full alias
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "full",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "jarvis-text-8b"
    print("✅ 'full' alias resolves to jarvis-text-8b")
    
    # Test lightweight alias
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lightweight",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "jarvis-text-1b"
    print("✅ 'lightweight' alias resolves to jarvis-text-1b")
    
    # Test vision alias
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vision",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "jarvis-vision-11b"
    print("✅ 'vision' alias resolves to jarvis-vision-11b")
    
    # Test direct model ID
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "jarvis-text-8b",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["model"] == "jarvis-text-8b"
    print("✅ Direct model ID 'jarvis-text-8b' works")
    
    print("✅ PASS: Model aliases work correctly")


def test_nonexistent_model():
    """Test that requesting a non-existent model returns error"""
    print("\n" + "="*60)
    print("TEST: Non-existent Model Error")
    print("="*60)
    
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "non-existent-model",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 404
    data = response.json()
    # FastAPI wraps errors in 'detail'
    error = data.get("detail", {}).get("error") or data.get("error")
    assert error is not None
    assert error["type"] == "model_not_found"
    print("✅ PASS: Non-existent model returns 404")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VISION COMPLETIONS MANUAL TEST SUITE")
    print("="*60)
    
    tests = [
        test_health,
        test_list_models,
        test_text_chat_with_string,
        test_text_chat_with_structured,
        test_vision_chat_fails_on_text_model,
        test_vision_chat_works,
        test_model_aliases,
        test_nonexistent_model,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

