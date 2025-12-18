import httpx
import os
import time
import json
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

from managers.chat_types import ChatResult, GenerationParams, ImagePart, NormalizedMessage, TextPart

class RestClient:
    def __init__(self, base_url: str, model_name: str = "jarvis-llm", model_type: str = "main"):
        """
        Initialize REST client for remote API providers
        
        Args:
            base_url: Base URL for the API (e.g., http://localhost:11434, https://api.openai.com)
            model_name: Model name to use for requests
            model_type: Type of model ("main" or "lightweight")
        """
        self.base_url = base_url.rstrip('/')
        self.model_type = model_type
        
        # Allow environment variable override of model name based on model type
        if model_type == "lightweight":
            env_model_name = os.getenv("JARVIS_REST_LIGHTWEIGHT_MODEL_NAME")
        else:
            env_model_name = os.getenv("JARVIS_REST_MODEL_NAME")
            
        if env_model_name:
            self.model_name = env_model_name
        else:
            self.model_name = model_name
            
        self.last_usage = None
        
        # Get authentication configuration from environment
        self.auth_type = os.getenv("JARVIS_REST_AUTH_TYPE", "none").lower()
        self.auth_token = os.getenv("JARVIS_REST_AUTH_TOKEN", "")
        self.auth_header_name = os.getenv("JARVIS_REST_AUTH_HEADER", "Authorization")
        
        # Get provider-specific configuration
        self.provider = os.getenv("JARVIS_REST_PROVIDER", "generic").lower()
        
        # Get request format configuration
        self.request_format = os.getenv("JARVIS_REST_REQUEST_FORMAT", "openai").lower()
        
        # Get timeout configuration
        self.timeout = int(os.getenv("JARVIS_REST_TIMEOUT", "60"))
        
        print(f"🌐 Initialized REST backend for {self.provider}")
        print(f"🔗 Base URL: {self.base_url}")
        print(f"🔑 Auth type: {self.auth_type}")
        print(f"📝 Request format: {self.request_format}")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        # Set up headers
        self.headers = self._setup_headers()
    
    def _setup_headers(self) -> Dict[str, str]:
        """Set up headers based on authentication type and provider"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Jarvis-LLM-Proxy/1.0"
        }
        
        if self.auth_type == "bearer" and self.auth_token:
            headers[self.auth_header_name] = f"Bearer {self.auth_token}"
        elif self.auth_type == "api_key" and self.auth_token:
            headers[self.auth_header_name] = self.auth_token
        elif self.auth_type == "custom" and self.auth_token:
            # For custom auth patterns, expect JARVIS_REST_AUTH_HEADER to be set
            headers[self.auth_header_name] = self.auth_token
        
        return headers
    
    def _format_messages_for_provider(self, messages: List[Dict[str, str]]) -> Any:
        """Format messages according to provider requirements"""
        if self.request_format == "openai":
            # OpenAI format: {"messages": [...], "model": "...", "temperature": ...}
            return {
                "messages": messages,
                "model": self.model_name
            }
        elif self.request_format == "ollama":
            # Ollama format: {"messages": [...], "model": "..."}
            return {
                "messages": messages,
                "model": self.model_name
            }
        elif self.request_format == "chatml":
            # ChatML format: concatenated text with role prefixes
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
            return {"prompt": formatted}
        else:
            # Generic format: pass through as-is
            return {"messages": messages}
    
    def _get_endpoint_for_provider(self) -> str:
        """Get the appropriate endpoint for the provider"""
        if self.provider == "openai":
            return "/v1/chat/completions"
        elif self.provider == "anthropic":
            return "/v1/messages"
        elif self.provider == "ollama":
            return "/api/chat"
        elif self.provider == "lmstudio":
            return "/v1/chat/completions"
        else:
            # Generic endpoint
            return "/v1/chat/completions"
    
    def _parse_response_for_provider(self, response_data: Dict[str, Any]) -> str:
        """Parse response according to provider format"""
        if self.provider == "openai" or self.provider == "lmstudio":
            # OpenAI format
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                # Store usage information if available
                if "usage" in response_data:
                    self.last_usage = response_data["usage"]
                return content
        elif self.provider == "anthropic":
            # Anthropic format
            if "content" in response_data and len(response_data["content"]) > 0:
                content = response_data["content"][0].get("text", "")
                # Store usage information if available
                if "usage" in response_data:
                    self.last_usage = response_data["usage"]
                return content
        elif self.provider == "ollama":
            # Ollama format
            if "message" in response_data:
                content = response_data["message"].get("content", "")
                # Ollama doesn't provide usage info, so we'll estimate
                self._estimate_usage(content)
                return content
        
        # Generic fallback
        if "content" in response_data:
            return response_data["content"]
        elif "text" in response_data:
            return response_data["text"]
        elif "response" in response_data:
            return response_data["response"]
        else:
            # Last resort: return the whole response as string
            return str(response_data)

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send an OpenAI-style chat completion request. Supports structured content.

        The method intentionally keeps streaming disabled for now; callers can
        choose to stream responses at a higher layer using the returned content.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        endpoint = self._get_endpoint_for_provider()
        url = urljoin(self.base_url, endpoint)

        response = await self.client.post(url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def _estimate_usage(self, content: str):
        """Estimate token usage when provider doesn't provide it"""
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(content) // 4
        self.last_usage = {
            "prompt_tokens": 0,  # We don't know the prompt tokens
            "completion_tokens": estimated_tokens,
            "total_tokens": estimated_tokens
        }
    
    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Chat method with temperature support"""
        return await self.chat_with_temperature(messages, temperature)
    
    async def chat_with_temperature(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Send chat request with temperature control"""
        start_time = time.time()
        
        # Format messages for the provider
        formatted_data = self._format_messages_for_provider(messages)
        
        # Add temperature and other parameters
        if self.request_format == "openai":
            formatted_data["temperature"] = temperature
        elif self.request_format == "ollama":
            formatted_data["options"] = {"temperature": temperature}
        elif self.request_format == "chatml":
            formatted_data["temperature"] = temperature
        
        # Get endpoint
        endpoint = self._get_endpoint_for_provider()
        url = urljoin(self.base_url, endpoint)
        
        print(f"🌐 Sending request to {url}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"📝 Provider: {self.provider}")
        
        try:
            # Send request
            response = await self.client.post(
                url,
                json=formatted_data,
                headers=self.headers
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            content = self._parse_response_for_provider(response_data)
            
            # Calculate timing
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print performance metrics
            if self.last_usage:
                completion_tokens = self.last_usage.get("completion_tokens", 0)
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                print(f"🚀 Generated {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                print(f"📊 Usage: {self.last_usage}")
            else:
                print(f"🚀 Generated response in {total_time:.2f}s")
            
            return content.strip()
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            print(f"❌ Request error: {e}")
            raise Exception(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            raise Exception(f"Invalid JSON response: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise Exception(f"Unexpected error: {e}")
    
    async def process_context(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process context without generating a response - for warm-up purposes"""
        try:
            # For REST backends, we'll just store the messages
            # since we can't easily process context without making a request
            processed_context = {
                "messages": messages,
                "context_processed": True,
                "timestamp": time.time(),
                "backend": "rest",
                "provider": self.provider
            }
            
            print(f"🔥 Context processed for {len(messages)} messages (REST backend)")
            return processed_context
            
        except Exception as e:
            print(f"⚠️  Error processing context: {e}")
            return {
                "messages": messages,
                "context_processed": False,
                "timestamp": time.time(),
                "backend": "rest",
                "provider": self.provider
            }
    
    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: GenerationParams,
    ) -> ChatResult:
        """
        Generate a vision chat response by converting NormalizedMessage to OpenAI format
        and calling the remote API.
        
        This method converts NormalizedMessage (which can contain ImagePart) to the
        OpenAI-style structured content format expected by remote APIs.
        """
        start_time = time.time()
        
        # Convert NormalizedMessage to OpenAI-style messages with structured content
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content_parts: List[Dict[str, Any]] = []
            
            for part in msg.content:
                if isinstance(part, TextPart):
                    content_parts.append({
                        "type": "text",
                        "text": part.text
                    })
                elif isinstance(part, ImagePart):
                    # Convert ImagePart back to data URL format
                    data_url = part.to_data_url()
                    image_part: Dict[str, Any] = {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                    if part.detail:
                        image_part["image_url"]["detail"] = part.detail
                    content_parts.append(image_part)
            
            openai_messages.append({
                "role": msg.role,
                "content": content_parts
            })
        
        print(f"🖼️  Sending vision request to {self.provider} with {len(messages)} messages")
        
        try:
            # Use the chat_completion method which supports structured content
            response_data = await self.chat_completion(
                messages=openai_messages,
                temperature=params.temperature or 0.7,
                max_tokens=params.max_tokens,
                stream=False,  # For now, no streaming
                model=self.model_name,
            )
            
            # Parse response
            content = self._parse_response_for_provider(response_data)
            
            # Calculate timing
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print performance metrics
            if self.last_usage:
                completion_tokens = self.last_usage.get("completion_tokens", 0)
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                print(f"🚀 [Vision] Generated {completion_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.1f} tok/s)")
                print(f"📊 Usage: {self.last_usage}")
            else:
                print(f"🚀 [Vision] Generated response in {total_time:.2f}s")
            
            return ChatResult(content=content.strip(), usage=self.last_usage)
            
        except Exception as e:
            print(f"❌ Error in vision chat: {e}")
            raise
    
    async def unload(self):
        """Clean up resources"""
        if hasattr(self, 'client'):
            await self.client.aclose()
        print(f"🔄 Unloaded REST backend: {self.provider}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'client'):
            # Note: This is not ideal for async cleanup, but it's a fallback
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.client.aclose())
            except:
                pass
