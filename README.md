# Jarvis LLM Proxy API

A flexible, high-performance proxy API for Large Language Models (LLMs) that supports multiple backends and provides OpenAI-compatible endpoints.

## Features

- **Multiple Backend Support**: MLX, GGUF, Transformers, REST (OpenAI, Anthropic, Ollama, LM Studio, custom APIs)
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API
- **Conversation Caching**: Session-based conversation memory with warm-up support
- **Dual Model Support**: Main model + lightweight model for different use cases
- **Async Processing**: High-performance async request handling
- **Flexible Authentication**: Support for various authentication methods
- **Health Monitoring**: Built-in health checks and metrics

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd jarvis-llm-proxy-api
```

### 2. Platform Setup

Run the interactive setup to configure your platform and hardware acceleration:

```bash
./setup.sh
```

This will detect your OS and hardware, then prompt you to select the appropriate acceleration method (CPU, CUDA, Metal, or ROCm).

### 3. Configure Environment

Copy the environment template and configure your settings:

```bash
cp env.template .env
# Edit .env with your model paths and preferences
```

### 4. Run the Application

```bash
# Development (automatically installs dependencies)
./run.sh

# Production
./run-prod.sh
```

For detailed setup instructions, including platform-specific requirements and troubleshooting, see [SETUP.md](SETUP.md).

## Backend Options

### GGUF Backend (Recommended)
- **Use Case**: Local inference with hardware acceleration
- **Inference Engines**:
  - **llama.cpp** (default): Reliable, well-tested inference
  - **vLLM**: High-performance, optimized for throughput
- **Acceleration**: 
  - **CUDA**: NVIDIA GPU acceleration (Linux/Windows)
  - **Metal**: Apple Silicon optimization (macOS)
  - **ROCm**: AMD GPU acceleration (Linux, experimental)
  - **CPU**: Fallback for any system
- **Setup**: Set `JARVIS_MODEL_BACKEND=GGUF` and `JARVIS_MODEL_NAME=/path/to/model.gguf`
- **vLLM**: Set `JARVIS_INFERENCE_ENGINE=vllm` for high-performance inference

### MLX Backend
- **Use Case**: Apple Silicon-specific optimization (macOS only)
- **Setup**: Set `JARVIS_MODEL_BACKEND=MLX` and `JARVIS_MODEL_NAME=/path/to/model`

### GGUF Backend
- **Use Case**: Local inference with llama.cpp
- **Setup**: Set `JARVIS_MODEL_BACKEND=GGUF` and `JARVIS_MODEL_NAME=/path/to/model.gguf`

### Transformers Backend
- **Use Case**: Local inference with regular HuggingFace models (not converted to GGUF/MLX)
- **Setup**: Set `JARVIS_MODEL_BACKEND=TRANSFORMERS` and `JARVIS_MODEL_NAME=model_name_or_path`
- **Features**: Supports quantization, multiple devices (CUDA/MPS/CPU), chat templates

### REST Backend
- **Use Case**: Remote APIs (OpenAI, Anthropic, Ollama, LM Studio, custom)
- **Setup**: Set `JARVIS_MODEL_BACKEND=REST` and configure REST-specific variables

## Transformers Backend Configuration

The Transformers backend provides native support for HuggingFace models without requiring conversion:

### Environment Variables

```bash
# Required
JARVIS_MODEL_BACKEND=TRANSFORMERS
JARVIS_MODEL_NAME=microsoft/DialoGPT-medium  # Or local path

# Device Configuration
JARVIS_DEVICE=auto  # auto, cuda, mps, cpu
JARVIS_TORCH_DTYPE=auto  # auto, float16, float32, bfloat16

# Quantization (optional, requires bitsandbytes)
JARVIS_USE_QUANTIZATION=false
JARVIS_QUANTIZATION_TYPE=4bit  # 4bit, 8bit

# Generation Parameters
JARVIS_MAX_TOKENS=2048
JARVIS_TOP_P=0.95
JARVIS_TOP_K=50
JARVIS_REPETITION_PENALTY=1.1
JARVIS_DO_SAMPLE=true

# Chat Format
JARVIS_MODEL_CHAT_FORMAT=generic  # generic, chatml, llama
JARVIS_MODEL_CONTEXT_WINDOW=4096

# Memory Optimization
JARVIS_USE_CACHE=true
JARVIS_TRUST_REMOTE_CODE=false
```

### Supported Features

- **Multiple Devices**: Automatic device selection (CUDA, MPS, CPU)
- **Quantization**: 4-bit and 8-bit quantization via bitsandbytes
- **Chat Templates**: Automatic detection and fallback formatting
- **Memory Optimization**: Efficient memory usage with caching
- **Thread Safety**: Safe concurrent access to models

### Example Usage

```bash
# Copy example configuration
cp transformers-backend-example.env .env

# Edit configuration as needed
# Run the application
python main.py
```

## REST Backend Configuration

The REST backend supports various remote API providers with flexible authentication:

### Environment Variables

```bash
# Required
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://api.openai.com

# Authentication
JARVIS_REST_AUTH_TYPE=bearer  # none, bearer, api_key, custom
JARVIS_REST_AUTH_TOKEN=your_api_key
JARVIS_REST_AUTH_HEADER=Authorization  # Optional, defaults to Authorization

# Provider Configuration
JARVIS_REST_PROVIDER=openai  # openai, anthropic, ollama, lmstudio, generic
JARVIS_REST_REQUEST_FORMAT=openai  # openai, ollama, chatml, generic
JARVIS_REST_TIMEOUT=60
```

### Quick Examples

#### OpenAI
```bash
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://api.openai.com
JARVIS_REST_PROVIDER=openai
JARVIS_REST_AUTH_TYPE=bearer
JARVIS_REST_AUTH_TOKEN=sk-your-key-here
```

#### Ollama (Local)
```bash
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=http://localhost:11434
JARVIS_REST_PROVIDER=ollama
JARVIS_REST_AUTH_TYPE=none
JARVIS_REST_REQUEST_FORMAT=ollama
```

#### Custom API
```bash
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://your-api.com
JARVIS_REST_PROVIDER=generic
JARVIS_REST_AUTH_TYPE=api_key
JARVIS_REST_AUTH_TOKEN=your_key
JARVIS_REST_AUTH_HEADER=X-API-Key
```

For detailed configuration options, see [REST_BACKEND_CONFIG.md](REST_BACKEND_CONFIG.md).

## API Endpoints

### Chat Completion
```bash
POST /api/v1/chat
```

### Lightweight Chat
```bash
POST /api/v1/lightweight/chat
```

### Session Warm-up
```bash
POST /api/v1/chat/conversation/{conversation_id}/warmup
```

### Model Swap
```bash
POST /api/v1/model-swap
```

### Health Check
```bash
GET /api/v1/health
```

## Testing

Test your REST backend configuration:

```bash
python test_rest_backend.py
```

## Configuration Files

- `.env`: Main configuration file
- `rest-backend-example.env`: Example REST backend configuration
- `REST_BACKEND_CONFIG.md`: Detailed REST backend documentation

## Development

### Project Structure

```
jarvis-llm-proxy-api/
├── backends/           # Backend implementations
│   ├── mlx_backend.py
│   ├── gguf_backend.py
│   └── rest_backend.py
├── managers/           # Model and cache managers
├── cache/             # Caching system
├── main.py            # FastAPI application
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

### Adding New Backends

1. Create a new backend class in `backends/`
2. Implement the required methods: `chat()`, `chat_with_temperature()`, `process_context()`
3. Update `managers/model_manager.py` to support the new backend
4. Add configuration options to environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

## Changelog

### v1.0.0
- Initial release with MLX and GGUF backends
- OpenAI-compatible API endpoints
- Conversation caching system

### v1.1.0
- Added REST backend support
- Support for OpenAI, Anthropic, Ollama, LM Studio
- Flexible authentication system
- Async request handling
