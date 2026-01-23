# Jarvis LLM Proxy API

LLM inference proxy supporting multiple backends (vLLM, Transformers, GGUF) with per-request LoRA adapter loading.

## Running the Project

**Always use the provided scripts:**

```bash
# Start the server (handles venv, deps, and all processes)
./run.sh

# Stop all processes cleanly
./cleanup-processes.sh
```

Do NOT manually start uvicorn or kill processes - use these scripts.

## Key Architecture

- **API server** (port 8000): FastAPI endpoints for chat completions
- **Model service**: Manages backend (vLLM/Transformers) and model loading
- **Queue worker**: Processes async jobs (adapter training)

## Adapter Storage

Adapters are stored flat by hash:
```
{LLM_PROXY_ADAPTER_DIR}/{hash}/
├── adapter_config.json
├── adapter_model.safetensors
└── ...
```

Environment variable: `LLM_PROXY_ADAPTER_DIR` (default: `/tmp/jarvis-adapters`)

## Environment Configuration

Copy `env.template` to `.env` and configure. Key settings:

- `JARVIS_MODEL_NAME`: Model path or HuggingFace ID
- `JARVIS_MODEL_BACKEND`: VLLM, TRANSFORMERS, or GGUF
- `JARVIS_VLLM_QUANTIZATION`: Set to `awq` for AWQ models
- `LLM_PROXY_ADAPTER_DIR`: Where adapters are stored

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```
