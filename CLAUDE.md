# jarvis-llm-proxy-api

OpenAI-compatible LLM inference proxy with multiple backend support, LoRA adapter training, and conversation caching.

## Quick Reference

```bash
# Development (auto-installs deps, detects platform)
./run.sh

# Force rebuild deps
./run.sh --rebuild

# Docker development
./run-docker-dev.sh

# Tests
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py

# Tests with coverage
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py --cov=app
```

## Architecture

```
jarvis-llm-proxy-api/
├── main.py                # FastAPI entry point (87 lines)
├── run.sh                 # Smart dev runner (platform detection)
├── api/                   # Route modules
│   ├── chat_routes.py     # /api/v1/chat (main inference endpoint)
│   ├── queue_routes.py    # Async job queue
│   ├── model_routes.py    # Model info, swap
│   ├── health_routes.py   # /api/v1/health
│   ├── training_routes.py # LoRA adapter training
│   ├── adapter_routes.py  # Adapter CRUD
│   ├── pipeline_routes.py # Multi-step pipelines
│   └── settings_routes.py # Runtime settings
├── backends/              # LLM inference engines
│   ├── base.py            # Abstract backend interface
│   ├── gguf_backend.py    # llama.cpp (primary local backend)
│   ├── mlx_backend.py     # Apple Silicon (macOS)
│   ├── vllm_backend.py    # High-perf GPU inference (Linux)
│   ├── rest_backend.py    # Remote API proxy
│   ├── transformers_backend.py
│   ├── mock_backend.py    # Testing
│   └── *_vision_backend.py  # Vision model variants
├── managers/
│   ├── model_manager.py   # Core model lifecycle & inference
│   └── chat_types.py      # Request/response types
├── cache/                 # Conversation caching
│   ├── cache_manager.py
│   ├── conversation_cache.py
│   └── local_conversation_cache.py
├── storage/
│   └── object_store.py    # S3/MinIO adapter storage
├── auth/
│   └── app_auth.py        # App-to-app authentication
├── config/
│   ├── logging_config.py
│   ├── debug_config.py
│   └── service_config.py  # Service discovery
├── db/                    # SQLAlchemy models (TrainingJob, Settings)
├── alembic/               # 4 migrations (training jobs, settings, multitenant)
└── tests/
    ├── manual/            # Latency, vision tests (excluded from CI)
    └── *.py               # Unit tests
```

## Ports

| Port | Process | Description |
|------|---------|-------------|
| 7704 | API Server | FastAPI HTTP endpoints |
| 7705 | Model Service | Backend model process (auto-disabled on macOS) |

## Model System

The proxy runs a **2-model architecture**:

| Model | Purpose | Routing |
|-------|---------|---------|
| **live** | Real-time voice, chat | Chat endpoint defaults here |
| **background** | Async tasks (summarisation, OCR, research) | Queue worker always uses this |

If both resolve to the same model path + backend, they **share a single backend instance** (memory optimisation for weaker hardware). Either can be local (MLX/GGUF) or remote (REST).

Old aliases are supported for backwards compatibility with deprecation warnings:
- `full`, `lightweight` → **live**
- `cloud`, `vision` → **background**

### Config Fallback Chain

```
model.live.*    → JARVIS_LIVE_MODEL_*    → JARVIS_MODEL_* (legacy)
model.background.* → JARVIS_BACKGROUND_MODEL_* → falls back to live config
```

## API Endpoints

**Chat (main):**
- `POST /api/v1/chat` — Chat completion (OpenAI-compatible, defaults to live model)
- `POST /api/v1/chat/conversation/{id}/warmup` — Pre-warm conversation cache

**Models:**
- `GET /api/v1/health` — Health check
- `POST /api/v1/model-swap` — Hot-swap loaded model
- `GET /api/v1/model/info` — Current model info

**Training:**
- `POST /v1/training` — Start LoRA adapter training job
- `GET /v1/training/status/{job_id}` — Check training job status

**Adapters:**
- `GET /v1/adapters` — List available adapters
- `POST /v1/adapters/activate` — Activate a LoRA adapter

**Settings:**
- `GET/PUT /settings/*` — Runtime LLM configuration (stored in DB)

## Backends

| Backend | Platform | Use Case |
|---------|----------|----------|
| `GGUF` | All | llama.cpp, primary local inference |
| `MLX` | macOS | Apple Silicon optimized |
| `VLLM` | Linux + CUDA | High-throughput GPU inference |
| `REST` | All | Proxy to remote OpenAI-compatible API |
| `TRANSFORMERS` | All | HuggingFace transformers |
| `MOCK` | All | Testing without a model |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 7704 | API server port |
| `MODEL_SERVICE_PORT` | 7705 | Model service port |
| `RUN_MODEL_SERVICE` | true | Run model service (false on macOS) |
| `JARVIS_MODEL_BACKEND` | GGUF | Default backend type (legacy fallback) |
| `JARVIS_MODEL_NAME` | - | Default model path (legacy fallback) |
| `JARVIS_LIVE_MODEL_NAME` | - | Live model path (falls back to JARVIS_MODEL_NAME) |
| `JARVIS_LIVE_MODEL_BACKEND` | - | Live backend (falls back to JARVIS_MODEL_BACKEND) |
| `JARVIS_BACKGROUND_MODEL_NAME` | - | Background model path (falls back to live) |
| `JARVIS_BACKGROUND_MODEL_BACKEND` | - | Background backend (falls back to live) |
| `DATABASE_URL` | - | PostgreSQL connection |
| `REDIS_URL` | - | Redis for async queue |
| `REDIS_PASSWORD` | redis | Redis password |
| `JARVIS_CONFIG_URL` | http://localhost:7700 | Service discovery |
| `JARVIS_APP_ID` | jarvis-llm-proxy-api | App credential ID |
| `JARVIS_APP_KEY` | - | App credential key |
| `MODEL_SERVICE_TOKEN` | - | Internal API token |
| `LLM_PROXY_INTERNAL_TOKEN` | - | Internal service token |
| `HUGGINGFACE_HUB_TOKEN` | - | For gated model downloads |
| `S3_FORCE_PATH_STYLE` | false | MinIO compatibility |

## Database

- Alembic migrations for training jobs, settings tables, and model config cleanup
- Stores: `TrainingJob` (LoRA training state), `MultitenantSettings` (per-service LLM config)
- Run migrations: `alembic upgrade head`

## Date Key Extraction

Deterministic regex-based matcher in `services/date_key_matcher.py`. 100% accuracy on 4,987 test examples
(`data/jarvis_training.jsonl`). Handles combined keys, abbreviations, relative offsets, meal times, dynamic
time patterns, and negative filtering (media titles, idioms, durations).

**Do NOT attempt neural classifiers for this task.** DeBERTa-v3-small produces NaN gradients on this dataset
(known issue with disentangled attention + multi-label classification). DistilBERT learns nothing useful
(all predictions below threshold due to label sparsity — 62 labels, ~2% positive rate each). The regex
approach is the correct solution: deterministic, instant, debuggable, and testable.

LoRA adapters exist for non-Qwen3 models (Qwen2.5, Llama, Hermes) at 96-99% accuracy but Qwen3 GGUF LoRA
conversion is broken upstream (llama.cpp #21125). Adapters are supplemental, not primary.

## Testing

```bash
# Standard test run (excludes manual tests and flaky adapter cache test)
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py

# Date key matcher tests (4,987 examples + adversarial suite)
pytest tests/test_date_key_classifier.py -v

# No [dev] extras — pytest must be installed separately if not using ./jarvis test
```

`tests/manual/` contains latency benchmarks and vision tests that require a running model.

## Dependencies

**Service Dependencies:**
- **Required**: `jarvis-config-service` (7700) — service discovery
- **Required**: `jarvis-auth` (7701) — app credential validation
- **Required**: PostgreSQL — training jobs, settings
- **Optional**: Redis — async job queue
- **Optional**: MinIO/S3 — adapter storage

**Used By:**
- `jarvis-command-center` — voice command inference
- `jarvis-tts` — wake word responses
- `jarvis-ocr-service` — LLM-powered text validation
- `jarvis-recipes-server` — recipe extraction, meal planning

**Impact if Down:**
- No LLM-based command parsing
- No adapter training
- No AI-powered features across the system
