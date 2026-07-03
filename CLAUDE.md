# jarvis-llm-proxy-api

OpenAI-compatible LLM inference layer with multi-backend support, an async job queue, embeddings, and LoRA adapter management. **Three cooperating processes**, two models (`live` + `background`), and a deliberately thin HTTP surface.

> **Identity rule:** the API server doesn't load models. The **model service** does. The API and the queue worker both proxy to the model service over HTTP so a single in-process model serves both synchronous chat and async background jobs without double-loading weights.

---

## The three processes

| Process | Port | What it owns | Started by |
|---|---|---|---|
| **API server** (`main.py`) | 7704 | OpenAI-compatible HTTP surface, auth, settings, training-job records, request routing. **Stateless.** | `uvicorn main:app` |
| **Model service** (`services.model_service:app`) | 7705 | The `ModelManager` singleton — actually loads weights into memory. Binds the port FIRST, then loads models in a background thread (per-slot fault-isolated + retried). Exposes `/internal/model/*`. Speaks `X-Internal-Token`. | `uvicorn services.model_service:app` (run by `run.sh`; auto-disabled on macOS by `run.sh` since GPU-dependent backends run locally). In containers: `scripts/serve.sh` supervises it (respawn with backoff). |
| **Queue worker** (`scripts/queue_worker.py`) | — | Consumes Redis jobs (`adapter_train`, async LLM jobs), calls the model service for inference, fires callbacks. Optional. | `LLM_PROXY_PROCESS_ROLE=worker python scripts/queue_worker.py` |

```
Caller (e.g. command-center)
   │
   ▼  POST /v1/chat/completions      (X-Jarvis-App-Id/Key)
┌──────────────────┐                                  ┌──────────────────────┐
│  API server      │  ─── X-Internal-Token ──────▶   │   Model service      │
│  :7704           │  POST /internal/model/chat       │   :7705              │
│  (stateless)     │  ◀── 200 / SSE stream ──────    │   ModelManager       │
└──────────────────┘                                  │   (live + background)│
       ▲                                              └──────────────────────┘
       │ POST /internal/queue/enqueue                            ▲
       │  (also app-auth)                                        │
       │                                                          │ X-Internal-Token
┌──────────────────┐                                              │
│  Queue worker    │  ◀── Redis BLPOP ──── Redis ◀── enqueue ─────┘
│                  │  ──── POST /internal/model/chat ─────────────┘
│  (separate proc) │
└──────────────────┘
       │
       ▼ HTTP callback to caller's callback_url on completion
```

---

## Two-model architecture

Two **roles**:
- **`live`** — used by the synchronous chat endpoint. Optimized for latency. Always served from the model service's `live_model` slot.
- **`background`** — used by the queue worker. Optimized for accuracy/quality (can be slower). Served from `background_model` slot.

**Sharing rule** (`ModelManager._populate_registry`):
- If `live.path + live.backend` == `background.path + background.backend`, both aliases resolve to a **single backend instance** — no double-load.
- Otherwise two backends are instantiated and both consume memory.

**In practice:**
- **Dev / Mac** — shared (one model, both aliases). Limited memory.
- **Prod** — split (e.g. 14b for `live`, 32b for `background`) running simultaneously on the GPU box.

The chat endpoint defaults `model` to `live` unless the caller explicitly passes `"background"`. The queue worker always uses `background`.

---

## Quick Reference

```bash
# Dev (auto-installs deps, detects platform, starts API + model + worker)
./run.sh

# Force rebuild deps
./run.sh --rebuild

# Docker
./run-docker-dev.sh

# Tests (excludes manual + the flaky adapter cache test)
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py

# With coverage
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py --cov=.

# Migrations
alembic upgrade head
```

---

## Dependency graph

**Upstream (proxy depends on):**
- **Model service** (port 7705) — required for inference. If down, chat / embeddings / pipeline all 5xx.
- **PostgreSQL** (required) — training jobs, settings (multitenant)
- **Redis** (required for queue/worker path; chat works without it)
- **jarvis-auth** (port 7701) — app-creds validation (chat/embeddings/queue) + superuser JWT (settings/pipeline)
- **jarvis-config-service** (port 7700) — service discovery
- **MinIO / S3** (optional) — adapter storage; needed only if adapters are actively used
- **HuggingFace Hub** (optional) — gated model downloads

**Downstream (depends on proxy):**
- **jarvis-command-center** — main consumer; `/v1/chat/completions` for live inference + `/internal/queue/enqueue` for adapter train, deep research, memory extraction. Embeddings for memory features.
- **jarvis-tts** — wake-word response phrasing
- **jarvis-ocr-service** — LLM validation of OCR'd text
- **jarvis-recipes-server** — recipe extraction, meal planning

**Impact if down:**
- No LLM-based command parsing → all voice commands fail
- No adapter training (when adapters are active)
- No AI-powered features anywhere

---

## API surface (OpenAI-compatible where possible)

### Chat (`api/chat_routes.py`, app-auth)
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible. Body field `model` accepts `live` (default) or `background`. `stream=true` returns SSE proxied from the model service. |

### Embeddings (`api/embedding_routes.py`, app-auth)
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/embeddings` | OpenAI-compatible. Default model `all-MiniLM-L6-v2`. Used by **command-center** for memory similarity search. |

### Queue (`api/queue_routes.py`, app-auth)
| Method | Path | Notes |
|---|---|---|
| POST | `/internal/queue/enqueue` | Enqueues `adapter_train` or generic LLM jobs. Validated per `job_type`. Idempotency via `idempotency_key`. Callback fired to `callback.url` with bearer token. |

### Training jobs (`api/training_routes.py`, app-auth)
| Method | Path | Notes |
|---|---|---|
| POST | `/v1/training` | Start LoRA training job (record in DB; actual training enqueued or run by worker) |
| GET | `/v1/training/status/{job_id}` | Poll job status |

### Adapters (`api/adapter_routes.py`, app-auth)
| Method | Path | Notes |
|---|---|---|
| GET | `/v1/adapters` | List available adapters |
| POST | `/v1/adapters/activate` | Activate a LoRA adapter for a model |

### Model (`api/model_routes.py`, mixed auth)
| Method | Path | Notes |
|---|---|---|
| GET | `/api/v1/model/info` | Current model info |
| POST | `/api/v1/model-swap` | Hot-swap loaded model |

### Pipeline (`api/pipeline_routes.py`, superuser JWT) — **WIP, ignore for now**
Full model build pipeline (generate → train → validate → merge → convert to GGUF/MLX). Maintainer-only today; not yet a stable surface. **Don't develop against this** unless explicitly working on the pipeline itself.

### Settings (`api/settings_routes.py`, library mount)
- Reads: superuser JWT OR app credentials
- Writes: superuser JWT only

### Health
| Method | Path |
|---|---|
| GET | `/health` (also legacy `/v1/health`) |

**HTTP status codes are load-bearing** (the docker healthcheck fails on non-2xx):
- **200 `healthy`** — model service reachable, live slot ready.
- **200 `initializing`** — live slot still loading and model-service uptime < 15 min (grace so slow 32B loads don't flap container health).
- **200 `degraded` + `MODEL_SERVICE_URL not set`** — deliberate passthrough posture (macOS dev); stays green.
- **503** — model service unreachable/timeout, live slot `failed`, or still loading past the grace window.

The model service's own `/health` (:7705) is always HTTP 200 while the process is alive; its body carries `status` (`ok`/`loading`/`degraded`), `models`, `aliases`, plus per-slot `slots` state, `uptime_s`, `started_at`. Inference endpoints return **503 `model_not_loaded`** (and trigger a non-blocking retry) when the addressed slot has no backend.

---

## "How to..." recipes

### Add a new chat-related setting

Settings live in the **DB**, not env. `env.template` is bootstrap/secrets only.

1. Add a row in `services/settings_service.py` (definitions) — set key, value_type, default, env_fallback.
2. Read with `services/settings_helpers.py:get_setting` / `get_int_setting` / `get_float_setting`. Always include an env-var fallback name as the second arg — it's the migration story for older deployments.
3. Seed for fresh installs: `python scripts/seed_settings.py --force`.

### Add a new backend

1. Subclass `backends/base.py:BaseBackend`. Implement `generate`, `generate_stream`, model loading, tokenizer setup, chat-format wiring.
2. Register the backend type in `managers/model_manager.py:_create_backend` (`backend_type` switch). Add to the `_initialize_models` discovery path.
3. If the backend supports vision, add a `*_vision_backend.py` companion (existing pattern: each backend has a vision variant when applicable).
4. Add a chat format if needed in `backends/chat_formats.py`.

### Add a new OpenAI-compatible route

Add a router in `api/`. Use `Depends(require_app_auth)` (from `auth/app_auth.py`) for service callers, or `Depends(create_superuser_auth(get_auth_url()))` for admin-only flows. Proxy to model service via `MODEL_SERVICE_URL` + `X-Internal-Token`. Wire it up in `main.py`.

### Add a queue job type

1. Add a Pydantic request model in `models/queue_models.py`.
2. Handle the new `job_type` in `api/queue_routes.py:enqueue_job_endpoint` validation switch.
3. Implement the worker handler in `queues/tasks.py` (or wire a new module). The worker dispatches on `job_type`.
4. The caller provides a `callback.url` + bearer token; the worker fires HTTP POST on completion.

### Run an LLM inference inside this service (rare)

Don't talk to the model directly — go through the model service over HTTP using `MODEL_SERVICE_URL` and `MODEL_SERVICE_TOKEN`. The chat route is the reference implementation. **No code in the API process should import a backend.**

---

## Invariants & gotchas

1. **API server is stateless — model service owns the weights.** Restarting the API doesn't drop the model. Restarting the model service drops everything and reloads on next request. Don't try to load a model from the API codebase; the only reason that would "work" would be to load it twice. **Model loads are per-slot fault-isolated and self-healing** (since the 2026-07 boot-fatal-load incident): the model service binds :7705 first, then loads in a background thread; a failed slot records `model_states[slot] = "failed"` and 503s (`model_not_loaded`) instead of killing the process, and a retry loop re-attempts it with 60s→600s exponential cooldown. `scripts/serve.sh` (the container launcher) additionally respawns the whole model service if it dies natively (llama.cpp crashes).
2. **Settings DB is the source of truth — env is bootstrap/secrets only.** When adding a new config knob, **always prefer adding to settings_service** with an env-var fallback for migration. The `env.template` file states this explicitly: secrets, service discovery, ports, and library env (`LLAMA_METAL`, etc.) stay in env; everything else goes in the DB.
3. **`MODEL_SERVICE_TOKEN` vs `LLM_PROXY_INTERNAL_TOKEN` are different tokens with different scopes.** The first protects `/internal/model/*` on the model service (called by API + worker). The second protects `/internal/queue/*` on the API server (called by command-center and other queue producers). They can be the same string in practice (run.sh does this for dev convenience), but they're semantically distinct.
4. **`live` and `background` share a backend instance if their resolved path + backend match.** Saves memory on small dev boxes (Mac). On prod with separate models, both load. Don't write code that assumes they're always the same OR always different — check `ModelManager.aliases["live"] == ModelManager.aliases["background"]` when needed.
5. **Old aliases (`full`, `lightweight`, `cloud`, `vision`) have been removed.** Only `live` and `background` exist. If a downstream caller passes anything else, the chat route forces it to `live` (see `chat_routes.py:43`). Outdated docs may still mention legacy aliases — they're wrong.
6. **Adapter training pipeline exists but Qwen3 GGUF LoRA conversion is broken upstream** (llama.cpp #21125). Adapters work for older models (Qwen2.5, Llama, Hermes) at 96-99% accuracy on the date-key task. **Don't build features that depend on Qwen3 adapters until that lands.** Current command-center code has the adapter injection point hard-disabled — see CC's CLAUDE.md.
7. **Date-key extraction is regex, not neural — by design.** `services/date_key_matcher.py` hits 100% on 4,987 test examples. **Do not attempt neural classifiers for this task.** DeBERTa-v3-small produces NaN gradients (known issue: disentangled attention + multi-label classification). DistilBERT learns nothing useful (62 labels, ~2% positive rate each → all predictions below threshold). Regex is the right answer here: deterministic, instant, debuggable, testable.
8. **Pipeline routes are WIP.** Don't develop against `/v1/pipeline/*` unless you're working on the pipeline itself. The route surface and semantics may change.
9. **`tests/test_adapter_cache.py` is flaky and excluded from the default test run.** The standard test command already skips it. Don't add it back without fixing the underlying flakiness.
10. **`tests/manual/` requires a running model.** Excluded from CI / standard runs. Use for latency benchmarks and vision tests where you want to see real numbers.
11. **`multiprocessing.set_start_method("spawn", force=True)` at module top.** Required for vLLM + CUDA. Don't remove or move below imports that may touch multiprocessing — it breaks GPU init.
12. **`load_dotenv()` runs before any config/route imports.** Many module-level constants read env at import time; reordering will break them.

---

## Process / dev model

- **macOS (Apple Silicon):** GPU-dependent backends (MLX, Metal-accelerated GGUF) run **locally** on the host to access the GPU. The `jarvis` CLI overrides `mode=docker` → `mode=local` for this service. `run.sh` auto-disables the in-process model service start on macOS and points at a host-side model service.
- **Linux (NVIDIA GPU):** all three processes can run in Docker; vLLM backend uses `nvidia-docker` (`deploy.resources.reservations.devices`) for CUDA passthrough. The full stack runs as a single docker-compose.
- **Standalone testing:** `MOCK` backend (`backends/mock_backend.py`) for tests / CI / quick API smoke without loading anything real.

---

## Data model

PostgreSQL via Alembic (`alembic/versions/`):

| Table | Purpose |
|---|---|
| `training_jobs` | LoRA training job state — type, status, model_id, dataset_hash, params, artifact_metadata, error fields |
| `multitenant_settings` | The shared settings table (key + scope tuple) |

Redis (queues):
- Job lists per `job_type`
- Dedup keys (`idempotency_key`)
- Job status (transient — durable state is in Postgres)

Storage:
- **MinIO/S3** — adapter blobs (`storage/object_store.py`). `S3_FORCE_PATH_STYLE=true` for MinIO compatibility.

---

## Backends

| Backend | Platform | Use case |
|---|---|---|
| `GGUF` | All | llama.cpp, primary local inference. Vision variant exists. |
| `MLX` | macOS (Apple Silicon) | Native Metal. Vision variant exists. |
| `VLLM` | Linux + CUDA | High-throughput GPU. Vision variant exists. |
| `REST` | All | Proxy to a remote OpenAI-compatible API |
| `TRANSFORMERS` | All | HuggingFace transformers (fallback, vision variant exists) |
| `MOCK` | All | Testing without a real model |

Each backend follows the same interface (`backends/base.py`). Chat format strategies live in `backends/chat_formats.py` (Qwen3, Llama, Hermes, Gemma, etc.). When a model uses an uncommon chat template, add a strategy there rather than coercing the backend to handle special cases.

---

## Config surface

Env vars (bootstrap + secrets only — see `env.template`):

| Variable | Required | Purpose |
|---|---|---|
| `JARVIS_CONFIG_URL` | yes | Service discovery |
| `JARVIS_APP_ID` / `JARVIS_APP_KEY` | yes | Logging app credentials |
| `JARVIS_AUTH_APP_ID` / `JARVIS_AUTH_APP_KEY` | yes | Auth app credentials |
| `MODEL_SERVICE_TOKEN` | yes | API ↔ model service auth |
| `LLM_PROXY_INTERNAL_TOKEN` | yes | Queue enqueue auth |
| `JARVIS_ADMIN_TOKEN` | yes | Admin endpoints |
| `DATABASE_URL` | yes | Postgres |
| `REDIS_URL` | yes (for queue) | Redis |
| `HUGGINGFACE_HUB_TOKEN` | optional | Gated model downloads |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | optional | S3/MinIO adapter storage |
| `S3_FORCE_PATH_STYLE` | optional | MinIO compatibility |
| `LLAMA_METAL` | optional | Force-enable Metal in llama.cpp |
| `LLAMA_LOG_LEVEL` | optional | llama.cpp log verbosity |
| `SERVER_HOST` / `SERVER_PORT` | optional (7704) | API server bind |
| `MODEL_SERVICE_PORT` | optional (7705) | Model service bind |
| `RUN_MODEL_SERVICE` | optional | Override the run.sh macOS auto-disable |
| `LLM_PROXY_PROCESS_ROLE` | optional | Set to `worker` for queue worker process |

**DB-backed settings (preferred for non-secret config):**
- `model_service.url`, `model_service.timeout_seconds`
- Live + background model paths, backends, context windows (`model.live.*`, `model.background.*`)
- Provider chat templates, stop tokens
- Adapter scheduler settings
- Inference defaults (temperature, top_p, max_tokens, etc.)

Seed via `python scripts/seed_settings.py --force` on a fresh install.

---

## Architecture

```
jarvis-llm-proxy-api/
├── main.py                         # API server entry — FastAPI app, router wiring
├── run.sh                          # Platform-aware dev runner (API + model service + worker)
├── api/                            # HTTP route modules — thin, all proxy to model service
│   ├── chat_routes.py              # /v1/chat/completions
│   ├── embedding_routes.py         # /v1/embeddings
│   ├── queue_routes.py             # /internal/queue/enqueue
│   ├── training_routes.py          # /v1/training/*
│   ├── adapter_routes.py           # /v1/adapters/*
│   ├── model_routes.py             # /api/v1/model/info, /model-swap
│   ├── pipeline_routes.py          # /v1/pipeline/* — WIP, maintainer-only
│   ├── settings_routes.py          # /settings/*
│   └── health_routes.py            # /api/v1/health
├── auth/
│   └── app_auth.py                 # require_app_auth dependency (X-Jarvis-App-Id/Key → auth)
├── managers/
│   ├── model_manager.py            # Singleton — loads backends, owns live + background slots
│   ├── embedding_manager.py        # Embedding model loader (defaults to MiniLM)
│   └── chat_types.py               # ModelConfig and shared types
├── backends/                       # Inference engines (loaded by model service only)
│   ├── base.py                     # BaseBackend abstract interface
│   ├── chat_formats.py             # Per-model chat templates (Qwen3, Llama, Hermes, Gemma, ...)
│   ├── gguf_backend.py             # llama.cpp — primary local
│   ├── mlx_backend.py              # macOS native (Metal)
│   ├── vllm_backend.py             # Linux + CUDA (high throughput)
│   ├── rest_backend.py             # Remote OpenAI-compatible proxy
│   ├── transformers_backend.py     # HuggingFace transformers
│   ├── mock_backend.py             # Tests
│   └── *_vision_backend.py         # Vision variant of each
├── services/                       # Business logic (DB writes, formatting, helpers)
│   ├── model_service.py            # Model service entrypoint (FastAPI at :7705)
│   ├── chat_runner.py
│   ├── settings_service.py / settings_helpers.py
│   ├── training_job_service.py
│   ├── adapter_cache.py / adapter_storage.py / adapter_training.py
│   ├── date_key_matcher.py         # Regex — 100% on 4,987 examples; DO NOT replace with NN
│   ├── date_key_adapter.py / date_keys.py
│   ├── json_grammar.py / json_repair_service.py
│   ├── pipeline_service.py         # WIP
│   ├── message_service.py / response_helpers.py
│   └── vision_inference.py
├── queues/
│   ├── redis_queue.py              # BLPOP, push, dedup, status
│   └── tasks.py                    # Worker handlers per job_type
├── scripts/
│   ├── serve.sh                    # Container launcher — API foreground + supervised model service (respawn w/ backoff); RUN_MODEL_SERVICE=false skips
│   ├── queue_worker.py             # Standalone worker process entry
│   ├── seed_settings.py            # Bootstrap settings DB from env
│   ├── train_adapter*.py           # CLI tools for adapter training
│   ├── build_jarvis_model.py       # Model build orchestration
│   ├── convert_to_gguf.py / convert_to_mlx.py
│   ├── merge_adapter.py / load_adapter.py
│   ├── quantize_awq.py
│   └── train_date_key_classifier.py  # Historical — see Invariants #7
├── config/                         # logging, debug, service config
├── storage/object_store.py         # S3/MinIO adapter for adapter blobs
├── db/                             # SQLAlchemy session + models
├── models/                         # Pydantic request/response shapes
├── alembic/                        # Migrations
├── data/jarvis_training.jsonl      # Date-key test dataset (4,987 examples)
├── adapters/                       # Local adapter cache directory
└── tests/
    ├── manual/                     # Latency, vision — excluded from CI
    └── test_*.py                   # Unit
```

---

## Testing

```bash
# Standard run (excludes manual + flaky adapter cache test)
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py

# With coverage
pytest -v --tb=short --ignore=tests/manual --ignore=tests/test_adapter_cache.py --cov=.

# Date-key matcher specifically
pytest tests/test_date_key_classifier.py -v
```

`tests/manual/` requires a running model — for benchmarks and vision smoke tests. `test_adapter_cache.py` is flaky; skip by default.

---

## Failure modes

| Failure | Behavior |
|---|---|
| Model service down | Chat / embeddings / pipeline return 500 with "MODEL_SERVICE_URL is not set; API is passthrough-only" or upstream error. API `/health` returns **503** → docker marks the container unhealthy. In containers, `scripts/serve.sh` respawns the model service with backoff. |
| Model failed to load (per slot) | Model service stays up; affected requests get **503 `model_not_loaded`**; retry loop re-attempts with 60s→600s cooldown. Live slot failed ⇒ API `/health` 503. |
| Postgres down | Settings reads fall back to env-var-fallback values; training routes fail |
| Redis down | Chat still works; queue enqueue fails |
| Auth down | All app-auth routes return 401/503; pipeline + settings writes 401 |
| `MODEL_SERVICE_TOKEN` mismatch | Model service rejects → 5xx |
| HuggingFace down | Cold start of a gated model fails; warm models keep working |
| MinIO/S3 down | Adapter download fails → adapter not applied → fallback to baseline model |

---

## Out of scope / explicitly not here

- **Voice / audio processing.** Whisper for STT, jarvis-tts for TTS. This service does text only (plus vision images via vision backends).
- **Conversation state.** Stateless. Command-center owns the conversation cache.
- **Tool execution.** Tool definitions and execution live in command-center. This service just emits `tool_calls` blocks.
- **Application-level authorization.** App-auth proves *who* the caller is; downstream services decide *what* they can do.
- **Model fine-tuning UX.** Pipeline routes are WIP; the production fine-tuning UX is still being designed.
