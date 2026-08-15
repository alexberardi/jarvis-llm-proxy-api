# ============================================================================
# Jarvis LLM Proxy API - Dockerfile (vLLM + CUDA)
# ============================================================================
# Multi-stage build targeting Linux with NVIDIA GPUs.
# The GGUF/llama-cpp path remains for native macOS use via run.sh.
#
# Build:  docker compose -f docker-compose.dev.yaml build
# Run:    ./run-docker-dev.sh
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install precompiled wheels
# ---------------------------------------------------------------------------
# Base image pinned by digest. NVIDIA pushes silent updates to the
# 12.4.1-runtime-ubuntu22.04 tag; in jarvis-whisper-api a newer rev's
# cuBLAS/cuDNN SIGILL'd whisper.cpp on prod's RTX 3090s during model
# load. Pin to the digest the currently-deployed prod image was built
# against to keep rebuilds reproducible. Bump deliberately, not via
# tag drift. To refresh:
#   docker buildx imagetools inspect nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build-time dependencies (git needed for pip git+ installs).
#
# python3.11 comes from the DEADSNAKES PPA, not jammy's archive: Ubuntu
# 22.04 froze its python3.11 package at 3.11.0rc1 (a 2022 release
# candidate), which lacks sys.get_int_max_str_digits — torch >=2.13's
# dynamo polyfills reference it unconditionally at import, which killed
# `import sentence_transformers` and 500'd /v1/embeddings on prod
# (2026-08-15: CC memory recall + the embedding sweep both died).
# Deadsnakes ships current 3.11.x (verified 3.11.15 has the API).
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

# Install base requirements first (better layer caching)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Install vLLM from precompiled wheel (no source compilation needed)
COPY requirements-vllm.txt .
RUN pip install --no-cache-dir -r requirements-vllm.txt

# Install llama-cpp-python with prebuilt CUDA 12.4 wheels (for GGUF backend).
# Pinned: unpinned installs let every CI rebuild silently pick up a new llama.cpp;
# this version has a prebuilt linux_x86_64 wheel on the cu124 index (no silent
# sdist/CPU-only fallback). Bump deliberately, not via rebuild drift.
#
# 0.3.34: REQUIRED for the Qwen3.5 hybrid SSM+attention models that carry an MTP
# (nextn) prediction layer. 0.3.20–0.3.23 parse the `qwen35` base arch but
# mis-handle the trailing MTP block — it loads as a plain SSM block and llama.cpp
# aborts with "missing tensor blk.<N>.ssm_conv1d.weight", so the model won't load
# at all. Verified 2026-08: 0.3.34 loads Qwen3.5-9B-Q4_K_M.gguf and serves on the
# CUDA box (also still loads the existing qwen3-8b tools GGUF). MTP speculative
# decoding itself needs a separate --spec-type draft-mtp draft config.
RUN pip install --no-cache-dir "llama-cpp-python==0.3.34" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim image with only what we need
# ---------------------------------------------------------------------------
# Same pinned digest as the builder stage above.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:517da2300c184c9999ec203c2665244bdebd3578d12fcc7065e83667932643d9 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime system dependencies
# gcc is required by torch inductor / triton for JIT CUDA graph compilation
# python3.11 from deadsnakes — jammy's is 3.11.0rc1; see the builder stage
# comment for the full story. Both stages MUST use the same source or the
# dist-packages copied from the builder run on a mismatched interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    libgomp1 \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY api/ ./api/
COPY auth/ ./auth/
COPY backends/ ./backends/
COPY cache/ ./cache/
COPY config/ ./config/
COPY db/ ./db/
COPY managers/ ./managers/
COPY models/ ./models/
COPY queues/ ./queues/
COPY scripts/ ./scripts/
COPY services/ ./services/
COPY storage/ ./storage/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY main.py .
# version.py is read by api/health_routes.py to report the running version on
# /health. The root .py files are copied EXPLICITLY here, so a new one must be
# added or the container dies at import: ModuleNotFoundError: No module named
# 'version' (caught by install-e2e, 2026-07-14).
COPY version.py .
COPY gpu_select.py .
COPY pyproject.toml .

# Create directories that may be needed at runtime
RUN mkdir -p /app/.models /app/logs /tmp/jarvis-adapters

# API (7704) + Model service (7705)
EXPOSE 7704 7705

# Default: supervised launcher (alembic migrations, then API in foreground +
# auto-respawning model service). Compose services typically override this
# with their own command; serve.sh is the migration target for those compose
# commands (it replaces the unsupervised `uvicorn model_service & exec uvicorn
# main` pattern). serve.sh runs `alembic upgrade head` itself (RUN_MIGRATIONS
# defaults true), so export-style compose commands that currently inline
# `alembic upgrade head && ...` can drop their command without silently
# losing migrations.
CMD ["bash", "scripts/serve.sh"]
