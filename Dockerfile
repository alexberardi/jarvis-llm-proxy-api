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

# Build-time dependencies (git needed for pip git+ installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
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
# 0.3.23 is the version in the currently-deployed prod image, verified working on
# the dual-3090 box. Bump deliberately, not via rebuild drift.
RUN pip install --no-cache-dir "llama-cpp-python==0.3.23" \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
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
COPY gpu_select.py .
COPY pyproject.toml .

# Create directories that may be needed at runtime
RUN mkdir -p /app/.models /app/logs /tmp/jarvis-adapters

# API (7704) + Model service (7705)
EXPOSE 7704 7705

# No CMD — each compose service specifies its own command
