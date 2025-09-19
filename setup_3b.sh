#!/bin/bash
# GGUF Backend Configuration for Llama 3.2 3B Instruct Q4_K_M
# Optimized for Mac Studio M4 Max (36GB RAM)

echo "🔧 Loading optimized configuration for Llama 3.2 3B Instruct Q4_K_M..."

# Core model settings
export JARVIS_MODEL_CONTEXT_WINDOW="8192"
export JARVIS_N_THREADS="10"
export JARVIS_N_GPU_LAYERS="-1"
export JARVIS_CHAT_FORMAT="llama3"
export JARVIS_SEED="42"
export JARVIS_VERBOSE="false"

# Memory and performance optimization
export JARVIS_N_BATCH="1024"
export JARVIS_N_UBATCH="1024"
export JARVIS_F16_KV="true"
export JARVIS_MUL_MAT_Q="true"
export JARVIS_ROPE_SCALING_TYPE="0"

# Context caching
export JARVIS_ENABLE_CONTEXT_CACHE="true"
export JARVIS_MAX_CACHE_SIZE="200"

# Inference parameters
export JARVIS_MAX_TOKENS="8000"
export JARVIS_TOP_P="0.95"
export JARVIS_TOP_K="40"
export JARVIS_REPEAT_PENALTY="1.1"
export JARVIS_MIROSTAT_MODE="0"
export JARVIS_MIROSTAT_TAU="5.0"
export JARVIS_MIROSTAT_ETA="0.1"

# Metal acceleration
export LLAMA_METAL="true"

echo "✅ Llama 3.2 3B configuration loaded"
echo "📊 Expected performance: 0.3-0.8s latency, 50-100 tokens/sec"
echo "💾 Memory usage: ~2-4GB"
