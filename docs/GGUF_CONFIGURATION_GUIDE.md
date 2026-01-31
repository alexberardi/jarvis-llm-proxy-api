# GGUF Backend Configuration Guide

## All Configurable Environment Variables

### Core Model Settings
```bash
# Context window size (tokens)
export JARVIS_MODEL_CONTEXT_WINDOW="4096"

# Number of CPU threads
export JARVIS_N_THREADS="10"

# Number of GPU layers (-1 = all layers)
export JARVIS_N_GPU_LAYERS="-1"

# Random seed for reproducible results
export JARVIS_SEED="42"

# Enable verbose logging
export JARVIS_VERBOSE="false"
```

### Memory and Performance Optimization
```bash
# Batch size for GPU processing
export JARVIS_N_BATCH="512"

# Micro batch size for memory efficiency
export JARVIS_N_UBATCH="512"

# Enable F16 precision for key/value cache
export JARVIS_F16_KV="true"

# Enable optimized matrix multiplication
export JARVIS_MUL_MAT_Q="true"

# RoPE scaling type (0 = disabled, 1 = linear, 2 = yarn)
export JARVIS_ROPE_SCALING_TYPE="0"
```

### Context Caching
```bash
# Enable context cache for prefix matching
export JARVIS_ENABLE_CONTEXT_CACHE="true"

# Maximum number of cached contexts
export JARVIS_MAX_CACHE_SIZE="100"
```

### Inference Parameters
```bash
# Maximum tokens to generate
export JARVIS_MAX_TOKENS="7000"

# Top-p sampling parameter
export JARVIS_TOP_P="0.95"

# Top-k sampling parameter
export JARVIS_TOP_K="40"

# Repeat penalty
export JARVIS_REPEAT_PENALTY="1.1"

# Mirostat mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
export JARVIS_MIROSTAT_MODE="0"

# Mirostat tau (target entropy)
export JARVIS_MIROSTAT_TAU="5.0"

# Mirostat eta (learning rate)
export JARVIS_MIROSTAT_ETA="0.1"
```

## Optimized Settings for Mac Studio M4 Max (36GB RAM)

### For Llama 3.2 3B Instruct Q4_K_M
```bash
# Core settings optimized for 3B model
export JARVIS_MODEL_CONTEXT_WINDOW="8192"
export JARVIS_N_THREADS="10"
export JARVIS_N_GPU_LAYERS="-1"
export JARVIS_N_BATCH="1024"
export JARVIS_N_UBATCH="1024"
export JARVIS_F16_KV="true"
export JARVIS_MUL_MAT_Q="true"
export JARVIS_ROPE_SCALING_TYPE="0"

# Memory management
export JARVIS_ENABLE_CONTEXT_CACHE="true"
export JARVIS_MAX_CACHE_SIZE="200"

# Inference settings
export JARVIS_MAX_TOKENS="8000"
export JARVIS_TOP_P="0.95"
export JARVIS_TOP_K="40"
export JARVIS_REPEAT_PENALTY="1.1"
export JARVIS_MIROSTAT_MODE="0"

# Metal acceleration
export LLAMA_METAL="true"
```

**Expected Performance:**
- Average latency: 0.3-0.8 seconds
- Memory usage: ~2-4GB
- Throughput: 50-100 tokens/second

### For Llama 3.1 8B Instruct Q8_0
```bash
# Core settings optimized for 8B model
export JARVIS_MODEL_CONTEXT_WINDOW="6144"
export JARVIS_N_THREADS="10"
export JARVIS_N_GPU_LAYERS="-1"
export JARVIS_N_BATCH="512"
export JARVIS_N_UBATCH="512"
export JARVIS_F16_KV="true"
export JARVIS_MUL_MAT_Q="true"
export JARVIS_ROPE_SCALING_TYPE="0"

# Memory management (more conservative for larger model)
export JARVIS_ENABLE_CONTEXT_CACHE="true"
export JARVIS_MAX_CACHE_SIZE="100"

# Inference settings
export JARVIS_MAX_TOKENS="6000"
export JARVIS_TOP_P="0.95"
export JARVIS_TOP_K="40"
export JARVIS_REPEAT_PENALTY="1.1"
export JARVIS_MIROSTAT_MODE="0"

# Metal acceleration
export LLAMA_METAL="true"
```

**Expected Performance:**
- Average latency: 0.8-2.0 seconds
- Memory usage: ~8-12GB
- Throughput: 20-40 tokens/second

## Quick Setup Scripts

### For 3B Model
```bash
#!/bin/bash
# Save as setup_3b.sh
export JARVIS_MODEL_CONTEXT_WINDOW="8192"
export JARVIS_N_THREADS="10"
export JARVIS_N_GPU_LAYERS="-1"
export JARVIS_N_BATCH="1024"
export JARVIS_N_UBATCH="1024"
export JARVIS_F16_KV="true"
export JARVIS_MUL_MAT_Q="true"
export JARVIS_ROPE_SCALING_TYPE="0"
export JARVIS_ENABLE_CONTEXT_CACHE="true"
export JARVIS_MAX_CACHE_SIZE="200"
export JARVIS_MAX_TOKENS="8000"
export JARVIS_TOP_P="0.95"
export JARVIS_TOP_K="40"
export JARVIS_REPEAT_PENALTY="1.1"
export JARVIS_MIROSTAT_MODE="0"
export LLAMA_METAL="true"
echo "✅ 3B model configuration loaded"
```

### For 8B Model
```bash
#!/bin/bash
# Save as setup_8b.sh
export JARVIS_MODEL_CONTEXT_WINDOW="6144"
export JARVIS_N_THREADS="10"
export JARVIS_N_GPU_LAYERS="-1"
export JARVIS_N_BATCH="512"
export JARVIS_N_UBATCH="512"
export JARVIS_F16_KV="true"
export JARVIS_MUL_MAT_Q="true"
export JARVIS_ROPE_SCALING_TYPE="0"
export JARVIS_ENABLE_CONTEXT_CACHE="true"
export JARVIS_MAX_CACHE_SIZE="100"
export JARVIS_MAX_TOKENS="6000"
export JARVIS_TOP_P="0.95"
export JARVIS_TOP_K="40"
export JARVIS_REPEAT_PENALTY="1.1"
export JARVIS_MIROSTAT_MODE="0"
export LLAMA_METAL="true"
echo "✅ 8B model configuration loaded"
```

## Usage Examples

### Load 3B Model Configuration
```bash
source setup_3b.sh
python your_app.py
```

### Load 8B Model Configuration
```bash
source setup_8b.sh
python your_app.py
```

### Test Performance
```bash
# Test 3B model
source setup_3b.sh
python test_latency.py /path/to/llama-3.2-3b-instruct-q4_k_m.gguf --iterations 5

# Test 8B model
source setup_8b.sh
python test_latency.py /path/to/llama-3.1-8b-instruct-q8_0.gguf --iterations 5
```

## Troubleshooting

### If You Experience Memory Issues
```bash
# Reduce context window
export JARVIS_MODEL_CONTEXT_WINDOW="4096"

# Reduce batch sizes
export JARVIS_N_BATCH="256"
export JARVIS_N_UBATCH="256"

# Reduce cache size
export JARVIS_MAX_CACHE_SIZE="50"
```

### If You Experience Slow Performance
```bash
# Increase batch sizes (if memory allows)
export JARVIS_N_BATCH="1024"
export JARVIS_N_UBATCH="1024"

# Increase context window
export JARVIS_MODEL_CONTEXT_WINDOW="8192"

# Enable more aggressive caching
export JARVIS_MAX_CACHE_SIZE="200"
```

### If You Experience Latency Spikes
```bash
# Ensure Metal is enabled
export LLAMA_METAL="true"

# Use consistent seed
export JARVIS_SEED="42"

# Disable mirostat
export JARVIS_MIROSTAT_MODE="0"

# Enable context cache
export JARVIS_ENABLE_CONTEXT_CACHE="true"
```

## Monitoring

### Check Current Configuration
```bash
python -c "
import os
jarvis_vars = [k for k in os.environ.keys() if k.startswith('JARVIS_')]
for var in sorted(jarvis_vars):
    print(f'{var}={os.environ[var]}')
"
```

### Monitor Performance
```bash
# Run latency test
python test_latency.py /path/to/model.gguf --iterations 3

# Monitor system resources
htop
sudo powermetrics --samplers gpu_power,cpu_power --sample-count 1 -n 1
```
