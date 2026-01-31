# GGUF Backend Optimization Guide

## Environment Variables for Latency Optimization

### Core Settings
```bash
# Context window - increase for better performance, but uses more memory
export JARVIS_MODEL_CONTEXT_WINDOW="4096"

# Thread count - should match your CPU cores
export JARVIS_N_THREADS="8"

# GPU layers - use -1 for all layers, or a specific number to balance CPU/GPU
export JARVIS_N_GPU_LAYERS="-1"

# Context caching - enables prefix matching optimization
export JARVIS_ENABLE_CONTEXT_CACHE="true"

# Maximum cache size - adjust based on available memory
export JARVIS_MAX_CACHE_SIZE="100"
```

### Advanced GPU Settings
```bash
# Metal acceleration (macOS)
export LLAMA_METAL="true"

# CUDA settings (Linux/Windows)
export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING="0"
```

### Memory Management
```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH="true"
```

## Common Latency Spike Causes

### 1. Context Window Issues
- **Problem**: Small context window (512) causing frequent reloads
- **Solution**: Increase to 4096 or higher
- **Trade-off**: More memory usage

### 2. GPU Memory Pressure
- **Problem**: All layers on GPU causing memory swapping
- **Solution**: Use fewer GPU layers or increase batch size
- **Monitor**: Watch GPU memory usage

### 3. Thread Contention
- **Problem**: Too many threads competing for resources
- **Solution**: Match thread count to CPU cores
- **Test**: Try different thread counts

### 4. Context Cache Misses
- **Problem**: No context reuse between similar requests
- **Solution**: Enable context caching
- **Monitor**: Check cache hit rates

## Performance Monitoring

### Run the Latency Test
```bash
# Basic test
python test_latency.py /path/to/your/model.gguf

# Concurrent test
python test_latency.py /path/to/your/model.gguf --concurrent

# Multiple iterations
python test_latency.py /path/to/your/model.gguf --iterations 5
```

### Monitor System Resources
```bash
# GPU memory usage
nvidia-smi -l 1

# CPU and memory
htop

# Power metrics (macOS)
sudo powermetrics --samplers gpu_power,cpu_power --sample-count 1 -n 1
```

## Troubleshooting

### If You Still Get Spikes:

1. **Check GPU Memory**
   ```bash
   nvidia-smi
   # Look for memory fragmentation or swapping
   ```

2. **Monitor Context Loading**
   - Look for "load time" in debug output
   - Should be minimal after first request

3. **Test Different Configurations**
   ```bash
   # Try CPU-only
   export JARVIS_N_GPU_LAYERS="0"
   
   # Try fewer threads
   export JARVIS_N_THREADS="4"
   
   # Try smaller context
   export JARVIS_MODEL_CONTEXT_WINDOW="2048"
   ```

4. **Check for Memory Leaks**
   - Monitor memory usage over time
   - Clear cache periodically if needed

## Expected Performance

With optimizations, you should see:
- **Average latency**: 0.5-2.0 seconds
- **Latency spikes**: <5 seconds
- **Consistency**: Standard deviation <50% of mean
- **Cache hit rate**: >80% for repeated patterns

## Model-Specific Optimizations

### Small Models (<7B parameters)
- Use all GPU layers
- Larger context windows
- Higher thread counts

### Large Models (>13B parameters)
- Balance GPU/CPU layers
- Smaller context windows
- Conservative thread counts
- Enable context caching
