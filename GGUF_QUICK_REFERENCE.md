# GGUF Backend Quick Reference

## All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_MODEL_CONTEXT_WINDOW` | `4096` | Context window size (tokens) |
| `JARVIS_N_THREADS` | `10` | Number of CPU threads |
| `JARVIS_N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `JARVIS_SEED` | `42` | Random seed |
| `JARVIS_VERBOSE` | `false` | Enable verbose logging |
| `JARVIS_N_BATCH` | `512` | Batch size for GPU |
| `JARVIS_N_UBATCH` | `512` | Micro batch size |
| `JARVIS_F16_KV` | `true` | F16 precision for KV cache |
| `JARVIS_MUL_MAT_Q` | `true` | Optimized matrix multiplication |
| `JARVIS_ROPE_SCALING_TYPE` | `0` | RoPE scaling (0=disabled) |
| `JARVIS_ENABLE_CONTEXT_CACHE` | `true` | Enable context caching |
| `JARVIS_MAX_CACHE_SIZE` | `100` | Max cached contexts |
| `JARVIS_MAX_TOKENS` | `7000` | Max tokens to generate |
| `JARVIS_TOP_P` | `0.95` | Top-p sampling |
| `JARVIS_TOP_K` | `40` | Top-k sampling |
| `JARVIS_REPEAT_PENALTY` | `1.1` | Repeat penalty |
| `JARVIS_MIROSTAT_MODE` | `0` | Mirostat mode (0=disabled) |
| `JARVIS_MIROSTAT_TAU` | `5.0` | Mirostat tau |
| `JARVIS_MIROSTAT_ETA` | `0.1` | Mirostat eta |

## Mac Studio M4 Max Optimized Settings

### Llama 3.2 3B Instruct Q4_K_M
```bash
source setup_3b.sh
```
- Context: 8192 tokens
- Batch: 1024
- Cache: 200 entries
- Expected: 0.3-0.8s latency

### Llama 3.1 8B Instruct Q8_0
```bash
source setup_8b.sh
```
- Context: 6144 tokens
- Batch: 512
- Cache: 100 entries
- Expected: 0.8-2.0s latency

## Quick Commands

```bash
# Load configuration
source setup_3b.sh    # For 3B model
source setup_8b.sh     # For 8B model

# Test performance
python test_latency.py /path/to/model.gguf --iterations 3

# Check current settings
python -c "import os; [print(f'{k}={os.environ[k]}') for k in sorted(os.environ.keys()) if k.startswith('JARVIS_')]"

# Monitor system
htop
sudo powermetrics --samplers gpu_power,cpu_power --sample-count 1 -n 1
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory errors | Reduce `JARVIS_N_BATCH`, `JARVIS_MODEL_CONTEXT_WINDOW` |
| Slow performance | Increase `JARVIS_N_BATCH`, enable `JARVIS_ENABLE_CONTEXT_CACHE` |
| Latency spikes | Set `LLAMA_METAL=true`, use consistent `JARVIS_SEED` |
| High memory usage | Reduce `JARVIS_MAX_CACHE_SIZE`, `JARVIS_MODEL_CONTEXT_WINDOW` |
