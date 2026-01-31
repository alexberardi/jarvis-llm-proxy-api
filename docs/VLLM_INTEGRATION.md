# vLLM Integration Guide

This document explains the vLLM integration in Jarvis LLM Proxy API, providing high-performance inference capabilities.

## Overview

The project now supports two inference engines:

1. **llama.cpp** (default): Reliable, well-tested, good for development
2. **vLLM**: High-performance, optimized for production throughput

## Quick Start

### Enable vLLM

1. **Set the inference engine**:
   ```bash
   # In your .env file
   JARVIS_INFERENCE_ENGINE=vllm
   ```

2. **Install vLLM dependencies**:
   ```bash
   ./run.sh  # Automatically detects and installs vLLM
   ```

3. **Configure vLLM settings** (optional):
   ```bash
   JARVIS_VLLM_TENSOR_PARALLEL_SIZE=1        # Number of GPUs
   JARVIS_VLLM_GPU_MEMORY_UTILIZATION=0.9    # GPU memory usage
   JARVIS_VLLM_MAX_MODEL_LEN=4096            # Context window
   ```

## Architecture

### Backend Integration

The vLLM integration uses a **delegate pattern**:

- `GGUFClient` detects `JARVIS_INFERENCE_ENGINE` setting
- Routes calls to either `llama.cpp` or `VLLMClient` backend
- Maintains the same API interface for seamless switching

### File Structure

```
backends/
├── gguf_backend.py      # Main backend with engine selection
├── vllm_backend.py      # vLLM-specific implementation
└── ...

requirements-vllm.txt    # vLLM dependencies
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JARVIS_INFERENCE_ENGINE` | `llama_cpp` | Choose: `llama_cpp` or `vllm` |
| `JARVIS_VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `JARVIS_VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory utilization (0.0-1.0) |
| `JARVIS_VLLM_MAX_MODEL_LEN` | `4096` | Maximum sequence length |

### Chat Format Support

vLLM backend supports the same chat formats as llama.cpp:

- `chatml`: ChatML format with `<|im_start|>` tokens
- `llama3`: Llama 3 format with header tags
- `mistral`: Mistral instruction format
- Custom formats can be added easily

## Performance Comparison

### llama.cpp vs vLLM

| Aspect | llama.cpp | vLLM |
|--------|-----------|------|
| **Throughput** | Good for single requests | Excellent for batching |
| **Memory Usage** | Lower | Higher (more efficient batching) |
| **Latency** | Good | Excellent (continuous batching) |
| **Model Support** | GGUF optimized | HuggingFace native, GGUF supported |
| **GPU Requirements** | 4GB+ VRAM | 8GB+ VRAM recommended |

### When to Use Each

#### Use llama.cpp when:
- Development and testing
- Single-user applications
- Limited GPU memory (< 8GB)
- Working primarily with GGUF models
- Need lower memory footprint

#### Use vLLM when:
- Production API serving
- High-throughput requirements
- Multiple concurrent users
- Batch processing
- Have sufficient GPU memory (8GB+)

## Model Compatibility

### GGUF Models with vLLM

vLLM can work with GGUF models, but performance is optimized for HuggingFace format:

```bash
# GGUF model (works with both engines)
JARVIS_MODEL_NAME=.models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# HuggingFace model (vLLM optimized)
JARVIS_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
```

### Recommended Models

For vLLM, consider these high-performance models:

- **Llama 3.1 8B**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Phi-3 Mini**: `microsoft/Phi-3-mini-4k-instruct`
- **Mistral 7B**: `mistralai/Mistral-7B-Instruct-v0.3`

## Installation Details

### Automatic Installation

The run scripts automatically detect vLLM usage:

```bash
# Checks JARVIS_INFERENCE_ENGINE environment variable
./run.sh
```

### Manual Installation

```bash
# Install base requirements
pip install -r requirements-base.txt

# Install vLLM and dependencies
pip install -r requirements-vllm.txt

# Install hardware acceleration (CUDA example)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce GPU memory utilization
JARVIS_VLLM_GPU_MEMORY_UTILIZATION=0.7

# Reduce context window
JARVIS_VLLM_MAX_MODEL_LEN=2048
```

#### 2. Model Loading Errors
```bash
# Check model path and format
# vLLM prefers HuggingFace model names or paths
JARVIS_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
```

#### 3. Performance Issues
```bash
# Enable tensor parallelism for multiple GPUs
JARVIS_VLLM_TENSOR_PARALLEL_SIZE=2

# Ensure CUDA is properly installed
nvidia-smi
```

### Debug Mode

Enable verbose logging to troubleshoot:

```bash
# In your .env file
DEBUG=true
LLAMA_LOG_LEVEL=debug
```

## API Compatibility

The vLLM integration maintains full API compatibility:

- Same OpenAI-compatible endpoints
- Identical request/response formats
- Seamless switching between engines
- No client-side changes required

## Future Enhancements

Planned improvements:

1. **Streaming Support**: Real-time response streaming
2. **Multi-GPU**: Advanced tensor parallelism
3. **Model Caching**: Faster model switching
4. **Quantization**: Built-in quantization support
5. **Monitoring**: Performance metrics and monitoring

## Contributing

To contribute to vLLM integration:

1. Test with different model formats
2. Report performance benchmarks
3. Submit compatibility fixes
4. Improve documentation

## Support

For vLLM-specific issues:

1. Check vLLM logs in server output
2. Verify GPU memory and CUDA installation
3. Test with smaller models first
4. Refer to [vLLM documentation](https://docs.vllm.ai/)

The integration provides a powerful, flexible inference solution that scales from development to production deployment.
