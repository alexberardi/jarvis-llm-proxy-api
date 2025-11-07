# Cross-Platform Setup Guide

This guide will help you set up Jarvis LLM Proxy API on different operating systems with appropriate hardware acceleration.

## Quick Start

### 1. Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd jarvis-llm-proxy-api

# Run the setup script to configure your platform
./setup.sh

# Start the development server (installs dependencies automatically)
./run.sh
```

### 2. Production Setup

```bash
# For production deployment
./run-prod.sh
```

## Supported Platforms

### Linux
- **CPU**: Standard CPU-only inference
- **CUDA**: NVIDIA GPU acceleration (requires CUDA toolkit)
- **ROCm**: AMD GPU acceleration (experimental, requires ROCm)

### macOS
- **CPU**: Standard CPU-only inference
- **Metal**: Apple Silicon GPU acceleration (M1/M2/M3/M4 chips)

### Windows (Future)
- **CPU**: Standard CPU-only inference  
- **CUDA**: NVIDIA GPU acceleration (planned)

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (for 3B models)
- **Storage**: 5GB free space
- **Python**: 3.11+

### Recommended Requirements
- **RAM**: 16GB+ (for 8B+ models)
- **GPU**: 8GB+ VRAM for GPU acceleration
- **Storage**: 20GB+ for multiple models

## Platform-Specific Setup

### Linux with NVIDIA GPU (CUDA)

1. **Install CUDA Toolkit** (if not already installed):
   
   **Option A: Ubuntu Repository (Recommended - Stable)**
   ```bash
   # Ubuntu/Debian - uses tested, stable versions
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   ```
   
   **Option B: NVIDIA Official Repository (Latest Versions)**
   ```bash
   # For more recent CUDA versions, use NVIDIA's official repository
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit
   ```
   
   **Verify installation:**
   ```bash
   nvcc --version
   nvidia-smi
   ```
   
   **If you encounter C++ compiler compatibility issues:**
   ```bash
   # Install compatible GCC version for CUDA
   sudo apt install gcc-10 g++-10
   
   # Set environment variables for compilation (temporary)
   export CC=/usr/bin/gcc-10
   export CXX=/usr/bin/g++-10
   export CUDAHOSTCXX=/usr/bin/g++-10
   
   # To make these permanent, add to ~/.bashrc:
   echo 'export CC=/usr/bin/gcc-10' >> ~/.bashrc
   echo 'export CXX=/usr/bin/g++-10' >> ~/.bashrc  
   echo 'export CUDAHOSTCXX=/usr/bin/g++-10' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Run setup**:
   ```bash
   ./setup.sh
   # Select option 2 (CUDA) when prompted
   ```

3. **Environment Configuration**:
   ```bash
   # Copy template and configure
   cp env.template .env
   # Edit .env to set JARVIS_N_GPU_LAYERS=-1 for full GPU usage
   ```

### Linux with AMD GPU (ROCm) - Experimental

1. **Install ROCm** (if not already installed):
   ```bash
   # Ubuntu 20.04/22.04
   wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   sudo apt install rocm-dev
   
   # Verify installation
   rocm-smi
   ```

2. **Run setup**:
   ```bash
   ./setup.sh
   # Select option 3 (ROCm) when prompted
   ```

### macOS with Apple Silicon

1. **Install Xcode Command Line Tools** (if not already installed):
   ```bash
   xcode-select --install
   ```

2. **Run setup**:
   ```bash
   ./setup.sh
   # Select option 2 (Metal) when prompted, or press Enter for auto-detection
   ```

3. **Environment Configuration**:
   ```bash
   cp env.template .env
   # Edit .env to set LLAMA_METAL=true
   ```

### CPU-Only Setup (All Platforms)

1. **Run setup**:
   ```bash
   ./setup.sh
   # Select option 1 (CPU only) when prompted
   ```

2. **Environment Configuration**:
   ```bash
   cp env.template .env
   # Edit .env to set JARVIS_N_GPU_LAYERS=0 and LLAMA_METAL=false
   ```

## Configuration

### Environment Variables

Key environment variables for hardware acceleration:

```bash
# GPU layers (-1 = all on GPU, 0 = CPU only, N = specific number)
JARVIS_N_GPU_LAYERS=-1

# Batch sizes (larger = faster but more memory)
JARVIS_N_BATCH=1024
JARVIS_N_UBATCH=1024

# Thread count (adjust based on your CPU)
JARVIS_N_THREADS=10

# Metal acceleration (macOS only)
LLAMA_METAL=true

# Context window size
JARVIS_MODEL_CONTEXT_WINDOW=8192
```

### Performance Tuning

#### For NVIDIA GPUs (CUDA):
```bash
JARVIS_N_GPU_LAYERS=-1      # All layers on GPU
JARVIS_N_BATCH=1024         # Large batch for throughput
JARVIS_N_UBATCH=1024        # Large micro-batch
```

#### For Apple Silicon (Metal):
```bash
JARVIS_N_GPU_LAYERS=-1      # All layers on GPU
JARVIS_N_BATCH=512          # Moderate batch size
JARVIS_N_UBATCH=512         # Moderate micro-batch
LLAMA_METAL=true           # Enable Metal
```

#### For AMD GPUs (ROCm):
```bash
JARVIS_N_GPU_LAYERS=-1      # All layers on GPU
JARVIS_N_BATCH=512          # Conservative batch size
JARVIS_N_UBATCH=512         # Conservative micro-batch
```

#### For CPU Only:
```bash
JARVIS_N_GPU_LAYERS=0       # Force CPU usage
JARVIS_N_BATCH=256          # Smaller batch for CPU
JARVIS_N_UBATCH=256         # Smaller micro-batch
JARVIS_N_THREADS=8          # Adjust based on CPU cores
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# If missing, install CUDA toolkit
sudo apt install nvidia-cuda-toolkit
```

#### 2. Metal Not Working (macOS)
```bash
# Ensure you're on Apple Silicon
uname -m  # Should show "arm64"

# Check Metal support
system_profiler SPDisplaysDataType | grep Metal
```

#### 3. ROCm Issues (Linux)
```bash
# Check ROCm installation
rocm-smi

# Verify GPU detection
rocminfo | grep "Agent"
```

#### 4. Model Format Issues
```bash
# Error: "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking"
# This means your model uses an outdated quantization format
```

**Solution:** Download a model with a supported quantization format:
```bash
# Example: Download Llama 3.2 3B with Q4_K_M quantization (recommended)
cd .models
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Update your .env file to use the new model
# Change: JARVIS_MODEL_NAME=.models/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf
# To:     JARVIS_MODEL_NAME=.models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Supported quantization formats:** `Q4_0`, `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`

#### 5. Memory Issues
- Reduce `JARVIS_MODEL_CONTEXT_WINDOW`
- Decrease `JARVIS_N_BATCH` and `JARVIS_N_UBATCH`
- Use smaller models (3B instead of 8B+)
- Set `JARVIS_N_GPU_LAYERS` to a lower value

#### 6. Slow Performance
- Increase `JARVIS_N_BATCH` if you have enough memory
- Ensure GPU acceleration is working (check logs)
- Use quantized models (Q4_K_M instead of Q8_0)

### Verification

Check if your setup is working correctly:

```bash
# Start the server
./run.sh

# In another terminal, test the API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jarvis-llm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

Check the server logs for:
- GPU detection messages
- Model loading confirmation
- Performance metrics

## Model Management

### Recommended Models by Platform

#### For GPU Systems (8GB+ VRAM):
- **Llama 3.1 8B Q4_K_M** - Good balance
- **Qwen 2.5 14B Q3_K_XL** - Higher quality
- **Mistral Nemo 12B Q4_K_M** - Fast inference

#### For CPU or Limited GPU:
- **Llama 3.2 3B Q4_K_M** - Fast and lightweight
- **Mistral 7B Q2_K** - Very fast but lower quality

#### For High-End Systems (16GB+ VRAM):
- **Qwen 2.5 14B Q6_K** - High quality
- **Llama 3.1 8B Q8_0** - Maximum quality for 8B

### Model Download

Models should be placed in the `.models/` directory:

```bash
mkdir -p .models
cd .models

# Example: Download a model (replace with your preferred model)
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

## Inference Engine Selection

### llama.cpp vs vLLM

Choose your inference engine based on your use case:

#### llama.cpp (Default)
- **Best for**: General use, development, single requests
- **Pros**: Stable, well-tested, lower memory usage, good for GGUF models
- **Cons**: Lower throughput for batch processing
- **Setup**: `JARVIS_INFERENCE_ENGINE=llama_cpp` (default)

#### vLLM 
- **Best for**: High-throughput production, batch processing, API serving
- **Pros**: Optimized for speed, excellent batching, continuous batching
- **Cons**: Higher memory usage, requires more GPU memory
- **Setup**: `JARVIS_INFERENCE_ENGINE=vllm`
- **Requirements**: NVIDIA GPU with 8GB+ VRAM recommended

### vLLM Configuration

When using vLLM, configure these environment variables:

```bash
# Enable vLLM
JARVIS_INFERENCE_ENGINE=vllm

# vLLM-specific settings
JARVIS_VLLM_TENSOR_PARALLEL_SIZE=1        # Number of GPUs
JARVIS_VLLM_GPU_MEMORY_UTILIZATION=0.9    # GPU memory usage (0.0-1.0)
JARVIS_VLLM_MAX_MODEL_LEN=4096            # Context window
```

### Model Compatibility

| Engine | GGUF Models | HuggingFace Models | Notes |
|--------|-------------|-------------------|--------|
| llama.cpp | ✅ Native | ❌ No | Optimized for GGUF format |
| vLLM | ✅ Supported | ✅ Native | Better with HuggingFace format |

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Model | Tokens/sec | Memory Usage |
|----------|-------|------------|--------------|
| RTX 4090 | Llama 3.1 8B Q4_K_M | 80-120 | 6GB |
| RTX 3080 | Llama 3.1 8B Q4_K_M | 50-80 | 6GB |
| M2 Pro | Llama 3.1 8B Q4_K_M | 30-50 | 8GB |
| M1 | Llama 3.2 3B Q4_K_M | 40-60 | 4GB |
| CPU (16 cores) | Llama 3.2 3B Q4_K_M | 5-15 | 4GB |

## Support

If you encounter issues:

1. Check the setup configuration: `cat .setup_config`
2. Review the server logs for error messages
3. Verify your hardware acceleration is working
4. Try a smaller model if you're running out of memory
5. Check the troubleshooting section above

For additional help, please refer to the project documentation or create an issue in the repository.
