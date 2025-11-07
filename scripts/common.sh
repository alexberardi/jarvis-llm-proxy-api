#!/usr/bin/env bash
# Common functions and setup for run scripts

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Get script root directory (parent of scripts directory)
get_root_dir() {
    local script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" >/dev/null 2>&1 && pwd)"
    # If we're in the scripts directory, go up one level
    if [[ "$(basename "$script_dir")" == "scripts" ]]; then
        dirname "$script_dir"
    else
        echo "$script_dir"
    fi
}

# Initialize common variables
init_common_vars() {
    ROOT="${ROOT:-$(get_root_dir)}"
    cd "$ROOT"
    
    PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
    VENV="$ROOT/venv"
    PY="$VENV/bin/python"
    PIP="$VENV/bin/pip"
    SETUP_CONFIG="$ROOT/.setup_config"
}

# Check and run setup if needed
check_setup() {
    if [[ ! -f "$SETUP_CONFIG" ]]; then
        echo -e "${YELLOW}⚠️  Setup configuration not found. Running setup first...${NC}"
        ./setup.sh
    fi
    
    # Load setup configuration
    source "$SETUP_CONFIG"
    echo -e "${GREEN}🔧 Using configuration: OS=$OS, Acceleration=$ACCELERATION${NC}"
}

# Create virtual environment (different strategies for dev vs prod)
create_venv_dev() {
    # Development: try pyenv first, fallback to system python
    BASE_PY=""
    if command -v pyenv >/dev/null 2>&1; then
        # install if missing, but non-interactively (-s = skip if already installed)
        PYENV_NONINTERACTIVE=1 pyenv install -s "$PYTHON_VERSION"
        BASE_PY="$(pyenv prefix "$PYTHON_VERSION")/bin/python"
    fi
    if [[ -z "${BASE_PY}" ]]; then
        # fallback to system/Homebrew python3
        BASE_PY="$(command -v python3)"
    fi
    if [[ -z "${BASE_PY}" ]]; then
        echo "❌ No usable python found (neither pyenv nor python3)."; exit 1
    fi
    
    # Create venv if missing
    if [[ ! -x "$PY" ]]; then
        echo "📦 Creating virtual environment with $BASE_PY"
        "$BASE_PY" -m venv "$VENV"
    fi
}

create_venv_prod() {
    # Production: use system python only
    if [[ ! -d "$VENV" ]]; then
        echo -e "${BLUE}📦 Creating virtual environment...${NC}"
        python3 -m venv "$VENV"
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    fi
    
    # Verify Python executable
    if [[ ! -x "$PY" ]]; then
        echo -e "${RED}❌ Python executable not found at $PY${NC}"
        exit 1
    fi
}

# Load environment variables
load_env() {
    local env_file="${1:-.env}"
    
    # Load environment variables early for dependency detection
    if [[ -f "$env_file" ]]; then
        echo -e "${BLUE}📄 Loading environment from $env_file${NC}"
        set -a  # automatically export all variables
        source "$env_file"
        set +a
    else
        echo -e "${YELLOW}⚠️  Environment file $env_file not found, using defaults${NC}"
    fi
}

# Install base requirements
install_base_requirements() {
    echo -e "${BLUE}📦 Installing base requirements${NC}"
    "$PIP" install -U pip setuptools
    "$PIP" install -r requirements-base.txt
}

# Check if llama-cpp-python needs to be installed/updated
should_install_llama_cpp() {
    local acceleration_type="$1"
    
    # Check if llama-cpp-python is installed
    if ! "$PIP" show llama-cpp-python >/dev/null 2>&1; then
        echo "true"  # Not installed
        return
    fi
    
    # Check if it was built with the correct acceleration
    local current_info
    current_info=$("$PY" -c "
try:
    import llama_cpp
    print('installed')
    # Try to detect build configuration
    try:
        # This will fail if not built with CUDA
        if hasattr(llama_cpp.llama_cpp, 'GGML_USE_CUDA'):
            print('cuda')
        elif hasattr(llama_cpp.llama_cpp, 'GGML_USE_METAL'):
            print('metal')
        elif hasattr(llama_cpp.llama_cpp, 'GGML_USE_HIPBLAS'):
            print('rocm')
        else:
            print('cpu')
    except:
        print('unknown')
except ImportError:
    print('not_installed')
" 2>/dev/null || echo "not_installed")
    
    if [[ "$current_info" == *"$acceleration_type"* ]]; then
        echo "false"  # Already installed with correct acceleration
    else
        echo "true"   # Needs reinstall
    fi
}

# Install llama-cpp-python with appropriate acceleration
install_llama_cpp() {
    local acceleration_type="$1"
    local cmake_args=""
    
    case "$acceleration_type" in
        "cuda")
            cmake_args="-DGGML_CUDA=on"
            ;;
        "metal")
            cmake_args="-DGGML_METAL=on"
            ;;
        "rocm")
            cmake_args="-DGGML_HIPBLAS=on"
            ;;
        "cpu"|*)
            cmake_args=""
            ;;
    esac
    
    echo -e "${YELLOW}🔄 Installing/updating llama-cpp-python for $acceleration_type acceleration...${NC}"
    "$PIP" uninstall -y llama-cpp-python || true
    
    # Set GCC 10 for CUDA compatibility (Ubuntu CUDA + GCC 11 issue)
    if [[ "$acceleration_type" == "cuda" ]]; then
        if ! command -v gcc-10 >/dev/null 2>&1; then
            echo -e "${RED}❌ GCC 10 not found. Installing...${NC}"
            sudo apt update && sudo apt install -y gcc-10 g++-10
        fi
        
        export CC=gcc-10
        export CXX=g++-10
        export CUDAHOSTCXX=g++-10
        gcc_env="CC=gcc-10 CXX=g++-10 CUDAHOSTCXX=g++-10"
    else
        gcc_env=""
    fi
    
    if [[ -n "$cmake_args" ]]; then
        echo -e "${BLUE}Building llama-cpp-python with: $cmake_args${NC}"
        if [[ -n "$gcc_env" ]]; then
            echo -e "${YELLOW}Using GCC 10 for CUDA compatibility${NC}"
            env $gcc_env CMAKE_ARGS="$cmake_args" "$PIP" install --no-cache-dir llama-cpp-python
        else
            CMAKE_ARGS="$cmake_args" "$PIP" install --no-cache-dir llama-cpp-python
        fi
    else
        echo -e "${BLUE}Installing llama-cpp-python (CPU-only)${NC}"
        "$PIP" install llama-cpp-python
    fi
}

# Check if llama-cpp-python is needed based on backend configuration
needs_llama_cpp() {
    # llama-cpp-python is needed for:
    # 1. GGUF backend (always)
    # 2. llama_cpp inference engine (when using other backends)
    
    if [[ "${JARVIS_MODEL_BACKEND:-}" == "GGUF" ]] || [[ "${JARVIS_LIGHTWEIGHT_MODEL_BACKEND:-}" == "GGUF" ]]; then
        echo "true"
        return
    fi
    
    if [[ "${JARVIS_INFERENCE_ENGINE:-llama_cpp}" == "llama_cpp" ]]; then
        echo "true"
        return
    fi
    
    echo "false"
}

# Install conditional requirements based on environment variables
install_conditional_requirements() {
    # Check if we need transformers backend
    if [[ "${JARVIS_MODEL_BACKEND:-}" == "TRANSFORMERS" ]] || [[ "${JARVIS_LIGHTWEIGHT_MODEL_BACKEND:-}" == "TRANSFORMERS" ]]; then
        echo -e "${BLUE}📦 Installing Transformers backend requirements${NC}"
        "$PIP" install -r requirements-transformers.txt
    fi
    
    # Check if we need vLLM
    if [[ "${JARVIS_INFERENCE_ENGINE:-}" == "vllm" ]]; then
        echo -e "${BLUE}📦 Installing vLLM requirements${NC}"
        "$PIP" install -r requirements-vllm.txt
    fi
}

# Install acceleration-specific requirements
install_acceleration_requirements() {
    local acceleration="$1"
    local should_install="$2"
    
    case "$acceleration" in
        "cuda")
            echo -e "${GREEN}Installing CUDA-accelerated llama-cpp-python...${NC}"
            if [[ "$should_install" == "true" ]]; then
                install_llama_cpp "cuda"
            else
                echo -e "${GREEN}✅ CUDA llama-cpp-python already installed${NC}"
            fi
            ;;
        "metal")
            echo -e "${GREEN}Installing Metal-accelerated requirements...${NC}"
            "$PIP" install -r requirements-metal.txt
            if [[ "$should_install" == "true" ]]; then
                install_llama_cpp "metal"
            else
                echo -e "${GREEN}✅ Metal llama-cpp-python already installed${NC}"
            fi
            ;;
        "rocm")
            echo -e "${GREEN}Installing ROCm-accelerated llama-cpp-python...${NC}"
            if [[ "$should_install" == "true" ]]; then
                install_llama_cpp "rocm"
            else
                echo -e "${GREEN}✅ ROCm llama-cpp-python already installed${NC}"
            fi
            ;;
        "cpu"|*)
            echo -e "${GREEN}Installing CPU-only llama-cpp-python...${NC}"
            if [[ "$should_install" == "true" ]]; then
                install_llama_cpp "cpu"
            else
                echo -e "${GREEN}✅ CPU llama-cpp-python already installed${NC}"
            fi
            ;;
    esac
}

# Run diagnostics
run_diagnostics() {
    # Only run llama.cpp diagnostics if llama-cpp-python is needed
    if [[ "$(needs_llama_cpp)" == "true" ]]; then
        echo -e "${BLUE}🔍 Running llama-cpp-python diagnostics...${NC}"
        "$PY" -c "
import llama_cpp
print(f'✅ llama-cpp-python version: {llama_cpp.__version__}')
try:
    # Test basic functionality
    print('🔍 Testing llama-cpp-python import and basic functionality...')
    print('✅ llama-cpp-python is working correctly')
except Exception as e:
    print(f'❌ llama-cpp-python test failed: {e}')
"
    else
        echo -e "${YELLOW}⏭️  Skipping llama-cpp-python diagnostics (not needed for current backends)${NC}"
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up processes...${NC}"
    
    # Kill vLLM processes
    pkill -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true
    
    # Clear CUDA cache if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${BLUE}🧹 Clearing CUDA cache...${NC}"
        python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('✅ CUDA cache cleared')
except ImportError:
    pass
" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set up signal handlers
setup_signal_handlers() {
    trap cleanup EXIT INT TERM
}

# Start server with appropriate configuration
start_server() {
    local enable_reload="${1:-true}"
    local host="${2:-0.0.0.0}"
    local port="${3:-8000}"
    
    echo -e "${GREEN}🚀 Starting Jarvis LLM Proxy API...${NC}"
    echo -e "${BLUE}📍 Server will be available at: http://$host:$port${NC}"
    echo -e "${BLUE}📍 Health check: http://$host:$port/health${NC}"
    echo -e "${BLUE}📍 API docs: http://$host:$port/docs${NC}"
    
    # Check if using vLLM - it doesn't work well with uvicorn workers
    local using_vllm=false
    if [[ "${JARVIS_INFERENCE_ENGINE:-}" == "vllm" ]] || [[ "${JARVIS_MODEL_BACKEND:-}" == "VLLM" ]] || [[ "${JARVIS_LIGHTWEIGHT_MODEL_BACKEND:-}" == "VLLM" ]]; then
        using_vllm=true
    fi
    
    if [[ "$enable_reload" == "true" ]]; then
        echo -e "${YELLOW}🔄 Development mode: auto-reload enabled (single worker)${NC}"
        "$VENV/bin/uvicorn" main:app --host "$host" --port "$port" --reload
    elif [[ "$using_vllm" == "true" ]]; then
        echo -e "${BLUE}🚀 vLLM mode: async single worker with high concurrency${NC}"
        echo -e "${YELLOW}   → vLLM handles request batching internally for optimal throughput${NC}"
        "$VENV/bin/uvicorn" main:app --host "$host" --port "$port" --loop asyncio --limit-concurrency 1000
    else
        echo -e "${GREEN}⚡ Production mode: using 4 workers for parallel requests${NC}"
        "$VENV/bin/uvicorn" main:app --host "$host" --port "$port" --workers 4
    fi
}
