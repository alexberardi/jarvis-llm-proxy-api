#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

# Parse arguments
AUTO_MODE="${AUTO_SETUP:-false}"
for arg in "$@"; do
    case "$arg" in
        --auto) AUTO_MODE=true ;;
    esac
done

echo -e "${BLUE}🚀 Jarvis LLM Proxy API Setup${NC}"
echo "=================================="

# Detect OS
OS="unknown"
ARCH="$(uname -m)"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo -e "${GREEN}Detected OS: $OS${NC}"
echo -e "${GREEN}Architecture: $ARCH${NC}"

# Detect GPU capabilities
GPU_TYPE="none"
GPU_INFO=""

if [[ "$OS" == "linux" ]]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_TYPE="cuda"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "NVIDIA GPU")
    # Check for AMD GPU
    elif command -v rocm-smi >/dev/null 2>&1; then
        GPU_TYPE="rocm"
        GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1 || echo "AMD GPU")
    fi
elif [[ "$OS" == "macos" ]]; then
    # Check for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        GPU_TYPE="metal"
        GPU_INFO="Apple Silicon (Metal)"
    fi
elif [[ "$OS" == "windows" ]]; then
    # Check for NVIDIA GPU on Windows
    if command -v nvidia-smi.exe >/dev/null 2>&1; then
        GPU_TYPE="cuda"
        GPU_INFO=$(nvidia-smi.exe --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "NVIDIA GPU")
    fi
fi

if [[ "$GPU_TYPE" != "none" ]]; then
    echo -e "${GREEN}Detected GPU: $GPU_INFO ($GPU_TYPE)${NC}"
else
    echo -e "${YELLOW}No GPU acceleration detected - will use CPU${NC}"
fi

echo ""
echo -e "${BLUE}Hardware Acceleration Options:${NC}"
echo "1) CPU only (no acceleration)"
if [[ "$OS" == "linux" ]]; then
    echo "2) CUDA (NVIDIA GPUs)"
    echo "3) ROCm (AMD GPUs) - experimental"
elif [[ "$OS" == "macos" ]]; then
    echo "2) Metal (Apple Silicon)"
elif [[ "$OS" == "windows" ]]; then
    echo "2) CUDA (NVIDIA GPUs)"
fi

echo ""
if [[ "$AUTO_MODE" == "true" ]]; then
    choice=""
else
    read -p "Select hardware acceleration (1-3, or press Enter for auto-detect): " choice
fi

# Auto-detect if no choice made
if [[ -z "$choice" ]]; then
    case "$GPU_TYPE" in
        "cuda") choice="2" ;;
        "metal") choice="2" ;;
        "rocm") choice="3" ;;
        *) choice="1" ;;
    esac
    echo -e "${GREEN}Auto-selected option $choice based on detected hardware${NC}"
fi

# Set acceleration type based on choice
ACCELERATION="cpu"
case "$choice" in
    1)
        ACCELERATION="cpu"
        echo -e "${YELLOW}Selected: CPU only${NC}"
        ;;
    2)
        if [[ "$OS" == "linux" || "$OS" == "windows" ]]; then
            ACCELERATION="cuda"
            echo -e "${GREEN}Selected: CUDA acceleration${NC}"
        elif [[ "$OS" == "macos" ]]; then
            ACCELERATION="metal"
            echo -e "${GREEN}Selected: Metal acceleration${NC}"
        fi
        ;;
    3)
        if [[ "$OS" == "linux" ]]; then
            ACCELERATION="rocm"
            echo -e "${PURPLE}Selected: ROCm acceleration (experimental)${NC}"
        else
            echo -e "${RED}ROCm is only available on Linux${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Ensure Docker can pass GPUs into containers (required for vLLM in Docker)
if [[ "$ACCELERATION" == "cuda" ]] && command -v docker >/dev/null 2>&1; then
    DOCKER_GPU_OK=false
    if docker info 2>/dev/null | grep -q "nvidia"; then
        DOCKER_GPU_OK=true
    fi

    if [[ "$DOCKER_GPU_OK" == "true" ]]; then
        echo -e "${GREEN}✅ Docker GPU passthrough already configured${NC}"
    elif [[ "$OS" == "linux" ]]; then
        echo ""
        echo -e "${YELLOW}NVIDIA Container Toolkit not found.${NC}"
        echo -e "${YELLOW}Docker needs this to pass GPUs into containers (required for vLLM in Docker).${NC}"
        if [[ "$AUTO_MODE" == "true" ]]; then
            install_ctk="Y"
        else
            read -p "Install NVIDIA Container Toolkit? [Y/n]: " install_ctk
            install_ctk=${install_ctk:-Y}
        fi
        if [[ "$install_ctk" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Installing NVIDIA Container Toolkit...${NC}"
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
            sudo apt-get update -qq
            sudo apt-get install -y nvidia-container-toolkit
            sudo nvidia-ctk runtime configure --runtime=docker
            sudo systemctl restart docker
            echo -e "${GREEN}✅ NVIDIA Container Toolkit installed and Docker configured${NC}"
        else
            echo -e "${YELLOW}Skipping. Docker will not be able to use the GPU.${NC}"
            echo -e "${YELLOW}You can install later:${NC}"
            echo -e "${YELLOW}  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html${NC}"
        fi
    elif [[ "$OS" == "windows" ]]; then
        echo ""
        echo -e "${YELLOW}Docker Desktop GPU passthrough not detected.${NC}"
        echo -e "${YELLOW}Docker Desktop uses WSL2 for GPU access. Please ensure:${NC}"
        echo -e "${YELLOW}  1. WSL2 backend is enabled in Docker Desktop settings${NC}"
        echo -e "${YELLOW}  2. WSL kernel is up to date: ${NC}${BLUE}wsl --update${NC}"
        echo -e "${YELLOW}  3. NVIDIA drivers are up to date (Game Ready or Studio)${NC}"
        echo -e "${YELLOW}  See: https://docs.docker.com/desktop/features/gpu/${NC}"
    fi
fi

# Create configuration file
CONFIG_FILE="$ROOT/.setup_config"
cat > "$CONFIG_FILE" << EOF
# Auto-generated setup configuration
OS=$OS
ARCH=$ARCH
GPU_TYPE=$GPU_TYPE
ACCELERATION=$ACCELERATION
SETUP_DATE="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
EOF

echo -e "${GREEN}✅ Configuration saved to .setup_config${NC}"
echo ""

# Set up version-locked GGUF LoRA converter (matches llama-cpp-python version)
VENDOR_CONVERTER="$ROOT/scripts/vendor/llama.cpp/convert_lora_to_gguf.py"
if [[ ! -f "$VENDOR_CONVERTER" ]]; then
    echo -e "${YELLOW}GGUF LoRA converter not found (needed for adapter training with GGUF models).${NC}"
    if [[ "$AUTO_MODE" == "true" ]]; then
        setup_converter="Y"
    else
        read -p "Download version-locked converter? [Y/n]: " setup_converter
        setup_converter=${setup_converter:-Y}
    fi
    if [[ "$setup_converter" =~ ^[Yy]$ ]]; then
        if command -v git >/dev/null 2>&1; then
            echo -e "${BLUE}Setting up GGUF converter (version-locked to llama-cpp-python)...${NC}"
            bash "$ROOT/scripts/vendor/setup_llama_cpp.sh"
            echo -e "${GREEN}✅ GGUF converter installed${NC}"
        else
            echo -e "${RED}git not found. Please install git or run scripts/vendor/setup_llama_cpp.sh manually.${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping GGUF converter setup. Adapter training will produce PEFT-only format.${NC}"
    fi
fi

# Set up llama-quantize (needed for GGUF quantization after model conversion)
if ! command -v llama-quantize >/dev/null 2>&1; then
    echo ""
    echo -e "${YELLOW}llama-quantize not found (needed to quantize GGUF models, e.g. f16 → Q4_K_M).${NC}"
    echo -e "${YELLOW}Without it, GGUF conversion will output unquantized f16 files (~15 GiB for 8B models).${NC}"
    if [[ "$OS" == "macos" ]]; then
        if command -v brew >/dev/null 2>&1; then
            if [[ "$AUTO_MODE" == "true" ]]; then
                install_llama="Y"
            else
                read -p "Install llama.cpp via Homebrew? (provides llama-quantize) [Y/n]: " install_llama
                install_llama=${install_llama:-Y}
            fi
            if [[ "$install_llama" =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}Installing llama.cpp...${NC}"
                brew install llama.cpp
                echo -e "${GREEN}✅ llama-quantize installed${NC}"
            else
                echo -e "${YELLOW}Skipping. You can install later: brew install llama.cpp${NC}"
            fi
        else
            echo -e "${YELLOW}Homebrew not found. Install llama.cpp manually: brew install llama.cpp${NC}"
        fi
    elif [[ "$OS" == "linux" ]]; then
        echo -e "${YELLOW}Install llama.cpp to get llama-quantize:${NC}"
        echo -e "${YELLOW}  Option 1: Build from source — git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && make${NC}"
        echo -e "${YELLOW}  Option 2: Set JARVIS_LLAMA_QUANTIZE_CMD to point to your llama-quantize binary${NC}"
    fi
else
    echo -e "${GREEN}✅ llama-quantize found: $(which llama-quantize)${NC}"
fi

echo -e "${BLUE}Next steps:${NC}"
echo "1. Run './run.sh' to install dependencies and start the development server"
echo "2. Or run './run-prod.sh' to start the production server"
echo ""
echo -e "${YELLOW}Note: The run scripts will automatically install the appropriate version of llama-cpp-python based on your configuration.${NC}"
