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
read -p "Select hardware acceleration (1-3, or press Enter for auto-detect): " choice

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
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run './run.sh' to install dependencies and start the development server"
echo "2. Or run './run-prod.sh' to start the production server"
echo ""
echo -e "${YELLOW}Note: The run scripts will automatically install the appropriate version of llama-cpp-python based on your configuration.${NC}"
