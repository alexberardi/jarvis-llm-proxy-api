#!/usr/bin/env bash
set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if ! command -v lsb_release >/dev/null 2>&1; then
  echo -e "${YELLOW}lsb_release not found; installing...${NC}"
  sudo apt update
  sudo apt install -y lsb-release
fi

UBUNTU_CODENAME="$(lsb_release -cs)"
UBUNTU_VERSION="$(lsb_release -rs)"

if [[ "$UBUNTU_VERSION" != "24.04" ]]; then
  echo -e "${YELLOW}Detected Ubuntu ${UBUNTU_VERSION} (${UBUNTU_CODENAME}).${NC}"
  echo -e "${YELLOW}This script is set up for Ubuntu 24.04 repo URLs.${NC}"
  echo -e "${YELLOW}Continue at your own risk or update the repo URL before running.${NC}"
fi

echo -e "${BLUE}Adding NVIDIA CUDA repo keyring...${NC}"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb

echo -e "${BLUE}Updating apt cache...${NC}"
sudo apt update

echo -e "${BLUE}Installing CUDA Toolkit 12.6...${NC}"
sudo apt install -y cuda-toolkit-12-6

CUDA_BIN="/usr/local/cuda/bin"
CUDA_LIB="/usr/local/cuda/lib64"

if ! grep -q "$CUDA_BIN" "$HOME/.bashrc" 2>/dev/null; then
  echo -e "${BLUE}Adding CUDA paths to ~/.bashrc...${NC}"
  {
    echo ""
    echo "# CUDA toolkit"
    echo "export PATH=${CUDA_BIN}:\$PATH"
    echo "export LD_LIBRARY_PATH=${CUDA_LIB}:\${LD_LIBRARY_PATH:-}"
  } >> "$HOME/.bashrc"
fi

echo -e "${GREEN}✅ CUDA toolkit install complete.${NC}"
echo -e "${GREEN}Run: source ~/.bashrc && nvcc --version${NC}"
