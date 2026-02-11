#!/usr/bin/env bash
# Download llama.cpp converter scripts at the commit matching llama-cpp-python 0.3.16.
#
# This uses a sparse checkout to only download the converter scripts,
# not the entire llama.cpp repository.

set -euo pipefail

LLAMA_CPP_COMMIT="4227c9b"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/llama.cpp"

if [ -d "${TARGET_DIR}" ]; then
    echo "llama.cpp directory already exists at ${TARGET_DIR}"
    echo "To re-download, remove it first: rm -rf ${TARGET_DIR}"
    exit 0
fi

echo "Cloning llama.cpp at commit ${LLAMA_CPP_COMMIT} (sparse checkout)..."
git clone --no-checkout --filter=blob:none \
    https://github.com/ggml-org/llama.cpp.git \
    "${TARGET_DIR}"

cd "${TARGET_DIR}"
git sparse-checkout set --no-cone \
    convert_lora_to_gguf.py \
    convert_hf_to_gguf.py \
    gguf-py

git checkout "${LLAMA_CPP_COMMIT}"

echo ""
echo "Done! Converter available at:"
echo "  ${TARGET_DIR}/convert_lora_to_gguf.py"
echo ""
echo "llama.cpp commit: ${LLAMA_CPP_COMMIT}"
echo "Matches llama-cpp-python: 0.3.16"
