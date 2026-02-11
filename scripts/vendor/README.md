# Vendored llama.cpp Converter Scripts

## Version Matching Requirement

The `convert_lora_to_gguf.py` converter **must** come from the same llama.cpp commit
that `llama-cpp-python` was built against. A version mismatch causes GGUF adapter
format incompatibilities that result in runtime assertion failures.

| llama-cpp-python | llama.cpp commit | Status |
|------------------|------------------|--------|
| 0.3.16           | `4227c9b`        | Current |

## Setup

Run the setup script to download the converter from the correct commit:

```bash
cd scripts/vendor
bash setup_llama_cpp.sh
```

This clones a sparse checkout of llama.cpp (only the converter scripts) at the
pinned commit into `scripts/vendor/llama.cpp/`.

### Manual setup

If the setup script doesn't work, manually clone and checkout:

```bash
cd scripts/vendor
git clone --no-checkout --filter=blob:none https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 4227c9b
git sparse-checkout set convert_lora_to_gguf.py convert_hf_to_gguf.py gguf-py
```

## How It Works

The training script (`scripts/train_adapter.py`) looks for the converter at:
1. `JARVIS_ADAPTER_GGUF_CONVERT_CMD` environment variable (if set)
2. `scripts/vendor/llama.cpp/convert_lora_to_gguf.py` (from this setup)

The converter depends on:
- `gguf` Python package (installed from `requirements-base.txt`)
- `convert_hf_to_gguf.py` (included in the sparse checkout)
- `transformers`, `torch`, `safetensors` (already project dependencies)

## Updating When Upgrading llama-cpp-python

When upgrading `llama-cpp-python` to a new version:

1. Find which llama.cpp commit the new version uses:
   ```bash
   pip show llama-cpp-python  # check version
   # Check the llama-cpp-python release notes or CMakeLists.txt for the commit
   ```

2. Update the commit in `setup_llama_cpp.sh`

3. Re-run the setup script

4. Update the version table above

5. Test adapter conversion with a known-good PEFT adapter
