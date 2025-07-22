# Model Reference Guide

## Context Windows and Chat Formats

### Current Models

| Model | Context Window | Chat Format | File Path | Notes |
|-------|---------------|-------------|-----------|-------|
| **Mistral Nemo Instruct 2407 Q3_K_L** | 32768 | `chatml` | `.models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q3_K_L.gguf` | Fastest, lowest quality (6.56GB) |
| **Mistral Nemo Instruct 2407 Q4_K_M** | 32768 | `chatml` | `.models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf` | Good balance (7.48GB) |
| **Mistral Nemo Instruct 2407 Q6_K** | 32768 | `chatml` | `.models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q6_K.gguf` | Higher quality, slower (10.1GB) |
| **Mistral Nemo Instruct 2407 Q8_0** | 32768 | `chatml` | `.models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q8_0.gguf` | Highest quality, slowest (13.0GB) |
| **Gemma 2 9B Instruct IQ4_XS** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-IQ4_XS.gguf` | Fastest, lowest quality (5.18GB) |
| **Gemma 2 9B Instruct Q3_K_L** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q3_K_L.gguf` | Very fast, low quality (5.13GB) |
| **Gemma 2 9B Instruct Q4_K_M** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf` | Good balance (5.76GB) |
| **Gemma 2 9B Instruct Q5_K_M** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q5_K_M.gguf` | Higher quality, slower (6.65GB) |
| **Gemma 2 9B Instruct Q6_K** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q6_K.gguf` | Higher quality, slower (7.59GB) |
| **Gemma 2 9B Instruct Q8_0** | 8192 | `chatml` | `.models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q8_0.gguf` | Highest quality, slowest (9.83GB) |
| **Mistral 7B Instruct v0.2** | 32768 | `chatml` | `.models/mistral-7b-instruct-v0.2.Q2_K.gguf` | Lightweight model (Q2_K) |
| **Qwen 2.5 14B Instruct Q3_K_XL** | 32768 | `qwen` | `.models/Qwen2.5-14B-Instruct-Q3_K_XL.gguf` | Good balance, medium size (8.6GB) |
| **Qwen 2.5 14B Instruct Q4_K_M** | 32768 | `qwen` | `.models/Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Good balance, smaller size (8.4GB) |
| **Qwen 2.5 14B Instruct Q6_K** | 32768 | `qwen` | `.models/Qwen2.5-14B-Instruct.Q6_K.gguf` | Higher quality, larger size (12.1GB) |
| **Llama 3.2 3B Instruct** | 8192 | `chatml` | `.models/Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Small, fast model |
| **Llama 3.1 8B Instruct** | 8192 | `llama3` | `.models/llama-3.1-8b-instruct/llama-3.1-8b-instruct.Q4_K_M.gguf` | Reliable JSON output |
| **Meta Llama 3 8B Instruct Q5** | 8192 | `llama3` | `.models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf` | Good balance of quality and speed |
| **Meta Llama 3 8B Instruct Q8** | 8192 | `llama3` | `.models/Meta-Llama-3-8B-Instruct-Q8_0.gguf` | Higher quality, larger file size |
| **Yi 1.5 9B Chat** | 8192 | `chatml` | `.models/yi-1.5-9b/Yi-1.5-9B-Chat-Q4_K_M.gguf` | Good instruction following |

### Potential Models to Test

| Model | Context Window | Chat Format | Notes |
|-------|---------------|-------------|-------|
| **CodeLlama 34B Instruct** | 100000 | `llama2` | Excellent for structured data |
| **DeepSeek Coder 33B** | 16384 | `deepseek` | Very good at JSON extraction |
| **Gemma 2 9B Instruct** | 8192 | `gemma` | Good instruction following |

## Chat Format Reference

### Available Chat Formats in llama-cpp-python 0.3.14:
- `chatml` - Most widely supported, good for general use
- `llama2` - Llama 2 instruction format
- `llama3` - Llama 3 instruction format  
- `mistral_instruct` - Mistral instruction format
- `mistral` - Mistral format (for older models)
- `qwen` - Qwen format
- `gemma` - Gemma format
- `alpaca` - Alpaca instruction format
- `baichuan` - Baichuan format
- `baichuan2` - Baichuan 2 format
- `chatglm3` - ChatGLM3 format
- `intel` - Intel format
- `oasst_llama` - Open Assistant Llama format
- `open_orca` - Open Orca format
- `openbuddy` - OpenBuddy format
- `openchat` - OpenChat format
- `phind` - Phind format
- `pygmalion` - Pygmalion format
- `redpajama_incite` - RedPajama Incite format
- `saiga` - Saiga format
- `snoozy` - Snoozy format
- `zephyr` - Zephyr format

## Model-Swap Examples

### Switch to Qwen 2.5 14B:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Qwen2.5-14B-Instruct.Q6_K.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "qwen", "new_model_context_window": 32768}'
```

### Switch to Mistral Nemo:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "chatml", "new_model_context_window": 32768}'
```

### Switch to Mistral Nemo (Q3_K_L - Fastest):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q3_K_L.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "chatml", "new_model_context_window": 32768}'
```

### Switch to Mistral Nemo (Q6_K - Higher Quality):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q6_K.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "chatml", "new_model_context_window": 32768}'
```

### Switch to Mistral Nemo (Q8_0 - Highest Quality):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q8_0.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "chatml", "new_model_context_window": 32768}'
```

### Switch to Gemma 2 9B Instruct (Q4_K_M - Recommended):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_M.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "gemma", "new_model_context_window": 8192}'
```

### Switch to Gemma 2 9B Instruct (IQ4_XS - Fastest):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/gemma-2-9b-it-GGUF/gemma-2-9b-it-IQ4_XS.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "gemma", "new_model_context_window": 8192}'
```

### Switch to Gemma 2 9B Instruct (Q8_0 - Highest Quality):
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q8_0.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "gemma", "new_model_context_window": 8192}'
```

### Switch to Yi 1.5 9B:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/yi-1.5-9b/Yi-1.5-9B-Chat-Q4_K_M.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "chatml", "new_model_context_window": 8192}'
```

### Switch to Llama 3.1 8B Instruct:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/llama-3.1-8b-instruct/llama-3.1-8b-instruct.Q4_K_M.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "llama3", "new_model_context_window": 8192}'
```

### Switch to Meta Llama 3 8B Instruct Q5:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "llama3", "new_model_context_window": 8192}'
```

### Switch to Meta Llama 3 8B Instruct Q8:
```bash
curl -X POST http://localhost:8000/model-swap \
  -H "Content-Type: application/json" \
  -d '{"new_model": ".models/Meta-Llama-3-8B-Instruct-Q8_0.gguf", "new_model_backend": "GGUF", "new_model_chat_format": "llama3", "new_model_context_window": 8192}'
```

## Environment Variables

### Current .env Settings:
```
JARVIS_MODEL_CONTEXT_WINDOW=8192
JARVIS_LIGHTWEIGHT_MODEL_CONTEXT_WINDOW=512
```

### Recommended Settings for Different Models:
- **Mistral Nemo**: `JARVIS_MODEL_CONTEXT_WINDOW=32768`
- **Qwen 2.5 14B**: `JARVIS_MODEL_CONTEXT_WINDOW=32768`
- **Yi 1.5 9B**: `JARVIS_MODEL_CONTEXT_WINDOW=8192`
- **Llama 3.x**: `JARVIS_MODEL_CONTEXT_WINDOW=8192`
- **Gemma 2 9B**: `JARVIS_MODEL_CONTEXT_WINDOW=8192`

## Performance Notes

- **Qwen 2.5 14B**: Excellent for JSON extraction, but slower
- **Mistral Nemo**: Good balance of speed and quality
- **Yi 1.5 9B**: Fast and good instruction following
- **Mistral 7B Q2_K**: Very fast but lower quality
- **Llama 3.2 3B**: Fastest but limited context
- **Gemma 2 9B**: Good for limited VRAM/RAM setups