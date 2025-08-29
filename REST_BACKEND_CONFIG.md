# REST Backend Configuration Guide

The REST backend allows you to use remote API providers like OpenAI, Anthropic, Ollama, LM Studio, and custom APIs through a unified interface.

## Environment Variables

### Required Variables

- `JARVIS_MODEL_BACKEND`: Set to `"REST"` to use REST backend for main model
- `JARVIS_LIGHTWEIGHT_MODEL_BACKEND`: Set to `"REST"` to use REST backend for lightweight model
- `JARVIS_REST_MODEL_URL`: Base URL for the main model API
- `JARVIS_REST_LIGHTWEIGHT_MODEL_URL`: Base URL for the lightweight model API

### Optional Model Name Overrides

- `JARVIS_REST_MODEL_NAME`: Override model name for main REST backend
- `JARVIS_REST_LIGHTWEIGHT_MODEL_NAME`: Override model name for lightweight REST backend

### Authentication Configuration

- `JARVIS_REST_AUTH_TYPE`: Authentication type (`"none"`, `"bearer"`, `"api_key"`, `"custom"`)
- `JARVIS_REST_AUTH_TOKEN`: Authentication token/API key
- `JARVIS_REST_AUTH_HEADER`: Custom header name for authentication (default: `"Authorization"`)

### Provider Configuration

- `JARVIS_REST_PROVIDER`: Provider type (`"openai"`, `"anthropic"`, `"ollama"`, `"lmstudio"`, `"generic"`)
- `JARVIS_REST_REQUEST_FORMAT`: Request format (`"openai"`, `"ollama"`, `"chatml"`, `"generic"`)
- `JARVIS_REST_TIMEOUT`: Request timeout in seconds (default: 60)

## Provider-Specific Configurations

### OpenAI/OpenAI-Compatible APIs

```bash
# Environment variables
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://api.openai.com
JARVIS_REST_PROVIDER=openai
JARVIS_REST_AUTH_TYPE=bearer
JARVIS_REST_AUTH_TOKEN=your_openai_api_key
JARVIS_REST_REQUEST_FORMAT=openai
```

### Anthropic Claude

```bash
# Environment variables
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://api.anthropic.com
JARVIS_REST_PROVIDER=anthropic
JARVIS_REST_AUTH_TYPE=bearer
JARVIS_REST_AUTH_TOKEN=your_anthropic_api_key
JARVIS_REST_REQUEST_FORMAT=openai
```

### Ollama (Local)

```bash
# Environment variables
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=http://localhost:11434
JARVIS_REST_PROVIDER=ollama
JARVIS_REST_AUTH_TYPE=none
JARVIS_REST_REQUEST_FORMAT=ollama
```

### LM Studio (Local)

```bash
# Environment variables
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=http://localhost:1234
JARVIS_REST_PROVIDER=lmstudio
JARVIS_REST_AUTH_TYPE=none
JARVIS_REST_REQUEST_FORMAT=openai
```

### Custom API with API Key

```bash
# Environment variables
JARVIS_MODEL_BACKEND=REST
JARVIS_REST_MODEL_URL=https://your-api.com
JARVIS_REST_PROVIDER=generic
JARVIS_REST_AUTH_TYPE=api_key
JARVIS_REST_AUTH_TOKEN=your_api_key
JARVIS_REST_AUTH_HEADER=X-API-Key
JARVIS_REST_REQUEST_FORMAT=generic
```

## Example .env File

```bash
# Main model configuration
JARVIS_MODEL_BACKEND=REST
JARVIS_MODEL_NAME=gpt-4
JARVIS_REST_MODEL_URL=https://api.openai.com
JARVIS_REST_PROVIDER=openai
JARVIS_REST_AUTH_TYPE=bearer
JARVIS_REST_AUTH_TOKEN=sk-your-openai-key-here

# Lightweight model configuration
JARVIS_LIGHTWEIGHT_MODEL_BACKEND=REST
JARVIS_LIGHTWEIGHT_MODEL_NAME=llama2:7b
JARVIS_REST_LIGHTWEIGHT_MODEL_URL=http://localhost:11434
JARVIS_REST_PROVIDER=ollama
JARVIS_REST_AUTH_TYPE=none
JARVIS_REST_REQUEST_FORMAT=ollama

# General REST settings
JARVIS_REST_TIMEOUT=60
```

## Request Format Details

### OpenAI Format
- **Endpoint**: `/v1/chat/completions`
- **Request**: `{"messages": [...], "temperature": 0.7, "model": "..."}`
- **Response**: `{"choices": [{"message": {"content": "..."}}], "usage": {...}}`

### Ollama Format
- **Endpoint**: `/api/chat`
- **Request**: `{"messages": [...], "model": "...", "options": {"temperature": 0.7}}`
- **Response**: `{"message": {"content": "..."}}`

### ChatML Format
- **Endpoint**: `/v1/chat/completions`
- **Request**: `{"prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n", "temperature": 0.7}`
- **Response**: `{"content": "..."}`

## Authentication Types

### `none`
No authentication required (e.g., local Ollama, LM Studio)

### `bearer`
Standard Bearer token authentication:
```
Authorization: Bearer your_token_here
```

### `api_key`
API key in custom header:
```
X-API-Key: your_api_key_here
```

### `custom`
Custom authentication pattern - you must set `JARVIS_REST_AUTH_HEADER` to specify the header name.

## Error Handling

The REST backend includes comprehensive error handling for:
- HTTP status errors (4xx, 5xx)
- Network/connection errors
- JSON parsing errors
- Timeout errors

All errors are logged with detailed information to help with debugging.

## Performance Features

- **Async HTTP client** for non-blocking requests
- **Configurable timeouts** to prevent hanging requests
- **Token usage tracking** when available from the provider
- **Response timing** for performance monitoring
- **Automatic cleanup** of HTTP connections

## Troubleshooting

### Common Issues

1. **Connection refused**: Check if the API server is running and the URL is correct
2. **Authentication failed**: Verify your API key and authentication type
3. **Timeout errors**: Increase `JARVIS_REST_TIMEOUT` for slower APIs
4. **Format errors**: Ensure `JARVIS_REST_REQUEST_FORMAT` matches your API

### Debug Mode

Enable debug logging by setting `DEBUG=true` in your environment variables.

### Health Check

Use the `/api/v1/health` endpoint to verify your backend configuration.

## Migration from Other Backends

To switch from MLX/GGUF to REST:

1. Set the appropriate environment variables
2. Restart the service
3. The model manager will automatically load the REST backend
4. No code changes required in your application

## Security Considerations

- Store API keys in environment variables, not in code
- Use HTTPS for production APIs
- Consider using a secrets manager for sensitive credentials
- Regularly rotate API keys
- Monitor API usage and costs
