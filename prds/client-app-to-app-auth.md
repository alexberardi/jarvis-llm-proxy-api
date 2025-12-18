# Client Guide: App-to-App Authentication for llm-proxy

This guide explains how **your service** (a Jarvis app) must authenticate when calling the llm-proxy API using Jarvis app-to-app headers enforced by jarvis-auth.

## What you need
- An `app_id` (e.g., `my-service`).
- The corresponding `app_key` (secret, only shown once by jarvis-auth).
- The llm-proxy base URL (e.g., `https://llm-proxy.internal`).

## Headers to send on every protected request
```
X-Jarvis-App-Id: <your-app-id>
X-Jarvis-App-Key: <your-raw-key>
```

## Protected endpoints
- `POST /v1/chat/completions` (all chat requests, text and vision)
- Any future internal endpoints that require app auth

Public (no app auth):
- `GET /health`
- `GET /health/live`
- `GET /health/ready`

## Example: call /v1/chat/completions (text)
```bash
curl -X POST https://llm-proxy.internal/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Jarvis-App-Id: $APP_ID" \
  -H "X-Jarvis-App-Key: $APP_KEY" \
  -d '{
    "model": "full",
    "messages": [{"role": "user", "content": "Hello, world"}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

## Example: call /v1/chat/completions (vision)
```bash
IMG_B64="$(base64 -i /path/to/image.png | tr -d '\n')"
curl -X POST https://llm-proxy.internal/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Jarvis-App-Id: $APP_ID" \
  -H "X-Jarvis-App-Key: $APP_KEY" \
  -d "{
    \"model\": \"vision\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${IMG_B64}\"}}
      ]
    }],
    \"temperature\": 0.2,
    \"max_tokens\": 200
  }"
```

## Failure modes
- Missing headers → `401 {"detail": "Missing app credentials"}`
- Invalid/revoked/old key → `401 {"detail": "Invalid app credentials"}`
- jarvis-auth unreachable → `502 {"detail": "Auth service unavailable: ..."}` (retry after resolving connectivity)

## Security notes
- Keep `app_key` in a secrets manager; never log it.
- Use TLS for all calls.
- Rotate keys via jarvis-auth admin APIs; update your secrets and redeploy.
- Do not cache validation; send headers on every request (jarvis-auth validates each time).

## Minimal client pseudo-code
```python
import os, requests

APP_ID = os.environ["JARVIS_APP_ID"]
APP_KEY = os.environ["JARVIS_APP_KEY"]
BASE = os.environ.get("LLM_PROXY_URL", "https://llm-proxy.internal")

def chat(messages, model="full"):
    resp = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "X-Jarvis-App-Id": APP_ID,
            "X-Jarvis-App-Key": APP_KEY,
        },
        json={"model": model, "messages": messages},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
```

## Checklist for callers
- [ ] Obtain your `app_id` and `app_key` from jarvis-auth admin (`POST /admin/app-clients`).
- [ ] Store `app_key` securely (env/secret manager); never commit or log it.
- [ ] Send both headers on every protected request to llm-proxy.
- [ ] Keep health checks unauthenticated (no headers) unless you explicitly lock them down.
- [ ] Rotate keys via jarvis-auth when needed; update secrets and redeploy. 

