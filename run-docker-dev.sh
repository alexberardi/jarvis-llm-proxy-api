#!/usr/bin/env bash
# ============================================================================
# Jarvis LLM Proxy API — Docker Development Runner
# ============================================================================
# Usage:
#   ./run-docker-dev.sh              # start (build only if no image)
#   ./run-docker-dev.sh --build      # rebuild image, then start
#   ./run-docker-dev.sh --rebuild    # full rebuild (no cache), then start
#   ./run-docker-dev.sh --down       # stop and remove containers
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.dev.yaml"

# Parse arguments
BUILD_FLAG=""
case "${1:-}" in
  --build)
    BUILD_FLAG="--build"
    ;;
  --rebuild)
    echo "🔨 Full rebuild (no cache)..."
    docker compose -f "$COMPOSE_FILE" build --no-cache
    ;;
  --down)
    docker compose -f "$COMPOSE_FILE" down
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--build|--rebuild|--down]"
    exit 1
    ;;
esac

# Check for .env file
if [[ ! -f .env ]]; then
  echo "⚠️  No .env file found. Copy the template and configure:"
  echo "   cp env.docker.template .env"
  exit 1
fi

echo "🚀 Jarvis LLM Proxy API — Docker Development Mode"
echo "📁 Compose file: $COMPOSE_FILE"
echo ""

docker compose -f "$COMPOSE_FILE" up $BUILD_FLAG
