#!/usr/bin/env bash
# Production run script

# Load common functions
source "$(dirname "$0")/scripts/common.sh"

# Initialize common variables
init_common_vars
ENV_FILE="${ENV_FILE:-prod.env}"

# Production-specific configuration
ENABLE_RELOAD="false"  # Always disable reload in production
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
# Ensure fork-safety env for macOS workers (Objective-C libs)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}

echo -e "${GREEN}🚀 Jarvis LLM Proxy API - Production Mode${NC}"
echo -e "${BLUE}📁 Root directory: $ROOT${NC}"

# Setup and configuration
check_setup
load_env "$ENV_FILE"
configure_vllm_env

# Create virtual environment (production strategy)
create_venv_prod
echo -e "${GREEN}🐍 Using Python: $PY${NC}"

# Install requirements
install_base_requirements
install_conditional_requirements

# Install llama-cpp-python only if needed
if [[ "$(needs_llama_cpp)" == "true" ]]; then
    echo -e "${BLUE}🔍 llama-cpp-python needed for current configuration${NC}"
    should_install=$(should_install_llama_cpp "$ACCELERATION")
    install_acceleration_requirements "$ACCELERATION" "$should_install"
else
    echo -e "${YELLOW}⏭️  Skipping llama-cpp-python installation (not needed for current backends)${NC}"
fi

# Run diagnostics
run_diagnostics

# Setup cleanup handlers
setup_signal_handlers

# Start model service if configured
# Auto-disable on macOS: model service requires vLLM which is Linux-only
MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-7705}"
if [[ "$(uname)" == "Darwin" ]]; then
    RUN_MODEL_SERVICE="${RUN_MODEL_SERVICE:-false}"
else
    RUN_MODEL_SERVICE="${RUN_MODEL_SERVICE:-true}"
fi
MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"
MODEL_SERVICE_TOKEN_EXPORT="${MODEL_SERVICE_TOKEN:-${LLM_PROXY_INTERNAL_TOKEN:-}}"

# Start production server
( start_server "$ENABLE_RELOAD" "$SERVER_HOST" "$SERVER_PORT" ) | sed -e 's/^/[api] /' &
API_PID=$!

MODEL_PID=""
if [[ "$RUN_MODEL_SERVICE" == "true" ]]; then
  ( MODEL_SERVICE_URL="$MODEL_SERVICE_URL" MODEL_SERVICE_TOKEN="$MODEL_SERVICE_TOKEN_EXPORT" "$VENV/bin/uvicorn" services.model_service:app --host 0.0.0.0 --port "$MODEL_SERVICE_PORT" ) | sed -e 's/^/[model] /' &
  MODEL_PID=$!
fi

# Start queue worker
RUN_QUEUE_WORKER="${RUN_QUEUE_WORKER:-true}"
WORKER_PID=""
if [[ "$RUN_QUEUE_WORKER" == "true" ]]; then
  ( "$PY" scripts/queue_worker.py ) | sed -e 's/^/[worker] /' &
  WORKER_PID=$!
fi

# Wait for any to exit
PIDS=("$API_PID")
if [[ -n "$MODEL_PID" ]]; then
  PIDS+=("$MODEL_PID")
fi
if [[ -n "$WORKER_PID" ]]; then
  PIDS+=("$WORKER_PID")
fi

DEAD_PID=""
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      DEAD_PID="$pid"
      break
    fi
  done
  if [[ -n "$DEAD_PID" ]]; then
    break
  fi
  sleep 1
done

if ps -p $API_PID >/dev/null 2>&1; then
  kill $API_PID 2>/dev/null || true
fi
if [[ -n "$MODEL_PID" ]] && ps -p $MODEL_PID >/dev/null 2>&1; then
  kill $MODEL_PID 2>/dev/null || true
fi
if [[ -n "$WORKER_PID" ]] && ps -p $WORKER_PID >/dev/null 2>&1; then
  kill $WORKER_PID 2>/dev/null || true
fi