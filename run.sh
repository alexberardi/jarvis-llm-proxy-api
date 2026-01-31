#!/usr/bin/env bash
# Development run script

# Load common functions
source "$(dirname "$0")/scripts/common.sh"

# Initialize common variables
init_common_vars
ENV_FILE="${ENV_FILE:-.env}"

# Development-specific configuration
ENABLE_RELOAD="false"

echo -e "${GREEN}🚀 Jarvis LLM Proxy API - Development Mode${NC}"
echo -e "${BLUE}📁 Root directory: $ROOT${NC}"

# Setup and configuration
check_setup
load_env "$ENV_FILE"
# Configure runtime env after loading .env
configure_vllm_env
# Disable reload unconditionally (overrides any .env ENABLE_RELOAD)
ENABLE_RELOAD="false"

# Create virtual environment (development strategy)
create_venv_dev
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

# Start development server + queue worker (aggregated logs)
RUN_QUEUE_WORKER="${RUN_QUEUE_WORKER:-true}"

echo -e "${BLUE}📜 Aggregated logs: [api] for server, [worker] for queue worker${NC}"

# Ensure fork-safety env for macOS workers (Objective-C libs)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}

# Start model service if configured
MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-8008}"
RUN_MODEL_SERVICE="${RUN_MODEL_SERVICE:-true}"
MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"
MODEL_SERVICE_TOKEN_EXPORT="${MODEL_SERVICE_TOKEN:-${LLM_PROXY_INTERNAL_TOKEN:-}}"

# Start API server in background
# NOTE: Do NOT pipe output through sed - vLLM's logging causes BrokenPipeError (SIGPIPE)
# when the pipe breaks during model initialization
echo -e "${BLUE}[api] Starting API server${NC}"
start_server "false" &
API_PID=$!
echo -e "${BLUE}🔢 API_PID=${API_PID}${NC}"

# Start model service in background
# NOTE: Same SIGPIPE issue applies here - no sed piping
MODEL_PID=""
if [[ "$RUN_MODEL_SERVICE" == "true" ]]; then
  echo -e "${BLUE}[model] Starting model service on port $MODEL_SERVICE_PORT${NC}"
  MODEL_SERVICE_URL="$MODEL_SERVICE_URL" MODEL_SERVICE_TOKEN="$MODEL_SERVICE_TOKEN_EXPORT" "$VENV/bin/uvicorn" services.model_service:app --host 0.0.0.0 --port "$MODEL_SERVICE_PORT" &
  MODEL_PID=$!
  echo -e "${BLUE}🔢 MODEL_PID=${MODEL_PID}${NC}"
fi

# Optionally start queue worker
# NOTE: Do NOT pipe worker output through sed - RQ forks work horses that inherit
# the pipe, and broken pipes cause SIGPIPE (signal 13) crashes. Write directly instead.
WORKER_PID=""
if [[ "$RUN_QUEUE_WORKER" == "true" ]]; then
    echo -e "${BLUE}[worker] Starting queue worker (output prefixed inline)${NC}"
    LLM_PROXY_PROCESS_ROLE=worker "$PY" scripts/queue_worker.py &
    WORKER_PID=$!
    echo -e "${BLUE}🔢 WORKER_PID=${WORKER_PID}${NC}"
fi

# Wait for either process to exit (portable wait-any loop)
PIDS=("$API_PID")
if [[ -n "$WORKER_PID" ]]; then
  PIDS+=("$WORKER_PID")
fi
if [[ -n "$MODEL_PID" ]]; then
  PIDS+=("$MODEL_PID")
fi

DEAD_PID=""
DEAD_STATUS=""
while true; do
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      DEAD_PID="$pid"
      # Try to collect an exit code for the dead process for easier debugging
      if wait "$pid" 2>/dev/null; then
        DEAD_STATUS=$?
      else
        DEAD_STATUS="unknown"
      fi
      break
    fi
  done
  if [[ -n "$DEAD_PID" ]]; then
    ROLE="unknown"
    if [[ "$DEAD_PID" == "$API_PID" ]]; then ROLE="api"; fi
    if [[ "$DEAD_PID" == "$MODEL_PID" ]]; then ROLE="model"; fi
    if [[ "$DEAD_PID" == "$WORKER_PID" ]]; then ROLE="worker"; fi
    echo -e "${YELLOW}⚠️  Process $DEAD_PID exited (role=${ROLE}, status=${DEAD_STATUS}). Stopping others...${NC}"
    break
  fi
  sleep 1
done

# If one exits, stop the other
if ps -p $API_PID >/dev/null 2>&1; then
  kill $API_PID 2>/dev/null || true
fi
if [[ -n "$WORKER_PID" ]] && ps -p $WORKER_PID >/dev/null 2>&1; then
  kill $WORKER_PID 2>/dev/null || true
fi
if [[ -n "$MODEL_PID" ]] && ps -p $MODEL_PID >/dev/null 2>&1; then
  kill $MODEL_PID 2>/dev/null || true
fi