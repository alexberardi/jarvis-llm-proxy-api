#!/usr/bin/env bash
# Stop the Jarvis LLM Proxy service gracefully, allowing vLLM to clean up.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

init_common_vars
ENV_FILE="${ENV_FILE:-.env}"
LAUNCHD_LABEL="${LAUNCHD_LABEL:-com.jarvis.llm-proxy}"

load_env "$ENV_FILE"

SERVER_PORT="${SERVER_PORT:-8000}"
MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-7705}"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-10}"

echo "Attempting to stop launchd service '$LAUNCHD_LABEL' (if loaded)..."
launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" >/dev/null 2>&1 || true

# Find processes on our ports
find_pids_on_port() {
    lsof -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

wait_for_death() {
    local pid="$1"
    local timeout="$2"
    local waited=0

    while kill -0 "$pid" 2>/dev/null && [[ $waited -lt $timeout ]]; do
        sleep 1
        waited=$((waited + 1))
    done

    ! kill -0 "$pid" 2>/dev/null
}

# Collect all PIDs
ALL_PIDS=""
for port in "$SERVER_PORT" "$MODEL_SERVICE_PORT"; do
    PIDS=$(find_pids_on_port "$port")
    if [[ -n "$PIDS" ]]; then
        echo "Found processes on port $port: $PIDS"
        ALL_PIDS="$ALL_PIDS $PIDS"
    fi
done

if [[ -z "${ALL_PIDS// /}" ]]; then
    echo "No processes are listening on ports $SERVER_PORT or $MODEL_SERVICE_PORT."
    echo "Stop routine completed."
    exit 0
fi

# Phase 1: SIGTERM for graceful shutdown
echo "Sending SIGTERM to processes:$ALL_PIDS"
for pid in $ALL_PIDS; do
    kill -TERM "$pid" 2>/dev/null || true
done

# Phase 2: Wait for graceful shutdown
echo "Waiting up to ${GRACEFUL_TIMEOUT}s for graceful shutdown..."
all_dead=true
for pid in $ALL_PIDS; do
    if ! wait_for_death "$pid" "$GRACEFUL_TIMEOUT"; then
        all_dead=false
        echo "  PID $pid still alive after timeout"
    fi
done

# Phase 3: Force kill if needed
if [[ "$all_dead" != "true" ]]; then
    echo "Force killing remaining processes..."
    for pid in $ALL_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
fi

# Phase 4: Kill orphaned vLLM processes
VLLM_PIDS=$(pgrep -f "VLLM::EngineCore" 2>/dev/null || true)
if [[ -n "$VLLM_PIDS" ]]; then
    echo "Killing orphaned vLLM processes: $VLLM_PIDS"
    for pid in $VLLM_PIDS; do
        kill -9 "$pid" 2>/dev/null || true
    done
fi

# Verify ports are free
sleep 1
for port in "$SERVER_PORT" "$MODEL_SERVICE_PORT"; do
    if lsof -t -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "WARNING: Port $port still has processes listening!"
    else
        echo "Port $port is now free."
    fi
done

echo "Stop routine completed."
