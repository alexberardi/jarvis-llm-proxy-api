#!/usr/bin/env bash
# Stop the Jarvis LLM Proxy service (launchd) and any leftover server process on the configured port.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

init_common_vars
ENV_FILE="${ENV_FILE:-prod.env}"
LAUNCHD_LABEL="${LAUNCHD_LABEL:-com.jarvis.llm-proxy}"

load_env "$ENV_FILE"

SERVER_PORT="${SERVER_PORT:-8000}"

echo "Attempting to stop launchd service '$LAUNCHD_LABEL' (if loaded)..."
launchctl bootout "gui/$(id -u)/$LAUNCHD_LABEL" >/dev/null 2>&1 || true

echo "Looking for processes listening on port $SERVER_PORT..."
PIDS=$(lsof -t -iTCP:"$SERVER_PORT" -sTCP:LISTEN 2>/dev/null || true)

if [[ -n "${PIDS:-}" ]]; then
    echo "Found processes on port $SERVER_PORT: $PIDS"
    kill $PIDS || true
    sleep 1
    if lsof -t -iTCP:"$SERVER_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Processes still listening on $SERVER_PORT; sending SIGKILL..."
        kill -9 $PIDS || true
    fi
    echo "Port $SERVER_PORT is now free."
else
    echo "No processes are listening on port $SERVER_PORT."
fi

echo "Stop routine completed."











