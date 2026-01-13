#!/usr/bin/env bash
# Tail the production launchd logs for the Jarvis LLM Proxy API.

set -euo pipefail

LOG_DIR="$HOME/Library/Logs/jarvis-llm-proxy"
OUT_LOG="$LOG_DIR/out.log"
ERR_LOG="$LOG_DIR/err.log"

mkdir -p "$LOG_DIR"
touch "$OUT_LOG" "$ERR_LOG"

echo "Tailing production logs (Ctrl+C to stop)..."
echo "  stdout: $OUT_LOG"
echo "  stderr: $ERR_LOG"

tail -n 200 -F "$OUT_LOG" "$ERR_LOG"

