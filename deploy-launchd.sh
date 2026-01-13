#!/usr/bin/env bash
# (Re)deploy the launchd agent that runs the Jarvis LLM Proxy API on login.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

init_common_vars

LABEL="${LAUNCHD_LABEL:-com.jarvis.llm-proxy}"
PLIST_TEMPLATE="$ROOT/scripts/launchd/$LABEL.plist"
AGENTS_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$AGENTS_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/jarvis-llm-proxy"

if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "❌ launchd template not found at $PLIST_TEMPLATE"
    exit 1
fi

mkdir -p "$AGENTS_DIR" "$LOG_DIR"

# Materialize plist with absolute paths
sed -e "s#__ROOT__#$ROOT#g" \
    -e "s#__USER__#$USER#g" \
    "$PLIST_TEMPLATE" > "$TARGET_PLIST"

echo "📄 Installed launchd plist to $TARGET_PLIST"

echo "🔄 Reloading launchd service $LABEL..."
launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$TARGET_PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "✅ LaunchAgent ready. Check status with: launchctl print gui/$(id -u)/$LABEL"






