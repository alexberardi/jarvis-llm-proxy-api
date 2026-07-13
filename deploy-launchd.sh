#!/usr/bin/env bash
# (Re)deploy the launchd agent that runs the Jarvis LLM Proxy API on login.
#
# On macOS this is the recommended way to run llm-proxy — native execution lets
# MLX / Metal-llama.cpp access the Apple GPU, which Docker on Mac can't reach.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

init_common_vars

LABEL="${LAUNCHD_LABEL:-com.jarvis.llm-proxy}"
PORT="${LLM_PROXY_PORT:-7704}"
ENV_FILE_PATH="${ENV_FILE_PATH:-prod.env}"

PLIST_TEMPLATE="$ROOT/scripts/launchd/$LABEL.plist"
AGENTS_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$AGENTS_DIR/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/jarvis-llm-proxy"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "❌ deploy-launchd.sh only supports macOS (detected $(uname -s))"
    exit 1
fi

if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "❌ launchd template not found at $PLIST_TEMPLATE"
    exit 1
fi

mkdir -p "$AGENTS_DIR" "$LOG_DIR"

sed -e "s#__ROOT__#$ROOT#g" \
    -e "s#__USER__#$USER#g" \
    -e "s#__PORT__#$PORT#g" \
    -e "s#__ENV_FILE__#$ENV_FILE_PATH#g" \
    "$PLIST_TEMPLATE" > "$TARGET_PLIST"

echo "📄 Installed launchd plist to $TARGET_PLIST"

echo "🔄 Reloading launchd service $LABEL..."
LAUNCH_UID="$(id -u)"

# `launchctl bootout` returns BEFORE launchd has finished tearing the job down.
# Bootstrapping immediately races that teardown: the bootstrap fails, and since
# the old job is already gone the service is left UNLOADED — i.e. redeploying
# over a RUNNING agent (what a platform update does) silently kills it. A fresh
# install never hit this: there was nothing to boot out.
# Wait for the label to actually disappear, then bootstrap, with retries.
launchctl bootout "gui/$LAUNCH_UID/$LABEL" >/dev/null 2>&1 || true
for _ in $(seq 1 50); do
    launchctl print "gui/$LAUNCH_UID/$LABEL" >/dev/null 2>&1 || break
    sleep 0.2
done

bootstrapped=0
for _ in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$LAUNCH_UID" "$TARGET_PLIST" 2>/dev/null; then
        bootstrapped=1
        break
    fi
    sleep 1
done
if [ "$bootstrapped" -ne 1 ]; then
    echo "❌ launchctl bootstrap failed for $LABEL — service is NOT running" >&2
    exit 1
fi

launchctl enable "gui/$LAUNCH_UID/$LABEL"
launchctl kickstart -k "gui/$LAUNCH_UID/$LABEL"

echo "✅ LaunchAgent ready. Check status with: launchctl print gui/$(id -u)/$LABEL"
echo "📜 Logs: $LOG_DIR/{out,err}.log"
