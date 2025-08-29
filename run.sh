#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
ENV_FILE="${ENV_FILE:-.env}"

# --- pick a base python to create venv (once) ---
BASE_PY=""
if command -v pyenv >/dev/null 2>&1; then
  # install if missing, but non-interactively (-s = skip if already installed)
  PYENV_NONINTERACTIVE=1 pyenv install -s "$PYTHON_VERSION"
  BASE_PY="$(pyenv prefix "$PYTHON_VERSION")/bin/python"
fi
if [[ -z "${BASE_PY}" ]]; then
  # fallback to system/Homebrew python3
  BASE_PY="$(command -v python3)"
fi
if [[ -z "${BASE_PY}" ]]; then
  echo "❌ No usable python found (neither pyenv nor python3)."; exit 1
fi

# --- create venv if missing ---
if [[ ! -x "$PY" ]]; then
  echo "📦 Creating virtual environment with $BASE_PY"
  "$BASE_PY" -m venv "$VENV"
fi

# --- install deps in venv ---
"$PIP" install -U pip setuptools wheel
[[ -f requirements.txt ]] && "$PIP" install -r requirements.txt

# --- env file ---
[[ -f "$ENV_FILE" ]] || { echo "❌ Missing $ENV_FILE"; exit 1; }
set -a; # export all sourced vars
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# --- diagnostics: prove interpreter & llama build ---
"$PY" - <<'PYINFO'
import sys
print("RUN PY:", sys.executable)
try:
    import llama_cpp
    print("RUN llama_cpp:", llama_cpp.__version__, llama_cpp.__file__)
except Exception as e:
    print("llama_cpp import error:", e)
PYINFO

# quiet by default; set LLAMA_LOG_LEVEL=debug to troubleshoot
export LLAMA_LOG_LEVEL="${LLAMA_LOG_LEVEL:-info}"

# --- run server using venv's python (never the shim) ---
exec "$PY" -m uvicorn main:app --host "${SERVER_HOST:-0.0.0.0}" --port "${SERVER_PORT:-8000}" --reload