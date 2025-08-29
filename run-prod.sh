#!/bin/bash

set -e

PYTHON_VERSION="3.11.9"
ENV_FILE="prod.env"

echo "🐍 Ensuring Python $PYTHON_VERSION with pyenv"
if ! pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
  echo "⬇️  Installing Python $PYTHON_VERSION via pyenv..."
  pyenv install "$PYTHON_VERSION"
fi

echo "$PYTHON_VERSION" > .python-version
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment"
  pyenv local "$PYTHON_VERSION"
  python -m venv venv
fi

echo "📦 Installing requirements"
source venv/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt

echo "🧪 Checking for $ENV_FILE"
if [ ! -f "$ENV_FILE" ]; then
  echo "❌ Missing $ENV_FILE file. Aborting."
  exit 1
fi

echo "🚀 Running app with $ENV_FILE"
export $(grep -v '^#' "$ENV_FILE" | xargs)
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
SERVER_PORT=${SERVER_PORT:-"8000"}
uvicorn main:app --host $SERVER_HOST --port $SERVER_PORT 
