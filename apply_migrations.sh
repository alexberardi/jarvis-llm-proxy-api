#!/usr/bin/env bash
set -euo pipefail

# Apply latest Alembic migrations using Python helper
# Activate venv if present, otherwise use python3
if [[ -d "venv" ]]; then
    source venv/bin/activate
fi

python3 scripts/apply_migrations.py
