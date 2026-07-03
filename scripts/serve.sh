#!/usr/bin/env bash
# Supervised in-container launcher for the LLM proxy.
#
# Why this exists (2026-07 incident): the compose command used to be
#   sh -c "uvicorn services.model_service:app ... & exec uvicorn main:app ..."
# When the model service crashed at import (boot-fatal background-model load),
# the `&` left a zombie and NOTHING respawned it — while the API on :7704 kept
# answering /health with HTTP 200, so docker showed "healthy" for 7 hours of
# 100%-failed completions. This script respawns the model service with backoff.
# (Model-load failures are now also non-fatal in Python — see
# services/model_service.py — so respawn mainly covers native llama.cpp
# crashes, which can still kill the process.)
#
# The API runs exec'd in the foreground (PID 1 semantics): docker stop's
# SIGTERM reaches uvicorn directly, and the background respawn loop dies with
# the container.
#
# Env:
#   MODEL_SERVICE_PORT  (default 7705)
#   SERVER_PORT         (default 7704)
#   RUN_MODEL_SERVICE   (default "true"; "false" skips the model service —
#                        REST/passthrough deployments, parity with run.sh's
#                        macOS behavior)
#   RUN_MIGRATIONS      (default "true"; runs `alembic upgrade head` before
#                        starting, like the sibling launchers and the
#                        compose-generated commands do — composes that keep
#                        their own migrate step can set "false")
#   MODEL_SERVICE_URL   (defaulted to http://127.0.0.1:$MODEL_SERVICE_PORT
#                        when RUN_MODEL_SERVICE=true — parity with
#                        run.sh/run-prod.sh; without it the API silently
#                        degrades to passthrough-only while /health stays 200)

set -u

MODEL_SERVICE_PORT="${MODEL_SERVICE_PORT:-7705}"
SERVER_PORT="${SERVER_PORT:-7704}"
RUN_MODEL_SERVICE="${RUN_MODEL_SERVICE:-true}"
RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"

if [[ "$RUN_MIGRATIONS" == "true" ]]; then
    echo "[serve] running alembic migrations" >&2
    if ! python -m alembic upgrade head; then
        echo "[serve] FATAL: alembic upgrade head failed" >&2
        exit 1
    fi
else
    echo "[serve] RUN_MIGRATIONS=$RUN_MIGRATIONS — skipping migrations" >&2
fi

if [[ "$RUN_MODEL_SERVICE" == "true" ]]; then
    # The API decides passthrough-vs-model-service from MODEL_SERVICE_URL. If
    # we run the model service, the API must know where it is — an unset URL
    # is the July incident signature: model service loads fine, every
    # completion 500s, and /health deliberately reports 200 "degraded"
    # (passthrough posture) so docker keeps showing healthy.
    export MODEL_SERVICE_URL="${MODEL_SERVICE_URL:-http://127.0.0.1:${MODEL_SERVICE_PORT}}"
    (
        backoff=5
        while true; do
            started_at=$(date +%s)
            python -m uvicorn services.model_service:app --host 0.0.0.0 --port "$MODEL_SERVICE_PORT"
            rc=$?
            # A run that survived >= 120s counts as healthy — reset the backoff.
            if (( $(date +%s) - started_at >= 120 )); then
                backoff=5
            fi
            echo "[serve] model service exited rc=$rc; respawning in ${backoff}s" >&2
            sleep "$backoff"
            backoff=$(( backoff * 2 ))
            if (( backoff > 60 )); then
                backoff=60
            fi
        done
    ) &
else
    echo "[serve] RUN_MODEL_SERVICE=$RUN_MODEL_SERVICE — skipping model service" >&2
fi

exec python -m uvicorn main:app --host 0.0.0.0 --port "$SERVER_PORT"
