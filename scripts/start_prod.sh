#!/usr/bin/env bash
# Production server — 4 uvicorn workers (gunicorn process manager)
# NOTE: WebSocket broadcast is per-worker (in-process); clients connect to one worker.
# For cross-worker broadcasting, add a Redis pub/sub layer.
set -e
cd "$(dirname "$0")/.."
exec gunicorn agentic_icu.api.main:app \
  --workers "${WORKERS:-4}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "${HOST:-0.0.0.0}:${PORT:-8000}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile -
