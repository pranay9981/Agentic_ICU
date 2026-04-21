#!/usr/bin/env bash
# Development server — single worker, auto-reload
set -e
cd "$(dirname "$0")/.."
python -m uvicorn agentic_icu.api.main:app --host 127.0.0.1 --port 8000 --reload
