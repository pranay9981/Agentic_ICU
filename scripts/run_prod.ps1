# Agentic-ICU Production Runner (Windows)
# This script starts the server with 4 concurrent workers and auto-warmup.

$env:PYTHONPATH = "src"
$env:AGENTIC_ICU_HOST = "127.0.0.1"
$env:AGENTIC_ICU_PORT = "8000"

Write-Host "Starting Agentic-ICU with 4 Workers..." -ForegroundColor Cyan
Write-Host "API: http://$env:AGENTIC_ICU_HOST:$env:AGENTIC_ICU_PORT" -ForegroundColor Gray

# Run using uvicorn with workers. 
# Note: --reload is omitted in production for stability and performance.
venv\Scripts\uvicorn agentic_icu.api.main:app `
    --host $env:AGENTIC_ICU_HOST `
    --port $env:AGENTIC_ICU_PORT `
    --workers 4 `
    --log-level info `
    --ws websockets `
    --proxy-headers
