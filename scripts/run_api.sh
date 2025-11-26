#!/usr/bin/env bash
#
# run_api.sh – Start the FastAPI server for UAV-ACAR-Sionna emulation API
#
# Usage:
#   ./scripts/run_api.sh [--reload]
#
# Options:
#   --reload    Enable auto-reload for development (watches for file changes)
#
# The server will start on http://0.0.0.0:8000 by default.
# API docs available at:
#   - Swagger UI: http://localhost:8000/docs
#   - ReDoc: http://localhost:8000/redoc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
LOGS_DIR="${REPO_ROOT}/logs/api"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Log file with timestamp
LOG_FILE="${LOGS_DIR}/api_$(date +%Y%m%d_%H%M%S).log"

# Default options
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD_FLAG=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reload)
            RELOAD_FLAG="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--reload]"
            exit 1
            ;;
    esac
done

# Change to repo root
cd "${REPO_ROOT}"

# Log the command
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting FastAPI server on ${HOST}:${PORT}" | tee -a "${LOG_FILE}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Logs: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Start the server with logging
if [[ -n "${RELOAD_FLAG}" ]]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running in development mode with auto-reload" | tee -a "${LOG_FILE}"
    uvicorn uav_acar_sionna.api.main:app \
        --host "${HOST}" \
        --port "${PORT}" \
        --reload \
        --log-level info 2>&1 | tee -a "${LOG_FILE}"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running in production mode" | tee -a "${LOG_FILE}"
    uvicorn uav_acar_sionna.api.main:app \
        --host "${HOST}" \
        --port "${PORT}" \
        --log-level info 2>&1 | tee -a "${LOG_FILE}"
fi
