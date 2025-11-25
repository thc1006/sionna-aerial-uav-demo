#!/usr/bin/env bash
# Simple wrapper that logs every command invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

echo "[$(date --iso-8601=seconds)] $PWD $*" >> "${LOG_DIR}/commands.log"

"$@"
