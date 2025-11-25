#!/usr/bin/env bash
# Run the Phase 0 three-distance demo and log the command.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
cd "${ROOT_DIR}"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

"${SCRIPT_DIR}/ccmd.sh" python -m uav_acar_demo.cli phase0 --output-dir data/phase0
