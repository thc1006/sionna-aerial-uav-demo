#!/usr/bin/env bash
# Create a virtual environment and install this project in editable mode.
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
