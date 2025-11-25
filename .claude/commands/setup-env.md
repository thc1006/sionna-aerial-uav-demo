---
description: Set up the Python + Sionna environment for this project.
argument-hint: "[no args]"
allowed-tools: Bash(python3:*), Bash(pip:*), Bash(uv:*), Bash(nvidia-smi:*), Bash(git:*)
---

You are helping set up the host Python environment for the UAV Sionna–pyAerial demo.

1. Prefer `uv` if available:

   ```bash
   uv sync
   ```

2. Otherwise fall back to a virtualenv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. Verify that TensorFlow can see the RTX 4090 GPU:

   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

4. If anything fails, explain clearly what went wrong and suggest concrete fixes
   (driver/CUDA/TensorFlow compatibility, missing packages, etc.).
