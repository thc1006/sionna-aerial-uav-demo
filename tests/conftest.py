"""Pytest configuration for test suite."""

import os
from pathlib import Path

# Find the nvidia cuda_nvcc package in the venv for libdevice.10.bc
# This is needed for TensorFlow XLA JIT compilation on GPU
_venv_site_packages = Path(__file__).parent.parent / ".venv/lib/python3.12/site-packages"
_cuda_nvcc_path = _venv_site_packages / "nvidia/cuda_nvcc"

if _cuda_nvcc_path.exists():
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_cuda_nvcc_path}"

# Reduce TensorFlow logging noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def pytest_configure(config):
    """Configure pytest before tests run."""
    pass
