# UAV-ACAR-Sionna

**Sionna → pyAerial → ACAR pipeline for UAV link simulations**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen.svg)](#testing)

## Overview

This repository implements a complete pipeline for UAV (Unmanned Aerial Vehicle) communication link simulation and emulation:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sionna    │ ──▶ │   Sionna     │ ──▶ │   MATLAB    │ ──▶ │    cuBB     │
│  (Channel)  │     │   Adapter    │     │  (TestVec)  │     │  (Emulate)  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                           │                                        │
                           ▼                                        ▼
                    ┌──────────────┐                         ┌─────────────┐
                    │  ML Models   │                         │  Throughput │
                    │  (PyTorch)   │                         │    BLER     │
                    └──────────────┘                         └─────────────┘
```

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | Three-distance simulation (near/mid/far) |
| Phase 1 | ✅ Complete | Sionna backend with Rayleigh fading |
| Phase 2 | ✅ Complete | Single-host skeleton (ML + API + Dataset) |
| Phase 3 | 🔄 Ready | Docker/NGC integration (requires permissions) |

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/thc1006/sionna-aerial-uav-demo.git
cd sionna-aerial-uav-demo

# Full setup (includes TensorFlow/Sionna + GPU check + model training)
./scripts/setup.sh

# Or quick setup (skip GPU dependencies, faster)
./scripts/setup.sh --quick

# Or Phase 3 setup (includes Docker/NGC checks)
./scripts/setup.sh --phase3
```

### 2. Activate Environment

```bash
source .venv/bin/activate
```

### 3. Run Tests

```bash
pytest tests/
# Expected: 14 passed
```

### 4. Train ML Models

```bash
# Train FC model (tabular features → throughput)
uav-train --model fc --epochs 100 --device cuda

# Or with CPU
uav-train --model fc --epochs 100 --device cpu
```

### 5. Start REST API

```bash
uav-api
# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/emulate` | POST | Run single emulation |
| `/results` | GET | List simulation results |

Example:
```bash
curl -X POST http://localhost:8000/emulate \
  -H "Content-Type: application/json" \
  -d '{"scenario_id": "test", "distance_m": 500}'
```

## Project Structure

```
sionna-aerial-uav-demo/
├── src/uav_acar_sionna/
│   ├── api/              # FastAPI REST endpoints
│   ├── ml/               # PyTorch models (FCRegressor, CNN1D)
│   ├── orchestrator/     # Sionna/MATLAB/cuBB adapters
│   ├── config.py         # Configuration dataclasses
│   └── sionna_backend.py # Sionna channel simulation
├── tests/                # pytest test suite (14 tests)
├── data/
│   ├── phase1/           # Phase 1 results
│   └── phase2_interference/  # Phase 2 datasets (summary.csv, npz, h5)
├── docker/               # Phase 3 Docker setup
│   ├── docker-compose.yml
│   └── Dockerfile.orchestrator
├── scripts/
│   ├── setup.sh          # One-click setup
│   └── ngc_login.sh      # NGC authentication
└── pyproject.toml        # Dependencies & config
```

## ML Models

| Model | Input | Target | Performance |
|-------|-------|--------|-------------|
| FCRegressor | distance, SNR, SINR | throughput_mbps | **R²=0.89** |
| CNN1D | channel data (4×4096) | throughput_mbps | Needs real channel data |

## Phase 3: Docker Deployment

### Prerequisites

1. Docker with GPU support (NVIDIA Container Toolkit)
2. NGC CLI and API key
3. Access to `nvidia/aerial` NGC organization

### Setup

```bash
# 1. Setup with Phase 3 dependencies
./scripts/setup.sh --phase3

# 2. Login to NGC and pull containers
./scripts/ngc_login.sh

# 3. Start services
cd docker/
docker-compose up -d
docker-compose logs -f
```

## Development

```bash
# Install all dependencies
pip install -e ".[all]"

# Run tests with verbose output
pytest tests/ -v

# Type checking (optional)
mypy src/

# Linting (optional)
ruff check src/
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UAV_ACAR_HOST_ROLE` | `single` | Host role (single/orchestrator/cubb/ru_emu) |
| `CUDA_VISIBLE_DEVICES` | auto | GPU selection |
| `XLA_FLAGS` | auto | TensorFlow XLA configuration |

## Troubleshooting

### TensorFlow XLA Error (`libdevice not found`)
```bash
pip install nvidia-cuda-nvcc-cu12
```
The `tests/conftest.py` automatically configures XLA_FLAGS.

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
newgrp docker  # or logout/login
```

### NGC Login Failed
```bash
ngc config set  # Re-configure credentials
docker login nvcr.io -u '$oauthtoken' -p <your-api-key>
```

## Using with Claude Code

1. Clone this repo on your target machine
2. Copy `CLAUDE.md.template` to `CLAUDE.md` and customize
3. Run `./scripts/setup.sh` to initialize
4. Ask Claude Code to continue Phase 3 integration

## References

- [Sionna Documentation](https://nvlabs.github.io/sionna/)
- [NVIDIA Aerial SDK](https://developer.nvidia.com/aerial-sdk)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
