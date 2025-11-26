#!/usr/bin/env bash
# =============================================================================
# UAV-ACAR-Sionna Setup Script
# =============================================================================
# Usage:
#   ./scripts/setup.sh          # Full setup (Python + all dependencies)
#   ./scripts/setup.sh --quick  # Quick setup (skip GPU/Sionna deps)
#   ./scripts/setup.sh --phase3 # Setup for Phase 3 (includes Docker/NGC)
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get script directory (works even when called from different locations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# =============================================================================
# Parse arguments
# =============================================================================
QUICK_MODE=false
PHASE3_MODE=false

for arg in "$@"; do
    case $arg in
        --quick)  QUICK_MODE=true ;;
        --phase3) PHASE3_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--quick|--phase3]"
            echo "  --quick   Skip GPU/Sionna dependencies (faster)"
            echo "  --phase3  Include Docker/NGC setup for Phase 3"
            exit 0
            ;;
    esac
done

# =============================================================================
# Step 1: Check Python version
# =============================================================================
log_info "Checking Python version..."

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -lt 13 ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    log_error "Python 3.10-3.12 required but not found!"
    exit 1
fi

log_ok "Found $PYTHON_CMD (version $version)"

# =============================================================================
# Step 2: Create virtual environment
# =============================================================================
log_info "Setting up virtual environment..."

if [[ ! -d ".venv" ]]; then
    "$PYTHON_CMD" -m venv .venv
    log_ok "Created .venv"
else
    log_ok ".venv already exists"
fi

# Activate venv
source .venv/bin/activate
log_ok "Activated virtual environment"

# =============================================================================
# Step 3: Upgrade pip and install base tools
# =============================================================================
log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools -q
log_ok "pip upgraded"

# =============================================================================
# Step 4: Install dependencies
# =============================================================================
log_info "Installing dependencies..."

if $QUICK_MODE; then
    # Quick mode: core + dev + train only
    pip install -e ".[dev,train]" -q
    log_ok "Installed core + dev + train dependencies (quick mode)"
else
    # Full mode: everything including Sionna/TensorFlow
    pip install -e ".[all]" -q
    log_ok "Installed all dependencies"
fi

# =============================================================================
# Step 5: Verify GPU (if not quick mode)
# =============================================================================
if ! $QUICK_MODE; then
    log_info "Checking GPU availability..."

    # Check PyTorch GPU
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        log_ok "PyTorch GPU: $GPU_NAME"
    else
        log_warn "PyTorch: No GPU detected (will use CPU)"
    fi

    # Check TensorFlow GPU
    if python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); assert len(gpus)>0" 2>/dev/null; then
        log_ok "TensorFlow GPU: Available"
    else
        log_warn "TensorFlow: No GPU detected (will use CPU)"
    fi
fi

# =============================================================================
# Step 6: Run tests to verify installation
# =============================================================================
log_info "Running tests to verify installation..."

if pytest tests/ -q --tb=no 2>/dev/null; then
    PASSED=$(pytest tests/ -q --tb=no 2>&1 | tail -1)
    log_ok "Tests: $PASSED"
else
    log_warn "Some tests failed - check with: pytest tests/ -v"
fi

# =============================================================================
# Step 7: Train models if checkpoints missing
# =============================================================================
if [[ ! -f "checkpoints/best_fc_throughput_mbps.pt" ]]; then
    log_info "Training FC model (checkpoints not found)..."
    python -m uav_acar_sionna.ml.train --model fc --epochs 50 --device cuda 2>/dev/null || \
    python -m uav_acar_sionna.ml.train --model fc --epochs 50 --device cpu
    log_ok "FC model trained"
else
    log_ok "FC model checkpoint exists"
fi

# =============================================================================
# Step 8: Phase 3 setup (Docker/NGC)
# =============================================================================
if $PHASE3_MODE; then
    log_info "Phase 3 setup: Checking Docker..."

    if command -v docker &>/dev/null; then
        if docker info &>/dev/null 2>&1; then
            log_ok "Docker: Available and running"
        else
            log_warn "Docker: Installed but permission denied"
            log_warn "  → Run: sudo usermod -aG docker \$USER && newgrp docker"
        fi
    else
        log_warn "Docker: Not installed"
        log_warn "  → Install: https://docs.docker.com/engine/install/"
    fi

    log_info "Phase 3 setup: Checking NGC CLI..."
    if command -v ngc &>/dev/null; then
        log_ok "NGC CLI: Available"
    else
        log_warn "NGC CLI: Not installed"
        log_warn "  → Install: wget -O ngc.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip"
        log_warn "  → Then: unzip ngc.zip && sudo mv ngc-cli/ngc /usr/local/bin/"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
log_ok "Setup complete!"
echo "=============================================="
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  pytest tests/                    # Run tests"
echo "  uav-train --model fc --epochs 50 # Train model"
echo "  uav-api                          # Start REST API"
echo ""
if $PHASE3_MODE; then
    echo "Phase 3 (after Docker/NGC ready):"
    echo "  ./scripts/ngc_login.sh           # Login to NGC"
    echo "  docker-compose -f docker/docker-compose.yml up"
    echo ""
fi
