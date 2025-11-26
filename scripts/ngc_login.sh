#!/usr/bin/env bash
# =============================================================================
# NGC Login Script for UAV-ACAR-Sionna Phase 3
# =============================================================================
# This script helps you:
#   1. Check if NGC CLI is installed
#   2. Configure NGC credentials
#   3. Login to nvcr.io Docker registry
#   4. Pull the Aerial cuBB container
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: Check NGC CLI
# =============================================================================
log_info "Checking NGC CLI..."

if ! command -v ngc &>/dev/null; then
    log_error "NGC CLI not found!"
    echo ""
    echo "Install NGC CLI:"
    echo "  wget -O ngc.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip"
    echo "  unzip ngc.zip"
    echo "  chmod +x ngc-cli/ngc"
    echo "  sudo mv ngc-cli/ngc /usr/local/bin/"
    echo ""
    exit 1
fi

log_ok "NGC CLI found: $(ngc --version 2>&1 | head -1)"

# =============================================================================
# Step 2: Configure NGC (if not already configured)
# =============================================================================
log_info "Checking NGC configuration..."

NGC_CONFIG="$HOME/.ngc/config"
if [[ ! -f "$NGC_CONFIG" ]]; then
    log_warn "NGC not configured. Running 'ngc config set'..."
    echo ""
    echo "You will need:"
    echo "  - API Key from https://ngc.nvidia.com/setup/api-key"
    echo "  - Organization (usually 'nvidia' or your org name)"
    echo ""
    ngc config set
else
    log_ok "NGC already configured"
fi

# =============================================================================
# Step 3: Docker login to nvcr.io
# =============================================================================
log_info "Logging into nvcr.io Docker registry..."

# Extract API key from NGC config
if [[ -f "$NGC_CONFIG" ]]; then
    API_KEY=$(grep -E "^apikey" "$NGC_CONFIG" | cut -d'=' -f2 | tr -d ' ' || true)
    if [[ -n "$API_KEY" ]]; then
        echo "$API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
        log_ok "Docker login to nvcr.io successful"
    else
        log_warn "Could not extract API key from NGC config"
        log_info "Manual login: docker login nvcr.io"
    fi
else
    log_warn "NGC config not found, please login manually:"
    echo "  docker login nvcr.io -u '\$oauthtoken' -p <your-api-key>"
fi

# =============================================================================
# Step 4: Pull Aerial cuBB image
# =============================================================================
CUBB_IMAGE="nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-2-cubb"

log_info "Pulling Aerial cuBB image..."
log_info "Image: $CUBB_IMAGE"
log_warn "This may take a while (image is ~10GB+)"

if docker pull "$CUBB_IMAGE"; then
    log_ok "Successfully pulled $CUBB_IMAGE"
else
    log_error "Failed to pull image"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check NGC API key permissions"
    echo "  2. Ensure you have access to nvidia/aerial org"
    echo "  3. Try: ngc registry image list nvidia/aerial/*"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
log_ok "NGC setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  cd docker/"
echo "  docker-compose up -d"
echo "  docker-compose logs -f"
echo ""
