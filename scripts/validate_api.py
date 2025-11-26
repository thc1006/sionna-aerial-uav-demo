#!/usr/bin/env python3
"""Quick validation script to check API structure without running the server."""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

try:
    from uav_acar_sionna.api import app
    from uav_acar_sionna.api.schemas import (
        EmulationRequest,
        EmulationResponse,
        HealthResponse,
        SummaryResponse,
    )

    print("✓ API module imports successfully")
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    print("\n✓ Available routes:")
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            methods = ", ".join(route.methods)
            print(f"  {methods:10} {route.path}")

    print("\n✓ Pydantic schemas validated:")
    print("  - EmulationRequest")
    print("  - EmulationResponse")
    print("  - HealthResponse")
    print("  - SummaryResponse")

    # Test schema instantiation
    req = EmulationRequest(scenario_id="test", time_point=0.0, distance_m=500.0)
    print(f"\n✓ Sample request: {req.model_dump_json()}")

    print("\n✅ All validations passed!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nPlease install the package first:")
    print("  pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"❌ Validation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
