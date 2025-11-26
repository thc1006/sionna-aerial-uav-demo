#!/usr/bin/env python3
"""Example client script for testing the UAV-ACAR-Sionna REST API."""

import argparse
import json
import sys
from typing import Any

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install it with:")
    print("  pip install requests")
    sys.exit(1)


def test_health(base_url: str) -> None:
    """Test the health check endpoint."""
    print("\n=== Testing Health Check ===")
    url = f"{base_url}/health"
    print(f"GET {url}")

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check failed: {e}")
        sys.exit(1)


def test_emulation(base_url: str, scenario_id: str, time_point: float, distance_m: float) -> dict[str, Any]:
    """Test the emulation endpoint."""
    print("\n=== Testing Emulation ===")
    url = f"{base_url}/emulate"
    payload = {
        "scenario_id": scenario_id,
        "time_point": time_point,
        "distance_m": distance_m,
    }

    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, timeout=300)  # 5 min timeout for emulation
        response.raise_for_status()
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {json.dumps(data, indent=2)}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"✗ Emulation failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"✗ Error details: {e.response.text}")
        sys.exit(1)


def test_results(base_url: str, limit: int = 10) -> None:
    """Test the results endpoint."""
    print("\n=== Testing Results Retrieval ===")
    url = f"{base_url}/results?limit={limit}"
    print(f"GET {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Total records: {data['total_records']}")
        print(f"✓ Retrieved: {len(data['records'])} records")

        if data["records"]:
            print("\nMost recent record:")
            print(json.dumps(data["records"][-1], indent=2))
    except requests.exceptions.RequestException as e:
        print(f"✗ Results retrieval failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"✗ Error details: {e.response.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test UAV-ACAR-Sionna REST API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--scenario-id",
        default="base",
        help="Scenario ID for emulation test (default: base)",
    )
    parser.add_argument(
        "--time-point",
        type=float,
        default=0.0,
        help="Time point for emulation test (default: 0.0)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=500.0,
        help="Distance in meters for emulation test (default: 500.0)",
    )
    parser.add_argument(
        "--skip-emulation",
        action="store_true",
        help="Skip the emulation test (only test health and results)",
    )

    args = parser.parse_args()

    print(f"Testing API at: {args.base_url}")

    # Test health check
    test_health(args.base_url)

    # Test emulation (unless skipped)
    if not args.skip_emulation:
        test_emulation(args.base_url, args.scenario_id, args.time_point, args.distance)
    else:
        print("\n⊘ Skipping emulation test (--skip-emulation flag set)")

    # Test results retrieval
    test_results(args.base_url)

    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()
