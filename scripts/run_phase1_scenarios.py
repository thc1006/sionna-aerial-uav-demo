#!/usr/bin/env python3
"""Run Phase 1 UAV link scenarios and export results.

This script serves as the Phase 0-1 milestone demonstration:
1. Runs near/mid/far UAV distance scenarios
2. Computes SNR, BER, and throughput using Sionna backend
3. Exports results to CSV and generates visualization
4. Prints summary table for quick inspection

Usage:
    python scripts/run_phase1_scenarios.py
    python scripts/run_phase1_scenarios.py --output-dir data/phase1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from uav_acar_sionna import (
    UavScenarioConfig,
    LinkResult,
    estimate_link_with_sionna_backend,
)


# =============================================================================
# Scenario Definitions
# =============================================================================

def create_phase1_scenarios() -> Dict[str, UavScenarioConfig]:
    """Create the canonical Phase 1 UAV scenarios.

    Returns three scenarios at different distances:
    - near: 200m (short range, high SNR expected)
    - mid: 500m (medium range)
    - far: 1000m (long range, low SNR expected)
    """
    scenarios = {
        "near": UavScenarioConfig(
            distance_m=200.0,
            name="near",
            tx_height_m=25.0,
            ue_height_m=120.0,
            carrier_freq_ghz=3.5,
            bandwidth_hz=20e6,
            tx_power_dbm=30.0,
            noise_figure_db=5.0,
            seed=42,
        ),
        "mid": UavScenarioConfig(
            distance_m=500.0,
            name="mid",
            tx_height_m=25.0,
            ue_height_m=120.0,
            carrier_freq_ghz=3.5,
            bandwidth_hz=20e6,
            tx_power_dbm=30.0,
            noise_figure_db=5.0,
            seed=43,
        ),
        "far": UavScenarioConfig(
            distance_m=1000.0,
            name="far",
            tx_height_m=25.0,
            ue_height_m=120.0,
            carrier_freq_ghz=3.5,
            bandwidth_hz=20e6,
            tx_power_dbm=30.0,
            noise_figure_db=5.0,
            seed=44,
        ),
    }
    return scenarios


# =============================================================================
# Scenario Runner
# =============================================================================

def run_scenarios(scenarios: Dict[str, UavScenarioConfig]) -> List[LinkResult]:
    """Run all scenarios and collect results.

    Parameters
    ----------
    scenarios : Dict[str, UavScenarioConfig]
        Dictionary of scenario name -> config.

    Returns
    -------
    List[LinkResult]
        Results for each scenario.
    """
    results = []

    print("\n" + "=" * 70)
    print("  UAV-ACAR-Sionna Phase 1 Scenario Evaluation")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    for name, config in scenarios.items():
        print(f"\n[{name.upper()}] Running scenario...")
        print(f"  Distance: {config.distance_m:.0f} m")
        print(f"  Carrier: {config.carrier_freq_ghz} GHz")
        print(f"  Bandwidth: {config.bandwidth_hz / 1e6:.0f} MHz")
        print(f"  TX Power: {config.tx_power_dbm} dBm")

        result = estimate_link_with_sionna_backend(config)
        results.append(result)

        print(f"  → SNR: {result.snr_db:.2f} dB")
        print(f"  → Throughput: {result.throughput_mbps:.2f} Mbps")
        print(f"  → Backend: {result.backend}")

    return results


# =============================================================================
# Export Functions
# =============================================================================

def export_csv(results: List[LinkResult], output_path: Path) -> None:
    """Export results to CSV format.

    Parameters
    ----------
    results : List[LinkResult]
        Simulation results.
    output_path : Path
        Output CSV file path.
    """
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "scenario_name",
            "distance_m",
            "carrier_freq_ghz",
            "bandwidth_mhz",
            "tx_power_dbm",
            "noise_figure_db",
            "snr_db",
            "rsrp_dbm",
            "throughput_mbps",
            "backend",
            "notes",
        ])

        # Data rows
        for result in results:
            writer.writerow([
                result.scenario.name,
                result.scenario.distance_m,
                result.scenario.carrier_freq_ghz,
                result.scenario.bandwidth_hz / 1e6,
                result.scenario.tx_power_dbm,
                result.scenario.noise_figure_db,
                f"{result.snr_db:.4f}",
                result.rsrp_dbm if result.rsrp_dbm else "N/A",
                f"{result.throughput_mbps:.4f}",
                result.backend,
                result.notes or "",
            ])

    print(f"\n✓ CSV exported: {output_path}")


def export_json(results: List[LinkResult], output_path: Path) -> None:
    """Export results to JSON format.

    Parameters
    ----------
    results : List[LinkResult]
        Simulation results.
    output_path : Path
        Output JSON file path.
    """
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "phase": "Phase 0-1 Milestone",
            "description": "UAV link simulation results using Sionna minimal backend",
        },
        "scenarios": []
    }

    for result in results:
        scenario_data = {
            "name": result.scenario.name,
            "config": {
                "distance_m": result.scenario.distance_m,
                "carrier_freq_ghz": result.scenario.carrier_freq_ghz,
                "bandwidth_hz": result.scenario.bandwidth_hz,
                "tx_power_dbm": result.scenario.tx_power_dbm,
                "noise_figure_db": result.scenario.noise_figure_db,
                "tx_height_m": result.scenario.tx_height_m,
                "ue_height_m": result.scenario.ue_height_m,
                "seed": result.scenario.seed,
            },
            "results": {
                "snr_db": result.snr_db,
                "rsrp_dbm": result.rsrp_dbm,
                "throughput_mbps": result.throughput_mbps,
            },
            "backend": result.backend,
            "notes": result.notes,
        }
        data["scenarios"].append(scenario_data)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ JSON exported: {output_path}")


def generate_plot(results: List[LinkResult], output_path: Path) -> None:
    """Generate visualization plot of results.

    Parameters
    ----------
    results : List[LinkResult]
        Simulation results.
    output_path : Path
        Output PNG file path.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not installed, skipping plot generation")
        return

    # Extract data
    names = [r.scenario.name for r in results]
    distances = [r.scenario.distance_m for r in results]
    snrs = [r.snr_db for r in results]
    throughputs = [r.throughput_mbps for r in results]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("UAV-ACAR-Sionna Phase 1 Results", fontsize=14, fontweight="bold")

    # Colors
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # green, orange, red

    # Plot 1: SNR vs Distance
    ax1 = axes[0]
    ax1.bar(names, snrs, color=colors, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("SNR (dB)", fontsize=11)
    ax1.set_xlabel("Scenario", fontsize=11)
    ax1.set_title("Signal-to-Noise Ratio", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    for i, (name, snr) in enumerate(zip(names, snrs)):
        ax1.annotate(f"{snr:.1f} dB", (i, snr + 1), ha="center", fontsize=10)

    # Plot 2: Throughput vs Distance
    ax2 = axes[1]
    ax2.bar(names, throughputs, color=colors, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Throughput (Mbps)", fontsize=11)
    ax2.set_xlabel("Scenario", fontsize=11)
    ax2.set_title("Estimated Throughput", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    for i, (name, tp) in enumerate(zip(names, throughputs)):
        ax2.annotate(f"{tp:.1f}", (i, tp + 0.3), ha="center", fontsize=10)

    # Plot 3: SNR and Throughput vs Distance (line plot)
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    line1 = ax3.plot(distances, snrs, "o-", color="#3498db", linewidth=2,
                     markersize=10, label="SNR (dB)")
    line2 = ax3_twin.plot(distances, throughputs, "s--", color="#e74c3c",
                          linewidth=2, markersize=10, label="Throughput (Mbps)")

    ax3.set_xlabel("Distance (m)", fontsize=11)
    ax3.set_ylabel("SNR (dB)", color="#3498db", fontsize=11)
    ax3_twin.set_ylabel("Throughput (Mbps)", color="#e74c3c", fontsize=11)
    ax3.set_title("Metrics vs Distance", fontsize=12)
    ax3.grid(alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Plot saved: {output_path}")


def print_summary_table(results: List[LinkResult]) -> None:
    """Print a formatted summary table of results.

    Parameters
    ----------
    results : List[LinkResult]
        Simulation results.
    """
    print("\n" + "=" * 70)
    print("  PHASE 0-1 MILESTONE: SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Scenario':<10} {'Distance':<12} {'SNR':<12} {'Throughput':<15} {'Backend'}")
    print(f"{'':─<10} {'':─<12} {'':─<12} {'':─<15} {'':─<20}")

    for result in results:
        print(f"{result.scenario.name:<10} "
              f"{result.scenario.distance_m:<12.0f} "
              f"{result.snr_db:<12.2f} "
              f"{result.throughput_mbps:<15.2f} "
              f"{result.backend}")

    print("=" * 70)

    # Validation checks
    print("\n✓ Validation Checks:")

    # Check monotonicity
    snrs = [r.snr_db for r in results]
    throughputs = [r.throughput_mbps for r in results]

    if snrs[0] > snrs[1] > snrs[2]:
        print("  [PASS] SNR decreases with distance (near > mid > far)")
    else:
        print("  [WARN] SNR does not decrease monotonically with distance")

    if throughputs[0] >= throughputs[1] >= throughputs[2]:
        print("  [PASS] Throughput decreases with distance (near >= mid >= far)")
    else:
        print("  [WARN] Throughput does not decrease monotonically with distance")

    # Check reasonable ranges
    if all(0 < snr < 100 for snr in snrs):
        print("  [PASS] SNR values in reasonable range (0-100 dB)")
    else:
        print("  [WARN] SNR values outside expected range")

    if all(0 < tp < 1000 for tp in throughputs):
        print("  [PASS] Throughput values in reasonable range (0-1000 Mbps)")
    else:
        print("  [WARN] Throughput values outside expected range")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for Phase 1 scenario runner."""
    parser = argparse.ArgumentParser(
        description="Run UAV link scenarios for Phase 0-1 milestone"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase1"),
        help="Output directory for results (default: data/phase1)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create scenarios
    scenarios = create_phase1_scenarios()

    # Run scenarios
    results = run_scenarios(scenarios)

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = args.output_dir / f"phase1_results_{timestamp}.csv"
    export_csv(results, csv_path)

    json_path = args.output_dir / f"phase1_results_{timestamp}.json"
    export_json(results, json_path)

    # Generate plot
    if not args.no_plot:
        plot_path = args.output_dir / f"phase1_results_{timestamp}.png"
        generate_plot(results, plot_path)

    # Print summary
    print_summary_table(results)

    print("\n" + "=" * 70)
    print("  PHASE 0-1 MILESTONE COMPLETE!")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
