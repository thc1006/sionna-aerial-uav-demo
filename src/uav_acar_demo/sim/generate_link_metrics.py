from __future__ import annotations

"""Simple Phase-0 link metric generation.

This module currently uses an analytic free-space path loss model to produce
RSRP, SNR, and a Shannon-capacity-style throughput estimate for three
canonical UAV distance scenarios (short/mid/far).

The intent is that you can later:
  * Replace the analytic model with a Sionna channel object.
  * Keep the public API and tests stable so higher layers (pyAerial/ACAR)
    do not need to change.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

from ..config import UavScenarioConfig
from .sionna_scenarios import default_scenarios

SPEED_OF_LIGHT = 3.0e8  # m/s
BOLTZMANN = 1.380649e-23  # J/K


def free_space_pathloss_db(distance_m: float, carrier_freq_hz: float) -> float:
    """Free-space path loss in dB (Friis)."""
    wavelength = SPEED_OF_LIGHT / carrier_freq_hz
    # 20 log10(4 pi d / lambda)
    return 20.0 * np.log10(4.0 * np.pi * distance_m / wavelength)


def noise_power_dbm(
    bandwidth_hz: float, noise_figure_db: float = 7.0, temp_k: float = 290.0
) -> float:
    """Thermal noise power (including NF) in dBm."""
    noise_watts = BOLTZMANN * temp_k * bandwidth_hz
    noise_dbm = 10.0 * np.log10(noise_watts * 1e3)  # W -> mW
    return noise_dbm + noise_figure_db


def simulate_link_metrics(cfg: UavScenarioConfig) -> Dict[str, float]:
    """Compute RSRP, SNR and approximate throughput for one scenario.

    Returns a dict with keys:
      - distance_m
      - rsrp_dbm
      - snr_db
      - throughput_mbps
    """
    pl_db = free_space_pathloss_db(cfg.distance_m, cfg.carrier_freq_hz)
    rsrp_dbm = cfg.tx_power_dbm - pl_db
    n_dbm = noise_power_dbm(cfg.bandwidth_hz, cfg.noise_figure_db)
    snr_db = rsrp_dbm - n_dbm
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Shannon capacity as an upper bound (single stream)
    capacity_bps = cfg.bandwidth_hz * np.log2(1.0 + snr_linear)
    throughput_mbps = float(capacity_bps / 1e6)

    return {
        "distance_m": float(cfg.distance_m),
        "rsrp_dbm": float(rsrp_dbm),
        "snr_db": float(snr_db),
        "throughput_mbps": throughput_mbps,
    }


def run_phase0(output_dir: Path) -> Path:
    """Run Phase 0 metric generation and write a JSON summary.

    Parameters
    ----------
    output_dir:
        Directory where `phase0_metrics.json` will be saved.

    Returns
    -------
    Path
        The path to the JSON file that was written.
    """
    scenarios = default_scenarios()
    metrics = {}

    for cfg in (scenarios.short, scenarios.mid, scenarios.far):
        metrics[cfg.name] = simulate_link_metrics(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase0_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return out_path
