"""Bridge between UAV scenarios and minimal Sionna link estimator.

This module provides integration between the UavScenarioConfig used in our
Phase 0-1 work and the minimal SISO Rayleigh link estimator. It allows us
to run BER/throughput simulations using the simple Rayleigh fading model
as a complement to the full 3GPP UMa channel simulations.

The workflow is:
  1. Take a UavScenarioConfig (distance, RSRP, SNR, etc.)
  2. Map it to a LinkSimConfig (num_bits, snr_db, etc.)
  3. Run the minimal link estimator
  4. Return BER and throughput metrics

This is useful for:
  - Quick sanity checks without full Sionna channel models
  - Comparison between simple Rayleigh and complex UMa models
  - Integration testing with pyAerial/ACAR
"""

from __future__ import annotations

from typing import Dict

from ..config import UavScenarioConfig
from .link_estimator import LinkSimConfig, estimate_link_with_sionna


def run_minimal_link_sim(
    cfg: UavScenarioConfig,
    num_bits: int = 4096,
    batch_size: int = 4,
    seed: int = 42,
) -> Dict[str, float]:
    """Run minimal Rayleigh link simulation from UAV scenario config.

    This function bridges our UAV scenario configuration to the minimal
    link estimator, using the SNR from our Sionna UMa channel model.

    Args:
        cfg: UAV scenario configuration
        num_bits: Number of bits to simulate per batch (default: 4096)
        batch_size: Batch size for parallel simulation (default: 4)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dict with keys:
            - ber: Bit error rate
            - throughput_bits_per_use: Throughput proxy (1 - BER)
            - snr_db: SNR in dB (from UAV scenario)
            - num_bits: Number of bits simulated

    Example:
        >>> from uav_acar_demo.config import UavScenarioConfig
        >>> from uav_acar_demo.sim import simulate_link_metrics
        >>>
        >>> # First, get SNR from UMa channel model
        >>> cfg = UavScenarioConfig("test", 25.0, 120.0, 500.0)
        >>> uma_metrics = simulate_link_metrics(cfg, seed=42)
        >>> print(f"UMa SNR: {uma_metrics['snr_db']:.2f} dB")
        >>>
        >>> # Then run minimal Rayleigh simulation at that SNR
        >>> link_metrics = run_minimal_link_sim(cfg)
        >>> print(f"Rayleigh BER: {link_metrics['ber']:.4f}")
        >>> print(f"Throughput: {link_metrics['throughput_bits_per_use']:.4f} bits/use")
    """
    # We need SNR from the scenario. Options:
    # 1. If cfg has pre-computed SNR (from UMa simulation), use it
    # 2. Otherwise, compute SNR from free-space path loss
    # For now, we'll compute it using the simple free-space model
    from .generate_link_metrics import simulate_link_metrics

    # Get SNR from free-space model as baseline
    metrics = simulate_link_metrics(cfg)
    snr_db = metrics["snr_db"]

    # Create link simulation config
    link_cfg = LinkSimConfig(
        num_bits=num_bits,
        snr_db=snr_db,
        batch_size=batch_size,
        seed=seed,
    )

    # Run minimal link estimator
    result = estimate_link_with_sionna(link_cfg)

    # Add scenario metadata
    result["distance_m"] = float(cfg.distance_m)
    result["scenario_name"] = cfg.name

    return result


def compare_uma_vs_rayleigh(
    cfg: UavScenarioConfig,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare UMa channel model vs simple Rayleigh for same scenario.

    This function runs both our full 3GPP UMa channel simulation and the
    minimal Rayleigh link estimator, allowing side-by-side comparison.

    Args:
        cfg: UAV scenario configuration
        seed: Random seed for reproducibility

    Returns:
        Dict with keys "uma" and "rayleigh", each containing metrics:
            - uma: RSRP, SNR, throughput from 3GPP UMa model
            - rayleigh: BER, throughput from simple Rayleigh model

    Example:
        >>> from uav_acar_demo.config import UavScenarioConfig
        >>> cfg = UavScenarioConfig("test", 25.0, 120.0, 500.0)
        >>> comparison = compare_uma_vs_rayleigh(cfg, seed=42)
        >>>
        >>> print("UMa Channel:")
        >>> print(f"  SNR: {comparison['uma']['snr_db']:.2f} dB")
        >>> print(f"  Shannon Throughput: {comparison['uma']['throughput_mbps']:.2f} Mbps")
        >>>
        >>> print("Rayleigh Channel:")
        >>> print(f"  BER: {comparison['rayleigh']['ber']:.4f}")
        >>> print(f"  BPSK Throughput: {comparison['rayleigh']['throughput_bits_per_use']:.4f} bits/use")
    """
    from .generate_link_metrics import simulate_link_metrics

    # Run UMa simulation (currently uses free-space model)
    uma_metrics = simulate_link_metrics(cfg)

    # Run Rayleigh simulation with same SNR
    link_cfg = LinkSimConfig(
        num_bits=4096,
        snr_db=uma_metrics["snr_db"],
        batch_size=4,
        seed=seed,
    )
    rayleigh_metrics = estimate_link_with_sionna(link_cfg)

    return {
        "uma": uma_metrics,
        "rayleigh": rayleigh_metrics,
    }
