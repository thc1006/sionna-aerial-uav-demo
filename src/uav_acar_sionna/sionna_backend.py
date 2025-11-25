"""Sionna backend for UAV link simulations.

This module bridges high-level UAV scenarios to the minimal Sionna-like
link estimator, providing SNR estimation and throughput calculation.
"""

from __future__ import annotations

import math

from .config import UavScenarioConfig, LinkResult
from sionna_link_minimal import LinkSimConfig, estimate_link_with_sionna


def _friis_snr_db(scn: UavScenarioConfig, kT_noise_dbmhz: float = -174.0) -> float:
    """Very rough SNR estimate using free-space path loss.

    This is *not* intended to be a faithful 5G model – it is just enough
    to map a high-level UAV geometry into a scalar SNR in dB, which is
    then passed down to the Rayleigh+BPSK link model as a starting point.

    Parameters
    ----------
    scn :
        High-level UAV scenario description.
    kT_noise_dbmhz :
        Thermal noise spectral density in dBm/Hz (default -174 dBm/Hz).

    Returns
    -------
    snr_db : float
        Estimated link SNR in dB.
    """
    # Free-space path loss (Friis) in dB:
    #   PL(dB) = 32.44 + 20 log10(d_km) + 20 log10(f_MHz)
    d_km = scn.distance_m / 1e3
    f_mhz = scn.carrier_freq_ghz * 1e3

    path_loss_db = 32.44 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(
        max(f_mhz, 1e-3)
    )

    rx_power_dbm = scn.tx_power_dbm - path_loss_db

    # Noise power in dBm over the system bandwidth
    noise_dbm = (
        kT_noise_dbmhz
        + 10.0 * math.log10(scn.bandwidth_hz)
        + scn.noise_figure_db
    )

    snr_db = rx_power_dbm - noise_dbm
    return snr_db


def scenario_to_link_config(scn: UavScenarioConfig) -> LinkSimConfig:
    """Map a UAV scenario into the low-level link simulator config.

    This function is the *bridge* between your ACAR/UAV world and the
    minimal Sionna-like backend. You can refine this mapping over time
    (e.g., to encode MCS, code rate, multi-antenna gains, etc).
    """
    snr_db = _friis_snr_db(scn)

    return LinkSimConfig(
        num_bits=4096,
        snr_db=snr_db,
        batch_size=4,
        seed=scn.seed,
    )


def estimate_link_with_sionna_backend(scn: UavScenarioConfig) -> LinkResult:
    """High-level entry point used by the rest of the project.

    This calls the minimal Rayleigh+BPSK link simulator and then converts
    the resulting "bits per channel use" throughput proxy into Mbps using
    the scenario's bandwidth.

    In a more realistic setting, you will likely replace this with a
    mapping that takes modulation, code rate, symbol rate and resource
    grid structure into account. The present form is intentionally simple
    so you can focus on end-to-end plumbing first.
    """
    cfg = scenario_to_link_config(scn)
    metrics = estimate_link_with_sionna(cfg)

    spectral_eff = metrics["throughput_bits_per_use"]  # bits / channel use
    # Very rough: assume 1 channel use per Hz per second
    throughput_mbps = spectral_eff * scn.bandwidth_hz / 1e6

    return LinkResult(
        scenario=scn,
        rsrp_dbm=None,
        snr_db=metrics["snr_db"],
        throughput_mbps=throughput_mbps,
        backend="sionna_minimal",
        notes="SISO Rayleigh+BPSK backend with Friis SNR estimate",
    )
