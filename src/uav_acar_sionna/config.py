from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class UavScenarioConfig:
    """Configuration for a single UAV link scenario.

    Combines Phase 0 geometry fields with Phase 1 simplified interface.
    All distances in meters, frequency in Hz/GHz, bandwidth in Hz, powers in dBm.

    Attributes
    ----------
    distance_m : float
        3D distance between gNB (TX) and UAV (RX) in meters.
    name : str
        Scenario identifier (e.g., 'short', 'mid', 'far').
    tx_height_m : float
        gNB antenna height in meters.
    ue_height_m : float
        UAV height in meters.
    carrier_freq_ghz : float
        Carrier frequency in GHz (default 3.5 GHz).
    bandwidth_hz : float
        System bandwidth in Hz (default 20 MHz).
    tx_power_dbm : float
        Transmit power at gNB in dBm (default 30 dBm).
    noise_figure_db : float
        Receiver noise figure in dB (default 5 dB).
    seed : int
        Random seed for reproducibility (default 42).
    """
    distance_m: float
    name: str = "default"
    tx_height_m: float = 25.0
    ue_height_m: float = 120.0
    carrier_freq_ghz: float = 3.5
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 5.0
    seed: int = 42


@dataclass
class LinkResult:
    """Container for link-level simulation results.

    Attributes
    ----------
    scenario : UavScenarioConfig
        Scenario that was simulated.
    rsrp_dbm : Optional[float]
        Reference signal received power in dBm (if available).
    snr_db : float
        SNR at the receiver in dB.
    throughput_mbps : float
        Estimated user throughput in Mbps.
    backend : str
        Name of the backend used (e.g., 'sionna_minimal').
    notes : Optional[str]
        Free-form notes (e.g., modeling assumptions).
    """
    scenario: UavScenarioConfig
    rsrp_dbm: Optional[float]
    snr_db: float
    throughput_mbps: float
    backend: str
    notes: Optional[str] = None


DATA_ROOT = Path("data").resolve()
