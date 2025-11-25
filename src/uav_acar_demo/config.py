from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UavScenarioConfig:
    """Configuration for a simple single-link UAV scenario.

    All distances in meters, frequency in Hz, bandwidth in Hz, powers in dBm.

    This is intentionally simple: the idea is that Phase 0 uses an analytic
    free-space model, while later phases can plug in Sionna/SionnaRT-based
    channels without changing this interface.
    """

    name: str
    tx_height_m: float
    ue_height_m: float
    distance_m: float
    carrier_freq_hz: float = 3.5e9
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 7.0
    bandwidth_hz: float = 20e6


DATA_ROOT = Path("data").resolve()
