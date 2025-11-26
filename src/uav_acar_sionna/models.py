from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LinkResult:
    """Simple container for link-level / emulation results.

    This is intentionally generic so it can represent both:
    - Pure Sionna link-level simulations (Phase 1).
    - Full cuBB + RU emulator results (Phase 2).
    """

    scenario_id: str
    time_point: float
    snr_db: float
    sinr_db: float
    throughput_mbps: float
    bler: float = 0.0
    notes: str | None = None


@dataclass
class EmulationArtifacts:
    """Paths to the main artifacts of a single emulation run.

    All paths are absolute; the orchestrator is responsible for constructing
    them based on the repo layout.
    """

    sionna_npz: Path
    matlab_h5_for_cubb: Path
    matlab_h5_for_ru: Path
    throughput_log: Path
