from __future__ import annotations

"""Scenario definitions for the UAV Sionna/analytic simulations.

For Phase 0 we keep this intentionally simple: three fixed distances (short,
mid, far). Later you can extend this module to use full Sionna channel
objects or SionnaRT scenes while preserving the same public interface.
"""

from dataclasses import dataclass
from typing import Dict

from ..config import UavScenarioConfig


@dataclass(frozen=True)
class ScenarioSet:
    """Container for the three canonical UAV distance cases."""

    short: UavScenarioConfig
    mid: UavScenarioConfig
    far: UavScenarioConfig


def default_scenarios() -> ScenarioSet:
    """Return the default (short/mid/far) UAV scenarios.

    Distances are chosen arbitrarily here; you should tune them to match
    whatever geometry you later use in Sionna/SionnaRT.
    """

    short = UavScenarioConfig(
        name="short",
        tx_height_m=25.0,
        ue_height_m=120.0,
        distance_m=200.0,
    )
    mid = UavScenarioConfig(
        name="mid",
        tx_height_m=25.0,
        ue_height_m=120.0,
        distance_m=500.0,
    )
    far = UavScenarioConfig(
        name="far",
        tx_height_m=25.0,
        ue_height_m=120.0,
        distance_m=1000.0,
    )
    return ScenarioSet(short=short, mid=mid, far=far)
