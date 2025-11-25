"""UAV-ACAR-Sionna integration package.

This package provides:
- UAV scenario definitions and configurations
- Sionna backend for link-level simulations
- pyAerial/ACAR interface stubs
- ML models for interference sensing
"""

from .config import UavScenarioConfig, LinkResult, DATA_ROOT
from .sionna_backend import (
    estimate_link_with_sionna_backend,
    scenario_to_link_config,
)

__all__ = [
    "UavScenarioConfig",
    "LinkResult",
    "DATA_ROOT",
    "estimate_link_with_sionna_backend",
    "scenario_to_link_config",
]

__version__ = "1.0.0"
