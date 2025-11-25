from __future__ import annotations

"""Stubs for interacting with NVIDIA pyAerial / ACAR.

Phase 0 in this repo does not require pyAerial yet; these functions are
design sketches for Phase 1, where you will:

  1. Mount Sionna-generated data (under `data/`) into the pyAerial container.
  2. Use pyAerial's APIs (e.g., PUSCH link simulation, dataset generation)
     to compute throughput and other KPIs for those scenarios.
  3. Feed the results back to the host as structured JSON/CSV so they can
     be compared to the analytic/Sionna predictions.

See NVIDIA's "Getting Started with pyAerial" documentation for container
build and installation details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass
class PyAerialConfig:
    """Host-side configuration for pyAerial integration.

    Attributes
    ----------
    cubb_root:
        Path to the cuBB SDK checkout on the host.
    test_vector_dir:
        Directory with test vectors or Sionna-exported data to be consumed
        by pyAerial / cuPHY.
    results_dir:
        Directory where throughput / KPI results from pyAerial should be
        written (e.g., JSON or CSV).
    """

    cubb_root: Path
    test_vector_dir: Path
    results_dir: Path


def run_throughput_eval_from_sionna(
    cfg: PyAerialConfig,
) -> Mapping[str, Any]:
    """Placeholder for running throughput evaluation in pyAerial.

    Expected high-level behavior (to be implemented):

      1. Ensure that `cfg.test_vector_dir` and `cfg.results_dir` are created.
      2. Launch or reuse the pyAerial container built from the ACAR SDK.
      3. Mount the cuBB SDK and this repository into the container.
      4. Inside the container, run a small Python script that:
         - Reads the Sionna-generated metrics/test vectors.
         - Configures pyAerial pipelines accordingly.
         - Computes throughput (and possibly BLER, latency, etc.).
         - Writes a JSON results file to `cfg.results_dir` that maps
           scenario names to throughput and KPIs.
      5. Back on the host, parse that JSON and return it as a dict.

    The exact implementation depends on your local pyAerial setup, so this
    function intentionally raises NotImplementedError for now.
    """
    raise NotImplementedError(
        "Implement pyAerial throughput evaluation using the NVIDIA docs and your local ACAR setup."
    )
