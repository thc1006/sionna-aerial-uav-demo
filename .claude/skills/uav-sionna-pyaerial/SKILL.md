---
name: uav-sionna-pyaerial
description: >
  Help design and implement Sionna/SionnaRT + pyAerial + ACAR experiments for
  UAV links (Phase 0–1), including link metric simulation, interference-aware
  evaluation, and simple ML models.
version: 0.1.0
tags:
  - wireless
  - sionna
  - pyAerial
  - 5g
  - ml
---

# UAV Sionna–pyAerial Skill

This Skill teaches you how to work effectively inside the
`sionna-aerial-uav-demo` repository.

## When to use this Skill

- The user mentions Sionna, SionnaRT, pyAerial, ACAR, cuBB, or UAV link simulations.
- The user asks for Phase 0 or Phase 1 experiments, RSRP/SNR/throughput analysis,
  interference modeling, or training a neural model on simulated data.
- The user wants to integrate Sionna-generated data with NVIDIA Aerial.

## High-level process

1. **Understand the scenario.**
   - Open `README.md` and `CLAUDE.md` to recall the Phase 0–1 goals.
   - Identify whether the task touches simulation (`uav_acar_demo.sim`),
     pyAerial integration (`uav_acar_demo.aerial`), or ML (`uav_acar_demo.ml`).

2. **For Phase 0 tasks (three-distance demo):**
   - Inspect `uav_acar_demo/sim/sionna_scenarios.py` and
     `uav_acar_demo/sim/generate_link_metrics.py`.
   - Maintain the contract of `simulate_link_metrics(...)` and `run_phase0(...)`.
   - Ensure tests in `tests/test_sionna_phase0.py` keep passing.
   - If improving the model (e.g., swapping in Sionna channels), document
     the assumptions in comments.

3. **For Phase 1 tasks (interference & ML):**
   - Add or extend scenario definitions to include interferers and SionnaRT
     scenes (geometry, materials, etc.).
   - Implement the stubs in `uav_acar_demo/aerial/py_aerial_interface.py`
     using pyAerial APIs inside the ACAR container.
   - Use `uav_acar_demo/ml/models.py` as the starting point for dense/CNN
     models and add training scripts as needed.
   - Always think about data formats (NumPy, NPZ, Parquet, etc.) and keep
     them consistent.

4. **Always keep things testable.**
   - Add or extend tests in `tests/` for new behavior.
   - Prefer deterministic seeds for stochastic simulations where possible.

## Implementation guidelines

- Use SI units and document any approximations (e.g., free-space path loss).
- Avoid hard-coding absolute paths; use `Path` objects relative to the repo.
- When interacting with Docker/pyAerial, prefer small wrapper scripts that
  can be called from the host and inspected easily.
- When adding ML models, keep them small and interpretable first; only scale
  up once the end-to-end pipeline works.

## Example tasks for this Skill

- "Replace the free-space path loss in Phase 0 with a simple Sionna channel."
- "Add an interference term based on a second TX and re-run the metrics."
- "Design a training set and Keras script to map Sionna features to throughput."
- "Wire up pyAerial so that it consumes Sionna-generated test vectors."
