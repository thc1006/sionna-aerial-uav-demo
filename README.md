# UAV-ACAR-Sionna Phase 2 (single-host skeleton)

This repo is a **single-host developer skeleton** for your UAV–ACAR–Sionna pipeline.

It assumes you currently have only one machine (e.g. a DGX / RTX 4090 box), but
you want the code to be **host-agnostic** so that in the future you can clone it
onto multiple hosts (Blender host, Sionna/Orchestrator host, cuBB host, RU emulator host)
without redesigning the repo.

Right now, everything runs on a *single* machine with logical host roles:

- Host A (Blender / scene orchestrator)
- Host B (SionnaRT + MATLAB + dataset writer + emulation orchestrator)
- cuBB host (testMAC + cuPHY)
- RU emulator host

On this single machine, all four roles are collocated; we still keep them
separate in code so that later you can move them to different boxes by changing
environment variables and connection settings, instead of rewriting logic.

## What is implemented in this skeleton?

This Phase 2 skeleton focuses on **Host B + cuBB/RU roles**:

- A minimal **orchestrator** that represents the steps

  1. Ensure SionnaRT `.npz` exists for a given `(scenario_id, time_point)`.
  2. Convert `.npz` to cuBB **TestVector `.h5`** via a MATLAB adapter (stub).
  3. Copy the `.h5` into locations representing the cuBB host & RU emulator host.
  4. Launch an emulation (currently stubbed) that produces a **throughput log**.
  5. Append a row to `data/phase2_interference/summary.csv` that follows the
     interference-aware dataset schema we designed.

- A small CLI:

  ```bash
  # After installing in a virtualenv:
  run_emulation_once --scenario-id uav_demo1 --time-point 1.0
  ```

  This currently uses **toy stubs** instead of واقعی Sionna / MATLAB / cuBB,
  but the file layout and function boundaries are aligned with how you will
  integrate the real tools.

- A data layout under `data/`:

  - `data/phase1/` – where you can drop the Phase 1 CSVs and plots.
  - `data/phase2_interference/summary.csv` – one row per `(scenario_id, time_point)`.
  - `data/phase2_interference/interferers.csv` – optional, one row per interferer
    (not auto-populated yet, but schema is defined in code).

- A tiny `tests/test_orchestrator_smoke.py` that exercises the orchestrator and
  verifies that it writes a row into `summary.csv`. This gives Claude Code a
  concrete test to keep passing while you swap the stubs for real integrations.

## How to use this with Claude Code

See **CLAUDE.md** for detailed instructions. Short version:

1. Create a virtualenv on your single machine (Linux recommended for Aerial):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .[dev]
   ```

2. Point Claude Code at this repo and ask it to:

   - Replace the stub Sionna adapter with real SionnaRT calls.
   - Replace the stub MATLAB adapter with calls that invoke your MATLAB scripts
     to generate `.h5` TVs.
   - Replace the stub cuBB adapter with scripts that launch testMAC + cuPHY +
     RU emulator and parse the official throughput logs.

3. As you scale out to multiple machines later, you can:

   - Set `UAV_ACAR_HOST_ROLE=orchestrator` on the box that runs SionnaRT + MATLAB.
   - Set `UAV_ACAR_HOST_ROLE=cubb` on the cuBB host.
   - Set `UAV_ACAR_HOST_ROLE=ru_emu` on the RU emulator host.

   The `host_roles.py` module is designed so you can swap local filesystem calls
   for SSH / REST calls without touching the higher-level orchestration logic.

## Where to plug in NVIDIA and Sionna tools

- **SionnaRT / Sionna PHY**

  Install the official Sionna / Sionna-RT packages and follow their
  tutorials for generating ray-traced channels and CIR datasets.\
  Sionna RT is the stand-alone ray tracing package of Sionna built on top of
  Mitsuba 3 and interoperable with TensorFlow/PyTorch/JAX. 【sionna-rt docs】

- **Aerial cuBB (testMAC, cuPHY, RU emulator)**

  Set up the Aerial cuBB containers as per NVIDIA's documentation. cuPHY-CP
  includes the built-in testMAC and RU emulator modules for end-to-end testing
  of 5G L1/L2 with FAPI. 【Aerial cuBB docs】

- **MATLAB TV generation**

  NVIDIA provides official guidance on using MATLAB to generate test vectors
  and launch patterns for cuPHY and cuBB. That is the natural place to connect
  your `.npz` → `.h5` conversion step. 【Running Aerial cuPHY docs】

For concrete links, look at the top comments inside the adapter modules
(`sionna_adapter.py`, `matlab_adapter.py`, `cubb_adapter.py`).
