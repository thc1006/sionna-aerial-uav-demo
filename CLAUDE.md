# Claude Code guide for `sionna-aerial-uav-demo`

You are **Claude Code Opus 4.5** acting as my co-developer for this project.
The goal is to prototype a Sionna → pyAerial → ACAR pipeline for UAV link
simulations (Phase 0–1 only). This file describes how you should behave in
this repository.

---

## Project goals

1. **Phase 0 – Deterministic three-distance demo**
   - Simulate a single gNB–UAV link at three distances (short/mid/far).
   - Compute RSRP, SNR, and an approximate throughput using a simple PHY model.
   - Export metrics as a JSON "test vector" that can later feed pyAerial / ACAR.
   - Provide unit tests that guarantee the expected monotonic behavior
     (near > mid > far for RSRP/SNR/throughput).

2. **Phase 1 – Interference-aware & ML-ready**
   - Add SionnaRT-based scenarios with and without interferers.
   - Implement thin adapters that read Sionna outputs into pyAerial and ACAR.
   - Prepare datasets and basic neural models (dense + CNN) for interference
     sensing and throughput prediction.
   - Keep all of this testable and reproducible so we can deploy to ACAR later.

You do **not** need to implement full O-RAN integration yet. Focus on having
a clean Python API, test coverage, and clear boundaries between Sionna,
pyAerial/ACAR, and ML code.

---

## Environment assumptions

- OS: Ubuntu 22.04 or 24.04
- GPU: NVIDIA RTX 4090 (compute capability 8.9)
- Python: 3.10–3.12
- CUDA drivers and toolkit are already installed and working
- Docker is installed and configured (for pyAerial / ACAR work)

Python dependencies are managed via `pyproject.toml`. Prefer using a virtual
environment or `uv` for reproducible installs.

---

## Golden rules for this repo

1. **Always log shell commands.**  
   - Wrap commands with `scripts/ccmd.sh` whenever possible. For example:  
     `scripts/ccmd.sh python -m uav_acar_demo.cli phase0`  
   - If you must run a command directly via the Bash tool, first record it in
     `logs/commands.log` using the same format as `ccmd.sh`.

2. **Run tests frequently.**  
   - After non-trivial changes, run `pytest -q`.  
   - If you touch Sionna simulation code, re-run `tests/test_sionna_phase0.py`.  
   - If you touch the ML module, re-run `tests/test_ml_models.py`.

3. **Prefer small, incremental changes.**  
   - Edit one module at a time.  
   - Explain in plain language _what_ you changed and _why_ in your commit
     messages or summaries.

4. **Keep Sionna vs. pyAerial concerns separated.**
   - `uav_acar_demo.sim` owns channel modeling, RSRP/SNR computation, etc.  
   - `uav_acar_demo.aerial` owns the bridge to pyAerial / ACAR.  
   - `uav_acar_demo.ml` owns model definitions and training utilities.

5. **Use the provided tests as contracts.**  
   - Do not change test expectations lightly. If you need to, explain the
     physical reasoning (e.g., different path loss model).

---

## Typical workflows

### 1. Set up the host Python environment

1. Create/activate a virtualenv (or use `uv`):
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -e .[dev]`
2. Verify Sionna + TensorFlow can see the GPU:
   - `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

There is a project slash-command `/setup-env` defined in `.claude/commands`
that describes this sequence.

### 2. Run the Phase 0 pipeline

- Main entry point: `python -m uav_acar_demo.cli phase0 --output-dir data/phase0`  
- This should:
  - Instantiate three `UavScenarioConfig` objects (short/mid/far).  
  - Call `simulate_link_metrics(...)` for each.  
  - Write `phase0_metrics.json` under `data/phase0/`.  

A project slash-command `/phase0-run` is provided as a reminder of this flow.

### 3. Extend Sionna modeling

When asked to "make the Sionna model more realistic":

1. Inspect `uav_acar_demo/sim/sionna_scenarios.py` and
   `uav_acar_demo/sim/generate_link_metrics.py`.
2. Replace or augment the simple free-space model with:
   - Sionna channel objects (e.g., SISO/OFDM channel, fading models).  
   - Eventually, SionnaRT scenes for 3D ray tracing.
3. Keep the public function signatures stable so tests and higher-level
   code continue to work.

### 4. Bridge to pyAerial / ACAR

When implementing pyAerial integration:

1. Read the docstrings in `uav_acar_demo/aerial/py_aerial_interface.py`.  
2. Follow NVIDIA's pyAerial docs to call the relevant APIs inside the
   pyAerial container.  
3. Use volume mounts so Sionna-generated data under `data/` is visible
   inside the container.  
4. Return structured results (throughput per scenario) from
   `run_throughput_eval_from_sionna(...)`.

For now, these functions raise `NotImplementedError` and serve as a design
sketch for Phase 1 work.

---

## Files and directories you should know

- `pyproject.toml` – dependency + package definition.
- `src/uav_acar_demo/config.py` – configuration dataclasses.
- `src/uav_acar_demo/cli.py` – simple CLI for Phase 0 and future phases.
- `src/uav_acar_demo/sim/` – Sionna / path loss modeling and metric generation.
- `src/uav_acar_demo/aerial/` – pyAerial / ACAR integration stubs.
- `src/uav_acar_demo/ml/models.py` – Keras models (dense + CNN) for Phase 1.
- `tests/` – unit tests for Phase 0 and ML module.
- `scripts/ccmd.sh` – command logger wrapper.
- `.claude/commands/` – project slash-commands (setup env, run Phase 0, run tests).
- `.claude/skills/uav-sionna-pyaerial/` – project Skill for this experiment.

---

## Style & quality guidelines

- Use type hints and docstrings for all public functions.  
- Prefer pure functions where possible; avoid hidden global state.  
- Use SI units consistently (meters, seconds, Hz, dBm, dB).  
- When approximating 5G/6G behavior, leave comments with references
  (3GPP/O-RAN/Sionna docs) so future work can refine the model.  
- Keep modules small and focused; if a file grows beyond ~300 lines, propose
  a refactor.

---

## TODO map (for future Claude Code sessions)

- [ ] Replace the analytic free-space model with a Sionna channel.  
- [ ] Add SionnaRT-based interference scenarios and small test scenes.  
- [ ] Implement `run_throughput_eval_from_sionna(...)` using pyAerial.  
- [ ] Add dataset writers (NumPy/Parquet) for training ML models.  
- [ ] Implement training scripts using `uav_acar_demo.ml.models`.  
- [ ] Wire the trained model back into ACAR as a pluggable component.

Treat this TODO list as a living document: update it as the project evolves.
