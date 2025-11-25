# Sionna–pyAerial UAV Demo (Phase 0–1)

This repository is a scaffold for experiments combining:

- **NVIDIA Sionna / SionnaRT** for link- and system-level simulation.
- **NVIDIA Aerial pyAerial** running inside the Aerial CUDA-Accelerated RAN (ACAR) container.
- A future **beam management / interference-aware controller** for O-RAN (Phase 2+).

The current scope is **Phase 0–1 only**:

1. **Phase 0 – Three distance cases (short / mid / far)**  
   - Use Sionna (or an analytic free-space model as a placeholder) to simulate a single TX base station and a single UAV UE at three distances.  
   - Export basic link metrics: RSRP, SNR and an approximate Shannon-capacity throughput.  
   - Treat these as "test vectors" that will eventually be consumed by pyAerial / ACAR.

2. **Phase 1 – Interference-aware simulations & ML prep**  
   - Extend to SionnaRT-based scenarios with interference (stubs are provided).  
   - Read Sionna outputs via pyAerial (planned in `uav_acar_demo.aerial.py_aerial_interface`).  
   - Validate that ACAR / pyAerial throughput under interference is consistent with Sionna's predictions.  
   - Prepare datasets and simple neural models (dense + CNN) for later integration into ACAR.

The code here is intentionally minimal and designed to be completed using **Claude Code Opus 4.5**. It gives you a clean structure, configuration objects, basic physics for RSRP/SNR/throughput, and test scaffolding.

## Requirements

- Linux (Ubuntu 22.04 / 24.04 recommended)
- NVIDIA RTX 4090 with recent drivers and CUDA toolkit
- Python 3.10–3.12
- Docker (for pyAerial / ACAR work)
- Access to the **Aerial CUDA-Accelerated RAN** container and SDK (cuBB)

Python-side dependencies are defined in `pyproject.toml` and include:

- `numpy`, `scipy`, `matplotlib`
- `sionna` (PHY/SYS; SionnaRT optional)
- `pytest` and a few dev tools

## Quickstart (host-side, Phase 0 only)

```bash
# 1) Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install in editable mode with dev extras
pip install -e .[dev]

# 3) Run the Phase 0 demo (short / mid / far UAV distances)
python -m uav_acar_demo.cli phase0 --output-dir data/phase0

# 4) Run tests
pytest -q
```

This will generate a small JSON file under `data/phase0` with three scenarios and their RSRP/SNR/throughput. The numbers are produced by a free-space path loss model; the intent is that you later swap this for a proper Sionna channel model.

## pyAerial / ACAR integration (high-level)

pyAerial runs inside its own container, built from the **Aerial CUDA-Accelerated RAN** (ACAR) SDK. This repo assumes that:

- You have followed NVIDIA's docs to obtain the ACAR container and cuBB SDK.
- You can build and run the pyAerial container, mounting this repository as a volume.
- You will implement the actual `run_throughput_eval_from_sionna(...)` logic in
  `uav_acar_demo/aerial/py_aerial_interface.py` by calling pyAerial APIs or
  cuPHY test-vector pipelines inside the container.

For now, those functions are stubs with detailed docstrings describing the intended flow.

## Phase 1 (interference & ML) – what this scaffold gives you

- A place (`uav_acar_demo/sim/`) to add SionnaRT-based scene definitions and interference scenarios.
- A small `uav_acar_demo/ml/models.py` module with ready-to-use Keras models:
  - `build_dense_baseline(...)`
  - `build_cnn_baseline(...)`
- Test scaffolding under `tests/` so you can validate that your Sionna + pyAerial + ML plumbing is working as you iterate.

## Working with Claude Code

- See **`CLAUDE.md`** for detailed instructions on how Claude Code should work in this repo.
- Project slash-commands live under `.claude/commands/`.
- A dedicated project Skill for this experiment lives under `.claude/skills/uav-sionna-pyaerial/`.

The goal is that you can open this repo in Claude Code, run `/help`, and immediately have a repeatable workflow for setting up the environment, running Phase 0, and gradually filling in the pyAerial / SionnaRT / RL pieces.
