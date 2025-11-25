---
description: Run the Phase 0 three-distance UAV demo and show where results were written.
argument-hint: "[no args]"
allowed-tools: Bash(python:*), Bash(./scripts/run_phase0.sh:*)
---

To run the Phase 0 pipeline:

1. Ensure the virtualenv is activated (or call `./scripts/setup_env.sh` first).
2. Prefer using the helper script so the command is logged:

   ```bash
   ./scripts/run_phase0.sh
   ```

3. After running, open `data/phase0/phase0_metrics.json` and summarize:
   - distances
   - RSRP (dBm)
   - SNR (dB)
   - throughput (Mbit/s)

4. If anything fails, inspect the traceback, fix the issue, and re-run the script.
