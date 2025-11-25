from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATA_ROOT
from .sim.generate_link_metrics import run_phase0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UAV Sionna–pyAerial demo CLI (Phase 0–1)."
    )
    parser.add_argument(
        "phase",
        choices=["phase0"],
        help="Which high-level pipeline to run. Only 'phase0' is implemented for now.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_ROOT / "phase0"),
        help="Directory where outputs (JSON/NPZ) will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "phase0":
        run_phase0(output_dir)
    else:
        raise SystemExit(f"Unsupported phase: {args.phase!r}")


if __name__ == "__main__":
    main()
