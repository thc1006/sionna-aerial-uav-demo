from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SummaryRow:
    """Row format for data/phase2_interference/summary.csv.

    This mirrors the interference-aware schema we designed earlier, but keeps
    only the core fields needed for a minimal prototype. You can safely extend
    it later (just make sure to update the header writing logic).
    """

    scenario_id: str
    time_point: float
    distance_m: float
    snr_db: float
    sinr_db: float
    throughput_mbps: float
    bler: float
    sionna_npz_path: str
    matlab_h5_cubb_path: str
    matlab_h5_ru_path: str
    throughput_log_path: str
    notes: str = ""


def append_summary_row(csv_path: Path, row: SummaryRow) -> None:
    """Append a SummaryRow to the CSV, writing a header if file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = list(asdict(row).keys())

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(row))
