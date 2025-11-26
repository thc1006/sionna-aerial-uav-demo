from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models import LinkResult, EmulationArtifacts
from . import dataset_writer
from .cubb_adapter import run_emulation_with_cubb
from .host_roles import current_role
from .sionna_adapter import SionnaConfig, ensure_sionna_npz
from .matlab_adapter import generate_testvector_h5


@dataclass
class EmulationConfig:
    """High-level configuration for a single emulation run."""

    scenario_id: str
    time_point: float
    distance_m: float = 500.0  # can be overridden by caller


def get_repo_root() -> Path:
    """Return the repository root assuming this file lives under src/"""
    return Path(__file__).resolve().parents[3]


def build_artifact_paths(cfg: EmulationConfig) -> EmulationArtifacts:
    root = get_repo_root()
    phase2_root = root / "data" / "phase2_interference"

    base_name = f"{cfg.scenario_id}_t{cfg.time_point:.3f}".replace(".", "p")

    sionna_npz = phase2_root / "raw" / f"{base_name}.npz"
    h5_cubb = phase2_root / "raw" / f"{base_name}_cubb.h5"
    h5_ru = phase2_root / "raw" / f"{base_name}_ru.h5"
    log_path = root / "logs" / "emulation" / f"{base_name}.log"

    return EmulationArtifacts(
        sionna_npz=sionna_npz,
        matlab_h5_for_cubb=h5_cubb,
        matlab_h5_for_ru=h5_ru,
        throughput_log=log_path,
    )


def run_emulation_once(cfg: EmulationConfig) -> LinkResult:
    """End-to-end orchestration of a *single* emulation.

    In this skeleton everything runs locally, but the function is structured in
    a way that later allows remote calls (SSH/REST) without changing the
    high-level flow.
    """
    artifacts = build_artifact_paths(cfg)

    # 1) Ensure SionnaRT output (.npz) exists.
    s_cfg = SionnaConfig(
        scenario_id=cfg.scenario_id,
        time_point=cfg.time_point,
        distance_m=cfg.distance_m,
    )
    snr_db, sinr_db = ensure_sionna_npz(artifacts.sionna_npz, s_cfg)

    # 2) Convert .npz → TestVector .h5 (for cuBB host and RU emulator).
    generate_testvector_h5(artifacts.sionna_npz, artifacts.matlab_h5_for_cubb)
    generate_testvector_h5(artifacts.sionna_npz, artifacts.matlab_h5_for_ru)

    # 3) Launch cuBB + RU emulator and capture throughput log.
    throughput_mbps, bler = run_emulation_with_cubb(
        artifacts.matlab_h5_for_cubb,
        artifacts.matlab_h5_for_ru,
        snr_db=snr_db,
        sinr_db=sinr_db,
        log_path=artifacts.throughput_log,
    )

    # 4) Append a row to the summary dataset.
    root = get_repo_root()
    summary_csv = root / "data" / "phase2_interference" / "summary.csv"

    row = dataset_writer.SummaryRow(
        scenario_id=cfg.scenario_id,
        time_point=cfg.time_point,
        distance_m=cfg.distance_m,
        snr_db=snr_db,
        sinr_db=sinr_db,
        throughput_mbps=throughput_mbps,
        bler=bler,
        sionna_npz_path=str(artifacts.sionna_npz.relative_to(root)),
        matlab_h5_cubb_path=str(artifacts.matlab_h5_for_cubb.relative_to(root)),
        matlab_h5_ru_path=str(artifacts.matlab_h5_for_ru.relative_to(root)),
        throughput_log_path=str(artifacts.throughput_log.relative_to(root)),
        notes=f"role={current_role().value}",
    )
    dataset_writer.append_summary_row(summary_csv, row)

    return LinkResult(
        scenario_id=cfg.scenario_id,
        time_point=cfg.time_point,
        snr_db=snr_db,
        sinr_db=sinr_db,
        throughput_mbps=throughput_mbps,
        bler=bler,
        notes=row.notes,
    )
