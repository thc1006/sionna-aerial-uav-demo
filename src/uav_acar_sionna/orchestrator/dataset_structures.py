"""Data structures and CSV helpers for interferers and beam KPI datasets.

This module defines Pydantic models for structured datasets used in Phase 2:
- Interferer configurations (position, power, frequency)
- Beam KPI metrics (azimuth, elevation, gain, SINR)

It also provides helper functions to write and read these datasets as CSV files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class InterfererRecord(BaseModel):
    """Single interferer configuration record.

    Attributes
    ----------
    interferer_id : str
        Unique identifier for this interferer (e.g., "int_001").
    position_x : float
        X coordinate in meters.
    position_y : float
        Y coordinate in meters.
    position_z : float
        Z coordinate (height) in meters.
    power_dbm : float
        Transmit power in dBm.
    frequency_ghz : float
        Operating frequency in GHz.
    """

    interferer_id: str = Field(..., description="Unique interferer identifier")
    position_x: float = Field(..., description="X coordinate in meters")
    position_y: float = Field(..., description="Y coordinate in meters")
    position_z: float = Field(..., description="Z coordinate (height) in meters")
    power_dbm: float = Field(..., description="Transmit power in dBm")
    frequency_ghz: float = Field(..., description="Operating frequency in GHz")


class BeamKpiRecord(BaseModel):
    """Single beam KPI measurement record.

    Attributes
    ----------
    beam_id : str
        Unique identifier for this beam configuration.
    azimuth_deg : float
        Beam azimuth angle in degrees.
    elevation_deg : float
        Beam elevation angle in degrees.
    gain_db : float
        Antenna gain in dB.
    sinr_db : float
        Measured SINR in dB.
    """

    beam_id: str = Field(..., description="Unique beam identifier")
    azimuth_deg: float = Field(..., description="Beam azimuth angle in degrees")
    elevation_deg: float = Field(..., description="Beam elevation angle in degrees")
    gain_db: float = Field(..., description="Antenna gain in dB")
    sinr_db: float = Field(..., description="Measured SINR in dB")


def write_interferers_csv(records: List[InterfererRecord], path: Path) -> None:
    """Write interferer records to CSV file.

    Parameters
    ----------
    records :
        List of InterfererRecord objects to write.
    path :
        Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "interferer_id",
        "position_x",
        "position_y",
        "position_z",
        "power_dbm",
        "frequency_ghz",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def read_interferers_csv(path: Path) -> List[InterfererRecord]:
    """Read interferer records from CSV file.

    Parameters
    ----------
    path :
        Input CSV file path.

    Returns
    -------
    records : List[InterfererRecord]
        List of parsed InterfererRecord objects.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            record = InterfererRecord(
                interferer_id=row["interferer_id"],
                position_x=float(row["position_x"]),
                position_y=float(row["position_y"]),
                position_z=float(row["position_z"]),
                power_dbm=float(row["power_dbm"]),
                frequency_ghz=float(row["frequency_ghz"]),
            )
            records.append(record)
    return records


def write_beam_kpi_csv(records: List[BeamKpiRecord], path: Path) -> None:
    """Write beam KPI records to CSV file.

    Parameters
    ----------
    records :
        List of BeamKpiRecord objects to write.
    path :
        Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["beam_id", "azimuth_deg", "elevation_deg", "gain_db", "sinr_db"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def read_beam_kpi_csv(path: Path) -> List[BeamKpiRecord]:
    """Read beam KPI records from CSV file.

    Parameters
    ----------
    path :
        Input CSV file path.

    Returns
    -------
    records : List[BeamKpiRecord]
        List of parsed BeamKpiRecord objects.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string values to appropriate types
            record = BeamKpiRecord(
                beam_id=row["beam_id"],
                azimuth_deg=float(row["azimuth_deg"]),
                elevation_deg=float(row["elevation_deg"]),
                gain_db=float(row["gain_db"]),
                sinr_db=float(row["sinr_db"]),
            )
            records.append(record)
    return records
