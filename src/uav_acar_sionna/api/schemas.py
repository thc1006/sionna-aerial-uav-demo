"""Pydantic schemas for FastAPI request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmulationRequest(BaseModel):
    """Request schema for running a single emulation."""

    scenario_id: str = Field(
        ...,
        description="Unique identifier for the scenario (e.g., 'base', 'int1', 'int2')",
        examples=["base"],
    )
    time_point: float = Field(
        ...,
        description="Time point in seconds for the scenario",
        ge=0.0,
        examples=[0.0],
    )
    distance_m: float = Field(
        default=500.0,
        description="Distance between gNB and UAV in meters",
        ge=0.0,
        examples=[500.0],
    )

    model_config = {"json_schema_extra": {"examples": [{"scenario_id": "base", "time_point": 0.0, "distance_m": 500.0}]}}


class EmulationResponse(BaseModel):
    """Response schema containing emulation results."""

    scenario_id: str = Field(..., description="Scenario identifier from the request")
    time_point: float = Field(..., description="Time point from the request")
    snr_db: float = Field(..., description="Signal-to-Noise Ratio in dB")
    sinr_db: float = Field(..., description="Signal-to-Interference-plus-Noise Ratio in dB")
    throughput_mbps: float = Field(..., description="Achieved throughput in Mbps")
    bler: float = Field(..., description="Block Error Rate (0.0 to 1.0)")
    notes: str | None = Field(None, description="Additional notes or metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scenario_id": "base",
                    "time_point": 0.0,
                    "snr_db": 25.3,
                    "sinr_db": 24.8,
                    "throughput_mbps": 150.2,
                    "bler": 0.001,
                    "notes": "role=sionna_host",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")


class SummaryRecord(BaseModel):
    """Individual record from summary.csv."""

    scenario_id: str
    time_point: float
    distance_m: float
    snr_db: float
    sinr_db: float
    throughput_mbps: float
    bler: float
    notes: str | None = None


class SummaryResponse(BaseModel):
    """Response containing recent emulation results from summary.csv."""

    total_records: int = Field(..., description="Total number of records in summary")
    records: list[SummaryRecord] = Field(..., description="List of emulation records")
