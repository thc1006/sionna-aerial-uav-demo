"""FastAPI application for UAV-ACAR-Sionna emulation REST API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..orchestrator.run_emulation import EmulationConfig, run_emulation_once, get_repo_root
from .schemas import (
    EmulationRequest,
    EmulationResponse,
    HealthResponse,
    SummaryResponse,
    SummaryRecord,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="UAV-ACAR-Sionna Emulation API",
    description="REST API for running UAV link emulations with Sionna, cuBB, and RU emulator",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint to verify the API is running."""
    return HealthResponse(status="healthy", version="0.2.0")


@app.post("/emulate", response_model=EmulationResponse, tags=["Emulation"])
async def run_emulation(request: EmulationRequest) -> EmulationResponse:
    """Run a single UAV link emulation.

    This endpoint orchestrates the full emulation pipeline:
    1. Generate Sionna channel snapshots
    2. Convert to MATLAB HDF5 test vectors
    3. Run cuBB + RU emulator
    4. Collect throughput and BLER metrics
    5. Store results in summary.csv

    Args:
        request: Emulation configuration (scenario_id, time_point, distance_m)

    Returns:
        EmulationResponse containing SNR, SINR, throughput, BLER, and metadata

    Raises:
        HTTPException: If emulation fails due to missing dependencies or runtime errors
    """
    try:
        logger.info(
            f"Starting emulation for scenario={request.scenario_id}, "
            f"t={request.time_point}, d={request.distance_m}m"
        )

        # Convert Pydantic model to EmulationConfig
        config = EmulationConfig(
            scenario_id=request.scenario_id,
            time_point=request.time_point,
            distance_m=request.distance_m,
        )

        # Run the emulation pipeline
        result = run_emulation_once(config)

        logger.info(
            f"Emulation completed: SNR={result.snr_db:.2f}dB, "
            f"SINR={result.sinr_db:.2f}dB, "
            f"Throughput={result.throughput_mbps:.2f}Mbps"
        )

        # Convert LinkResult to EmulationResponse
        return EmulationResponse(
            scenario_id=result.scenario_id,
            time_point=result.time_point,
            snr_db=result.snr_db,
            sinr_db=result.sinr_db,
            throughput_mbps=result.throughput_mbps,
            bler=result.bler,
            notes=result.notes,
        )

    except FileNotFoundError as e:
        logger.error(f"File not found during emulation: {e}")
        raise HTTPException(status_code=500, detail=f"Missing required file: {e}")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing required dependency (Sionna/TensorFlow/MATLAB): {e}",
        )
    except Exception as e:
        logger.error(f"Emulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Emulation failed: {str(e)}")


@app.get("/results", response_model=SummaryResponse, tags=["Results"])
async def get_results(limit: int = 50) -> SummaryResponse:
    """Retrieve recent emulation results from summary.csv.

    Args:
        limit: Maximum number of records to return (default: 50)

    Returns:
        SummaryResponse containing list of recent emulation results

    Raises:
        HTTPException: If summary.csv does not exist or cannot be read
    """
    try:
        root = get_repo_root()
        summary_csv = root / "data" / "phase2_interference" / "summary.csv"

        if not summary_csv.exists():
            logger.warning(f"Summary file not found: {summary_csv}")
            return SummaryResponse(total_records=0, records=[])

        # Read the CSV file
        df = pd.read_csv(summary_csv)
        total_records = len(df)

        # Get the most recent records (last N rows)
        df_recent = df.tail(limit)

        # Convert to list of SummaryRecord
        records = []
        for _, row in df_recent.iterrows():
            records.append(
                SummaryRecord(
                    scenario_id=row["scenario_id"],
                    time_point=float(row["time_point"]),
                    distance_m=float(row["distance_m"]),
                    snr_db=float(row["snr_db"]),
                    sinr_db=float(row["sinr_db"]),
                    throughput_mbps=float(row["throughput_mbps"]),
                    bler=float(row["bler"]),
                    notes=row.get("notes"),
                )
            )

        logger.info(f"Retrieved {len(records)} records (total: {total_records})")
        return SummaryResponse(total_records=total_records, records=records)

    except FileNotFoundError:
        logger.error(f"Summary file not found: {summary_csv}")
        raise HTTPException(status_code=404, detail="Summary file not found")
    except Exception as e:
        logger.error(f"Failed to read results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read results: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


def run_server() -> None:
    """Entry point for running the API server via CLI.

    This function is registered as a console script in pyproject.toml
    and can be invoked with: run_api
    """
    import uvicorn

    uvicorn.run(
        "uav_acar_sionna.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
