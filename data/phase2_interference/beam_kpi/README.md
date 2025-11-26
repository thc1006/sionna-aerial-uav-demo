# Beam KPI Directory

This directory contains beam-specific Key Performance Indicator (KPI) measurements.

## File Format

Each CSV file in this directory follows the schema defined in `dataset_structures.py`:

```csv
beam_id,azimuth_deg,elevation_deg,gain_db,sinr_db
beam_001,45.0,15.0,8.5,12.3
beam_002,90.0,20.0,9.2,14.7
```

## Column Descriptions

- **beam_id**: Unique identifier for the beam configuration
- **azimuth_deg**: Beam azimuth angle in degrees (0-360)
- **elevation_deg**: Beam elevation angle in degrees (-90 to 90)
- **gain_db**: Antenna gain in dB
- **sinr_db**: Measured Signal-to-Interference-plus-Noise Ratio in dB

## Usage

Use the helper functions in `uav_acar_sionna.orchestrator.dataset_structures` to read/write these files:

```python
from pathlib import Path
from uav_acar_sionna.orchestrator.dataset_structures import (
    BeamKpiRecord,
    read_beam_kpi_csv,
    write_beam_kpi_csv,
)

# Write beam KPI data
records = [
    BeamKpiRecord(
        beam_id="beam_001",
        azimuth_deg=45.0,
        elevation_deg=15.0,
        gain_db=8.5,
        sinr_db=12.3,
    ),
]
write_beam_kpi_csv(records, Path("beam_kpi/scenario_001.csv"))

# Read beam KPI data
records = read_beam_kpi_csv(Path("beam_kpi/scenario_001.csv"))
```
