# UAV-ACAR-Sionna REST API

FastAPI REST API for running UAV link emulations with Sionna, cuBB, and RU emulator.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or with uvicorn explicitly
pip install fastapi uvicorn[standard]
```

### Running the Server

#### Option 1: Using the script (recommended)

```bash
# Production mode
./scripts/run_api.sh

# Development mode with auto-reload
./scripts/run_api.sh --reload
```

#### Option 2: Using the CLI entry point

```bash
run_api
```

#### Option 3: Using uvicorn directly

```bash
uvicorn uav_acar_sionna.api.main:app --host 0.0.0.0 --port 8000
```

The server will start on http://0.0.0.0:8000

### API Documentation

Once the server is running, access the interactive API docs at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.2.0"
}
```

### Run Emulation

**POST** `/emulate`

Run a single UAV link emulation with the specified parameters.

**Request Body:**
```json
{
  "scenario_id": "base",
  "time_point": 0.0,
  "distance_m": 500.0
}
```

**Response:**
```json
{
  "scenario_id": "base",
  "time_point": 0.0,
  "snr_db": 25.3,
  "sinr_db": 24.8,
  "throughput_mbps": 150.2,
  "bler": 0.001,
  "notes": "role=sionna_host"
}
```

**Pipeline Steps:**
1. Generate Sionna channel snapshots (.npz)
2. Convert to MATLAB HDF5 test vectors
3. Run cuBB + RU emulator
4. Collect throughput and BLER metrics
5. Append results to summary.csv

### Get Results

**GET** `/results?limit=50`

Retrieve recent emulation results from summary.csv.

**Query Parameters:**
- `limit` (optional): Maximum number of records to return (default: 50)

**Response:**
```json
{
  "total_records": 100,
  "records": [
    {
      "scenario_id": "base",
      "time_point": 0.0,
      "distance_m": 500.0,
      "snr_db": 25.3,
      "sinr_db": 24.8,
      "throughput_mbps": 150.2,
      "bler": 0.001,
      "notes": "role=sionna_host"
    }
  ]
}
```

## Configuration

The API uses environment variables for configuration:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)

Example:
```bash
HOST=127.0.0.1 PORT=9000 ./scripts/run_api.sh
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `404 Not Found`: Resource not found (e.g., summary.csv doesn't exist)
- `500 Internal Server Error`: Emulation failed or server error

Error responses include detailed messages:
```json
{
  "detail": "Missing required dependency (Sionna/TensorFlow/MATLAB): ..."
}
```

## Logging

Logs are written to:
- Console (stdout/stderr)
- `logs/api/api_YYYYMMDD_HHMMSS.log`

Log format includes timestamps, logger names, and log levels.

## Testing

### Validate API Structure

```bash
python3 scripts/validate_api.py
```

### Test Endpoints with curl

```bash
# Health check
curl http://localhost:8000/health

# Run emulation
curl -X POST http://localhost:8000/emulate \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "base",
    "time_point": 0.0,
    "distance_m": 500.0
  }'

# Get results
curl http://localhost:8000/results?limit=10
```

### Test with Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Run emulation
response = requests.post(
    "http://localhost:8000/emulate",
    json={
        "scenario_id": "base",
        "time_point": 0.0,
        "distance_m": 500.0
    }
)
print(response.json())

# Get results
response = requests.get("http://localhost:8000/results?limit=10")
print(response.json())
```

## Architecture

The API follows a simple, focused architecture:

```
src/uav_acar_sionna/api/
├── __init__.py          # Package exports
├── main.py              # FastAPI app, endpoints, server entry point
└── schemas.py           # Pydantic request/response models
```

### Design Principles

- **YAGNI**: Only implements what's needed (no auth, no complex middleware)
- **Simple**: Direct mapping from API to orchestrator functions
- **Type-safe**: Full type hints with Pydantic V2
- **Observable**: Comprehensive logging for debugging
- **Self-documenting**: Auto-generated OpenAPI docs

## Integration

The API is a thin wrapper around the existing orchestrator:

```
FastAPI Request
    ↓
EmulationRequest (Pydantic schema)
    ↓
EmulationConfig (orchestrator dataclass)
    ↓
run_emulation_once() → LinkResult
    ↓
EmulationResponse (Pydantic schema)
    ↓
JSON Response
```

No business logic is duplicated—the API delegates to `uav_acar_sionna.orchestrator.run_emulation`.
