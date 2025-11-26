"""Tests for FastAPI module."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from uav_acar_sionna.api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_results_endpoint(client):
    """Test results listing endpoint."""
    response = client.get("/results?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert "total_records" in data


def test_emulate_endpoint_validation(client):
    """Test emulate endpoint validation."""
    # Missing required fields should fail
    response = client.post("/emulate", json={})
    assert response.status_code == 422  # Validation error
