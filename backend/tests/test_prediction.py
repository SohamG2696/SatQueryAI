"""
Tests for POST /api/predict-future Land Cover Future Prediction Endpoint.
"""

from pathlib import Path
import sys

# Ensure backend directory is on sys.path
THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_future_valid_region_01():
    """Verify valid request for region_01 in 2027 returns 200 and matches schema."""
    payload = {
        "region_id": "region_01",
        "target_year": 2027,
    }
    response = client.post("/api/predict-future", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["region_id"] == "region_01"
    assert data["forecast_year"] == 2027
    assert "built_up_pct" in data
    assert "vegetation_pct" in data
    assert "water_pct" in data
    assert data["method"] == "linear_regression"
    assert data["confidence"] in ["high", "moderate", "low"]
    assert "r2_scores" in data
    assert "annual_slopes" in data
    assert "disclaimer" in data

    # Check landcover percentages sum to ~100%
    total = data["built_up_pct"] + data["vegetation_pct"] + data["water_pct"]
    assert abs(total - 100.0) < 0.1, f"Percentages sum to {total}, expected ~100.0"


def test_predict_future_valid_region_02_saturated():
    """Verify valid request for region_02 (Mumbai) returns 200 with low_r2 interpretation notes."""
    payload = {
        "region_id": "region_02",
        "target_year": 2027,
    }
    response = client.post("/api/predict-future", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["region_id"] == "region_02"
    assert data["forecast_year"] == 2027
    assert isinstance(data["interpretation_notes"], list)


def test_predict_future_invalid_region():
    """Verify requesting an unknown region returns 400 Bad Request."""
    payload = {
        "region_id": "region_99_invalid",
        "target_year": 2027,
    }
    response = client.post("/api/predict-future", json=payload)
    assert response.status_code == 400
    assert "Unknown region_id" in response.json()["detail"]


def test_predict_future_past_year():
    """Verify requesting a year <= 2025 returns 400 Bad Request."""
    payload = {
        "region_id": "region_01",
        "target_year": 2024,
    }
    response = client.post("/api/predict-future", json=payload)
    assert response.status_code == 400
    assert "must be in the future" in response.json()["detail"]


def test_predict_future_extreme_year():
    """Verify requesting a year > 2075 returns 400 Bad Request."""
    payload = {
        "region_id": "region_01",
        "target_year": 2090,
    }
    response = client.post("/api/predict-future", json=payload)
    assert response.status_code == 400
    assert "exceeds maximum supported forecast horizon" in response.json()["detail"]
