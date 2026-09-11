"""
SatQuery AI — Focused Unit & Integration Tests for Sentinel-2 Satellite Acquisition.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.services.sentinel_service import sentinel_service

client = TestClient(app)


def test_sentinel_environment_config():
    """Test 1: Environment configuration loads correctly."""
    assert settings.sentinel_hub_base_url is not None
    assert "copernicus.eu" in settings.sentinel_hub_base_url
    assert settings.copernicus_token_url is not None
    assert "identity.dataspace.copernicus.eu" in settings.copernicus_token_url


def test_sentinel_service_token_acquisition():
    """Test 2: SentinelHubService OAuth token acquisition succeeds."""
    token = sentinel_service.get_access_token()
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 20


def test_sentinel_service_imagery_acquisition():
    """Test 3: SentinelHubService acquires real Sentinel-2 L2A imagery."""
    mumbai_bbox = [72.82, 18.95, 72.84, 18.97]
    result = sentinel_service.acquire_sentinel2_image(
        bbox=mumbai_bbox,
        date_from="2025-01-01T00:00:00Z",
        date_to="2025-01-31T23:59:59Z",
        width=512,
        height=512,
    )

    assert result["success"] is True
    assert result["source"] == "Copernicus Data Space"
    assert result["satellite"] == "Sentinel-2"
    assert result["product"] == "Sentinel-2 L2A"
    assert result["bbox"] == mumbai_bbox
    assert "image_url" in result
    assert result["width"] == 512
    assert result["height"] == 512
    assert result["size_bytes"] > 0
    # Ensure OAuth token is NEVER returned in metadata dictionary
    assert "token" not in result
    assert "access_token" not in result
    assert "client_secret" not in result


def test_sentinel_api_valid_bbox():
    """Test 4: POST /api/satellite/sentinel2 accepts valid bbox and returns 200 OK."""
    payload = {
        "bbox": [72.82, 18.95, 72.84, 18.97],
        "date_from": "2025-01-01T00:00:00Z",
        "date_to": "2025-01-31T23:59:59Z",
        "width": 512,
        "height": 512,
    }

    res = client.post("/api/satellite/sentinel2", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["source"] == "Copernicus Data Space"
    assert data["satellite"] == "Sentinel-2"
    assert data["product"] == "Sentinel-2 L2A"
    assert data["bbox"] == [72.82, 18.95, 72.84, 18.97]
    assert data["image_url"].startswith("/api/satellite/image/")

    # Ensure OAuth token is NEVER returned to frontend
    assert "token" not in data
    assert "access_token" not in data
    assert "client_secret" not in data

    # Test GET /api/satellite/image/{image_id}
    image_url = data["image_url"]
    img_res = client.get(image_url)
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/png"

    # Verify PNG image using PIL
    img_bytes = img_res.content
    with Image.open(io.BytesIO(img_bytes)) as img:
        img.verify()
    with Image.open(io.BytesIO(img_bytes)) as img:
        width, height = img.size
        assert width == 512
        assert height == 512
        assert img.format == "PNG"


def test_sentinel_api_invalid_bbox():
    """Test 5: POST /api/satellite/sentinel2 rejects invalid bbox."""
    # Bbox with only 3 coordinates (invalid length)
    bad_payload = {
        "bbox": [72.82, 18.95, 72.84],
        "date_from": "2025-01-01T00:00:00Z",
        "date_to": "2025-01-31T23:59:59Z",
    }
    res = client.post("/api/satellite/sentinel2", json=bad_payload)
    assert res.status_code in (400, 422)

    # Bbox with min_lon > max_lon
    bad_bounds_payload = {
        "bbox": [75.0, 18.95, 72.0, 18.97],
        "date_from": "2025-01-01T00:00:00Z",
        "date_to": "2025-01-31T23:59:59Z",
    }
    res2 = client.post("/api/satellite/sentinel2", json=bad_bounds_payload)
    assert res2.status_code == 400
