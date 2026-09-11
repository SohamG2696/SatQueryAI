"""
SatQuery AI — End-to-End Integration Test for Real Sentinel-2 Image Query Pipeline.
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


@pytest.fixture(scope="module")
def real_sentinel2_image_bytes():
    """Acquires a real Sentinel-2 L2A image bytes from CDSE Processing API."""
    mumbai_bbox = [72.82, 18.95, 72.84, 18.97]
    result = sentinel_service.acquire_sentinel2_image(
        bbox=mumbai_bbox,
        date_from="2025-01-01T00:00:00Z",
        date_to="2025-01-31T23:59:59Z",
        width=512,
        height=512,
    )
    assert result["success"] is True
    image_id = result["image_id"]
    temp_file = settings.upload_path / "temporary" / f"sentinel2_{image_id}.png"
    assert temp_file.exists()
    return temp_file.read_bytes()


def test_sentinel2_grounding_query(real_sentinel2_image_bytes):
    """TEST 1: Real Sentinel-2 image + Grounding Query ('Identify the buildings in this image')."""
    files = {
        "images": ("sentinel2_mumbai.png", real_sentinel2_image_bytes, "image/png")
    }
    data = {
        "query": "Identify the buildings in this image",
        "metadata": '{"source": "Copernicus Data Space Ecosystem (CDSE)", "satellite": "Sentinel-2"}',
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200, f"Query endpoint failed: {res.text}"
    json_data = res.json()

    assert json_data["task_detected"] == "grounding"
    assert "answer" in json_data
    assert json_data["visual_evidence"] is not None
    assert json_data["visual_evidence"]["type"] in ("mask", "bbox")
    assert len(json_data["visual_evidence"]["coordinates"]) == 4

    # Verify execution summary & parameters
    summary = json_data["execution_summary"]
    assert summary is not None
    assert "models_used" in summary
    assert summary["parameters"].get("source") == "Copernicus Data Space Ecosystem (CDSE)"


def test_sentinel2_vqa_counting_query(real_sentinel2_image_bytes):
    """TEST 2: Real Sentinel-2 image + Counting VQA Query ('How many buildings are visible?')."""
    files = {
        "images": ("sentinel2_mumbai.png", real_sentinel2_image_bytes, "image/png")
    }
    data = {
        "query": "How many buildings are visible?",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] in ("vqa", "counting_vqa")
    assert isinstance(json_data["answer"], str)
    assert len(json_data["answer"].strip()) > 0


def test_sentinel2_vqa_landcover_query(real_sentinel2_image_bytes):
    """TEST 3: Real Sentinel-2 image + Landcover VQA Query ('What type of land cover is visible in this image?')."""
    files = {
        "images": ("sentinel2_mumbai.png", real_sentinel2_image_bytes, "image/png")
    }
    data = {
        "query": "What type of land cover is visible in this image?",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "vqa"
    assert isinstance(json_data["answer"], str)
    assert len(json_data["answer"].strip()) > 0


def test_local_upload_regression():
    """TEST 4: Regression test ensuring local image uploads still route and work properly."""
    # Create dummy RGB image in memory
    img = Image.new("RGB", (256, 256), color=(70, 130, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    dummy_bytes = buf.read()

    files = {
        "images": ("local_sample.png", dummy_bytes, "image/png")
    }
    data = {
        "query": "highlight the forest region in this image",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "grounding"
    assert json_data["visual_evidence"]["type"] in ("mask", "bbox")
