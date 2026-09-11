"""
SatQuery AI — Task-Aware NLP Planner, Input Validator, and End-to-End Tests.

Comprehensive tests for:
- Task-aware NLP Analysis Planner
- Input requirement validation (task-specific error messages)
- GEE queries executing without uploaded images
- Disambiguation of GEE vs Change vs Grounding vs Fusion vs Multi-model
- End-to-end POST /api/query requests across all task modes
"""

from __future__ import annotations

import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.agent.planner import build_analysis_plan, validate_plan_inputs, AnalysisPlan

client = TestClient(app)


def _create_test_image_bytes(color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. GEE Matrix with NO Uploaded Image (End-to-End via POST /api/query)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def mock_ee():
    with patch("app.services.earth_engine.ee") as mock:
        yield mock


def test_gee_sentinel2_no_image_e2e(mock_ee):
    """1. Find a Sentinel-2 image at lat/lon between dates with cloud cover."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    img.get.side_effect = lambda key: MagicMock(getInfo=lambda: {
        "PRODUCT_ID": "S2_MUMBAI_2024",
        "CLOUDY_PIXEL_PERCENTAGE": 12.4,
        "system:time_start": 1600000000000,
    }.get(key))
    img.getThumbURL.return_value = "http://example.com/thumb.png"
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-15"

    response = client.post(
        "/api/query",
        data={
            "query": "Find a Sentinel-2 image at latitude 19.076 and longitude 72.8777 between June 1 and June 30 2024 with less than 20 percent cloud cover."
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "S2_MUMBAI_2024" in data["answer"] or "Sentinel-2" in data["answer"]
    assert "google_earth_engine" in data["execution_summary"]["models_used"]


def test_gee_sentinel1_no_image_e2e(mock_ee):
    """2. Find Sentinel-1 imagery at lat/lon using VV polarization."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    img.get.side_effect = lambda key: MagicMock(getInfo=lambda: {
        "system:index": "S1_VV_2024",
        "system:time_start": 1600000000000,
    }.get(key))
    img.getThumbURL.return_value = "http://example.com/thumb_s1.png"
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-10"

    response = client.post(
        "/api/query",
        data={
            "query": "Find Sentinel-1 imagery at latitude 19.076 and longitude 72.8777 using VV polarization."
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "google_earth_engine" in data["execution_summary"]["models_used"]


def test_gee_ndvi_no_image_e2e(mock_ee):
    """3. Calculate NDVI at latitude and longitude."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    ndvi = MagicMock()
    ndvi.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 0.68
    img.normalizedDifference.return_value = ndvi
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-15"

    response = client.post(
        "/api/query",
        data={"query": "Calculate NDVI at latitude 19.076 and longitude 72.8777."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "0.6800" in data["answer"]


def test_gee_ndwi_no_image_e2e(mock_ee):
    """4. Calculate NDWI at latitude and longitude."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    ndwi = MagicMock()
    ndwi.sample.return_value.first.return_value.get.return_value.getInfo.return_value = -0.05
    img.normalizedDifference.return_value = ndwi
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-15"

    response = client.post(
        "/api/query",
        data={"query": "Calculate NDWI at latitude 19.076 and longitude 72.8777."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "-0.0500" in data["answer"]


def test_gee_ndbi_no_image_e2e(mock_ee):
    """5. Calculate NDBI at latitude and longitude."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    ndbi = MagicMock()
    ndbi.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 0.14
    img.normalizedDifference.return_value = ndbi
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-06-15"

    response = client.post(
        "/api/query",
        data={"query": "Calculate NDBI at latitude 19.076 and longitude 72.8777."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "0.1400" in data["answer"]


def test_gee_elevation_no_image_e2e(mock_ee):
    """6. What is the elevation at latitude 19.076 and longitude 72.8777?"""
    mock_image = MagicMock()
    mock_image.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 42.5
    mock_ee.Image.return_value = mock_image

    response = client.post(
        "/api/query",
        data={"query": "What is the elevation at latitude 19.076 and longitude 72.8777?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"
    assert "42.50" in data["answer"]


def test_gee_landsat_no_image_e2e(mock_ee):
    """7. Find a Landsat image at latitude 19.076 and longitude 72.8777 during 2024."""
    coll = MagicMock()
    coll.filterBounds.return_value = coll
    coll.filterDate.return_value = coll
    coll.filter.return_value = coll
    coll.sort.return_value = coll
    coll.size.return_value.getInfo.return_value = 1

    img = MagicMock()
    img.get.side_effect = lambda key: MagicMock(getInfo=lambda: {
        "LANDSAT_PRODUCT_ID": "LC09_2024",
        "CLOUD_COVER": 5.0,
        "system:time_start": 1600000000000,
    }.get(key))
    img.select.return_value.multiply.return_value.add.return_value.getThumbURL.return_value = "http://example.com/thumb_landsat.png"
    coll.first.return_value = img
    mock_ee.ImageCollection.return_value = coll
    mock_ee.Date.return_value.format.return_value.getInfo.return_value = "2024-05-10"

    response = client.post(
        "/api/query",
        data={"query": "Find a Landsat image at latitude 19.076 and longitude 72.8777 during 2024."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "gee"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Image-Based Tasks (End-to-End via POST /api/query)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_vqa_with_image():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "What is present in this image?"},
    )
    assert response.status_code == 200
    assert response.json()["task_detected"] in ("vqa", "captioning")


def test_caption_with_image():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Describe this satellite image."},
    )
    assert response.status_code == 200
    assert response.json()["task_detected"] == "captioning"


def test_grounding_with_image():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Where are the buildings?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "grounding"
    assert data["visual_evidence"]["type"] in ("bbox", "mask")


def test_change_with_two_images():
    img1 = _create_test_image_bytes((100, 150, 100))
    img2 = _create_test_image_bytes((150, 100, 100))
    response = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2), "image/png")),
        ],
        data={
            "query": "What changed between these two satellite images?",
            "metadata": json.dumps({"dates": ["2022-01-01", "2025-01-01"]}),
        },
    )
    assert response.status_code == 200
    assert response.json()["task_detected"] == "change_vqa"


def test_fusion_with_optical_and_sar():
    opt = _create_test_image_bytes((120, 200, 100))
    sar = _create_test_image_bytes((80, 80, 80))
    response = client.post(
        "/api/query",
        files=[
            ("images", ("opt.png", io.BytesIO(opt), "image/png")),
            ("images", ("sar.png", io.BytesIO(sar), "image/png")),
        ],
        data={
            "query": "Does this area contain built-up regions according to both optical and SAR imagery?",
            "metadata": json.dumps({"modalities": ["optical", "sar"]}),
        },
    )
    assert response.status_code == 200
    assert response.json()["task_detected"] == "fusion"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Negative Cases & Task-Specific Error Messages
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_error_gee_missing_location():
    response = client.post(
        "/api/query",
        data={"query": "Calculate NDVI for this location."},
    )
    assert response.status_code == 400
    assert "Latitude and longitude are required" in response.json()["detail"]


def test_error_vqa_no_image():
    response = client.post(
        "/api/query",
        data={"query": "What is present in this image?"},
    )
    assert response.status_code == 400
    assert "requires one satellite image" in response.json()["detail"].lower()


def test_error_grounding_no_image():
    response = client.post(
        "/api/query",
        data={"query": "Where are the buildings?"},
    )
    assert response.status_code == 400
    assert "grounding requires" in response.json()["detail"].lower()


def test_error_change_one_image():
    """Single image + change query → NEEDS_CLARIFICATION (HTTP 200, valid=False)."""
    img1 = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("t1.png", io.BytesIO(img1), "image/png"))],
        data={
            "query": "What changed between these two satellite images?",
            "metadata": json.dumps({"dates": ["2022-01-01", "2025-01-01"]}),
        },
    )
    # Controller returns a structured needs_clarification response (HTTP 200)
    # rather than raising HTTP 400, so the endpoint returns 200.
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert data["task_detected"] == "change_analysis"
    assert "required" in (data.get("reason") or "").lower()


def test_error_fusion_missing_modalities():
    """Single optical image + optical-SAR fusion query → NEEDS_CLARIFICATION (HTTP 200)."""
    img1 = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("t1.png", io.BytesIO(img1), "image/png"))],
        data={
            "query": "Does this area contain built-up regions according to both optical and SAR imagery?",
            "metadata": json.dumps({"modalities": ["optical"]}),
        },
    )
    # Controller returns needs_clarification (HTTP 200, valid=False) instead of HTTP 400.
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert data.get("reason") is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Ambiguous Query Disambiguation Unit Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_disambiguity_gee_date_range_not_change():
    """'between June 1 and June 30' does NOT imply change detection."""
    plan = build_analysis_plan(
        "Find a Sentinel-2 image between June 1 and June 30 2024 at lat 19.076 lon 72.8777."
    )
    assert plan.primary_task == "gee"
    assert plan.start_date == "2024-06-01"
    assert plan.end_date == "2024-06-30"


def test_disambiguity_sentinel1_not_fusion():
    """'Find Sentinel-1 imagery' does NOT imply Fusion."""
    plan = build_analysis_plan(
        "Find Sentinel-1 imagery from June 1 to June 30 2024 at lat 19.076 lon 72.8777."
    )
    assert plan.primary_task == "gee"


def test_disambiguity_find_dataset_not_grounding():
    """'Find a Sentinel-2 image' does NOT imply Grounding."""
    plan = build_analysis_plan(
        "Find a Sentinel-2 image at lat 19.076 lon 72.8777."
    )
    assert plan.primary_task == "gee"


def test_disambiguity_what_changed():
    """'What changed between the June 1 and June 30 images?' implies Change detection."""
    plan = build_analysis_plan(
        "What changed between the June 1 and June 30 images?",
        image_count=2,
    )
    assert plan.primary_task in ("change_vqa", "change")


def test_disambiguity_find_buildings():
    """'Find the buildings in this image' implies Grounding."""
    plan = build_analysis_plan(
        "Find the buildings in this image.",
        image_count=1,
    )
    assert plan.primary_task == "grounding"


def test_disambiguity_compare_optical_and_sar():
    """'Compare optical and SAR imagery.' implies Fusion."""
    plan = build_analysis_plan(
        "Compare optical and SAR imagery.",
        image_count=2,
    )
    assert plan.primary_task == "fusion"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. User Reported Verification Cases (TEST 1 - TEST 6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_user_verification_test_1():
    """TEST 1: 2 images, 'Have the areas of buildings changed?' -> change_vqa, bi_temporal_change, min_images=2."""
    from app.agent.router import route_request
    plan = build_analysis_plan("Have the areas of buildings changed?", image_count=2)
    assert plan.primary_task == "change_vqa"
    assert plan.operation == "bi_temporal_change"
    assert plan.min_images == 2

    task, route = route_request("Have the areas of buildings changed?", image_count=2)
    assert task == "change_vqa"
    assert route == "bi_temporal_change_analysis"


def test_user_verification_test_2():
    """TEST 2: 2 images, 'What changed between the two images?' -> change_vqa."""
    from app.agent.router import route_request
    plan = build_analysis_plan("What changed between the two images?", image_count=2)
    assert plan.primary_task == "change_vqa"
    task, route = route_request("What changed between the two images?", image_count=2)
    assert task == "change_vqa"


def test_user_verification_test_3():
    """TEST 3: 2 images, 'Have the buildings changed between the two images?' -> change_vqa."""
    from app.agent.router import route_request
    plan = build_analysis_plan("Have the buildings changed between the two images?", image_count=2)
    assert plan.primary_task == "change_vqa"
    task, route = route_request("Have the buildings changed between the two images?", image_count=2)
    assert task == "change_vqa"


def test_user_verification_test_4():
    """TEST 4: 1 image, 'What types of land cover are visible in this satellite image?' -> vqa."""
    from app.agent.router import route_request
    plan = build_analysis_plan("What types of land cover are visible in this satellite image?", image_count=1)
    assert plan.primary_task == "vqa"
    task, route = route_request("What types of land cover are visible in this satellite image?", image_count=1)
    assert task == "vqa"


def test_user_verification_test_5():
    """TEST 5: 1 image, 'Locate the buildings in the image.' -> grounding."""
    from app.agent.router import route_request
    plan = build_analysis_plan("Locate the buildings in the image.", image_count=1)
    assert plan.primary_task == "grounding"
    task, route = route_request("Locate the buildings in the image.", image_count=1)
    assert task == "grounding"


def test_user_verification_test_6():
    """TEST 6: 2 images, Optical + SAR query -> fusion."""
    from app.agent.router import route_request
    plan = build_analysis_plan(
        "Does this area contain built-up regions according to both optical and SAR imagery?",
        image_count=2,
        metadata={"modalities": ["optical", "sar"]}
    )
    assert plan.primary_task == "fusion"
    task, route = route_request(
        "Does this area contain built-up regions according to both optical and SAR imagery?",
        image_count=2,
        modalities=["optical", "sar"]
    )
    assert task == "fusion"
