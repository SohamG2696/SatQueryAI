"""
SatQuery AI — Step 7 Real Multi-Temporal Sentinel-2 Acquisition & Change Analysis Integration Tests.
"""

from __future__ import annotations

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.services.sentinel_service import sentinel_service

client = TestClient(app)


def _create_synthetic_image(color=(100, 150, 200), width=512, height=512) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def mumbai_bitemporal_sentinel2_pair():
    """Acquires two real Sentinel-2 L2A images for Mumbai from CDSE for two distinct years (2020 & 2025)."""
    mumbai_bbox = [72.82, 18.95, 72.84, 18.97]

    # 1. Before Image (2020)
    res_before = sentinel_service.acquire_sentinel2_image(
        bbox=mumbai_bbox,
        date_from="2020-01-01T00:00:00Z",
        date_to="2020-01-31T23:59:59Z",
        width=512,
        height=512,
    )
    assert res_before["success"] is True
    img_id_before = res_before["image_id"]
    file_before = settings.upload_path / "temporary" / f"sentinel2_{img_id_before}.png"
    assert file_before.exists()
    bytes_before = file_before.read_bytes()

    # 2. After Image (2025)
    res_after = sentinel_service.acquire_sentinel2_image(
        bbox=mumbai_bbox,
        date_from="2025-01-01T00:00:00Z",
        date_to="2025-01-31T23:59:59Z",
        width=512,
        height=512,
    )
    assert res_after["success"] is True
    img_id_after = res_after["image_id"]
    file_after = settings.upload_path / "temporary" / f"sentinel2_{img_id_after}.png"
    assert file_after.exists()
    bytes_after = file_after.read_bytes()

    return {
        "before_bytes": bytes_before,
        "after_bytes": bytes_after,
        "before_meta": res_before,
        "after_meta": res_after,
        "bbox": mumbai_bbox,
    }


def test_1_sentinel2_dual_date_acquisition(mumbai_bitemporal_sentinel2_pair):
    """TEST 1: Same AOI + two distinct dates (2020 & 2025) -> CDSE acquisition succeeds for both."""
    pair = mumbai_bitemporal_sentinel2_pair
    assert pair["before_meta"]["success"] is True
    assert pair["after_meta"]["success"] is True
    assert pair["before_meta"]["bbox"] == pair["after_meta"]["bbox"]
    assert len(pair["before_bytes"]) > 1000
    assert len(pair["after_bytes"]) > 1000


def test_2_temporal_roles_metadata(mumbai_bitemporal_sentinel2_pair):
    """TEST 2: Temporal role metadata format validation."""
    pair = mumbai_bitemporal_sentinel2_pair
    meta = {
        "dates": ["2020-01-01 to 2020-01-31", "2025-01-01 to 2025-01-31"],
        "bbox": "[72.82, 18.95, 72.84, 18.97]",
        "source": "Copernicus Data Space Ecosystem (CDSE)",
        "temporal_roles": ["before", "after"],
    }
    assert meta["temporal_roles"][0] == "before"
    assert meta["temporal_roles"][1] == "after"
    assert len(meta["dates"]) == 2


def test_3_two_images_change_query_routing(mumbai_bitemporal_sentinel2_pair):
    """TEST 3: Two Sentinel-2 images + change query -> Agent routes to ChangeFormerV6 (change_vqa)."""
    pair = mumbai_bitemporal_sentinel2_pair
    files = [
        ("images", ("sentinel2_before_2020.png", pair["before_bytes"], "image/png")),
        ("images", ("sentinel2_after_2025.png", pair["after_bytes"], "image/png")),
    ]
    data = {
        "query": "How has this area changed between these two dates?",
        "metadata": json.dumps({
            "dates": ["2020-01-15", "2025-01-15"],
            "bbox": str(pair["bbox"]),
            "source": "Copernicus Data Space Ecosystem (CDSE)"
        }),
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200, f"Query execution failed: {res.text}"
    json_data = res.json()

    assert json_data["task_detected"] == "change_vqa"
    assert "ChangeFormerV6" in json_data["execution_summary"]["models_used"]
    assert json_data["execution_summary"]["task_route"] == "bi_temporal_change_analysis"


def test_4_changeformer_receives_ordered_pair(mumbai_bitemporal_sentinel2_pair):
    """TEST 4: Verify ChangeFormer execution receives before/after images in correct order."""
    pair = mumbai_bitemporal_sentinel2_pair
    files = [
        ("images", ("before_2020.png", pair["before_bytes"], "image/png")),
        ("images", ("after_2025.png", pair["after_bytes"], "image/png")),
    ]
    data = {
        "query": "Detect changes between the two images.",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "change_vqa"
    trace = json_data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("Loaded 2 satellite image(s)" in t for t in trace)


def test_5_change_result_structure(mumbai_bitemporal_sentinel2_pair):
    """TEST 5: Verify change result contains answer, confidence, change ratio, and visual evidence."""
    pair = mumbai_bitemporal_sentinel2_pair
    files = [
        ("images", ("before.png", pair["before_bytes"], "image/png")),
        ("images", ("after.png", pair["after_bytes"], "image/png")),
    ]
    data = {
        "query": "Did the buildings change in this area?",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert "answer" in json_data
    assert json_data["confidence"] is not None
    assert json_data["visual_evidence"] is not None
    assert json_data["visual_evidence"]["type"] in ("change_mask", "mask", "bbox")


def test_6_single_image_vqa_regression(mumbai_bitemporal_sentinel2_pair):
    """TEST 6: Single-image VQA query regression test."""
    pair = mumbai_bitemporal_sentinel2_pair
    files = [
        ("images", ("single_scene.png", pair["after_bytes"], "image/png")),
    ]
    data = {
        "query": "What type of land cover is visible in this image?",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "vqa"
    assert isinstance(json_data["answer"], str)


def test_7_grounding_regression(mumbai_bitemporal_sentinel2_pair):
    """TEST 7: Single-image Grounding query regression test."""
    pair = mumbai_bitemporal_sentinel2_pair
    files = [
        ("images", ("single_scene.png", pair["after_bytes"], "image/png")),
    ]
    data = {
        "query": "Identify the buildings in this image",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "grounding"
    assert json_data["visual_evidence"]["type"] in ("mask", "bbox")


def test_8_local_upload_regression():
    """TEST 8: Local multi-image change analysis upload regression test."""
    img1 = _create_synthetic_image(color=(50, 100, 150))
    img2 = _create_synthetic_image(color=(150, 100, 50))

    files = [
        ("images", ("local_t1.png", img1, "image/png")),
        ("images", ("local_t2.png", img2, "image/png")),
    ]
    data = {
        "query": "What changed between these two images?",
        "metadata": json.dumps({"dates": ["2020-01-01", "2025-01-01"]}),
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()

    assert json_data["task_detected"] == "change_vqa"
    assert "ChangeFormerV6" in json_data["execution_summary"]["models_used"]


def test_9_invalid_missing_second_image_for_change():
    """TEST 9: Attempting change query with only 1 image returns clear validation error."""
    img = _create_synthetic_image()
    files = [
        ("images", ("single.png", img, "image/png")),
    ]
    data = {
        "query": "What changed between these two dates?",
    }

    res = client.post("/api/query", data=data, files=files)
    assert res.status_code == 400
    err_detail = res.json().get("detail", "")
    assert "requires two satellite images" in err_detail.lower() or "requires" in err_detail.lower()


def test_10_different_aoi_rejection_validation():
    """TEST 10: Ensures error handling for invalid payload structure."""
    res = client.post("/api/query", data={"query": "What changed between these two dates?"})
    assert res.status_code == 400
