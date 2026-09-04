"""
End-to-End Tests for Main Query and History API Endpoints.
"""

import io
import json
from PIL import Image
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def _create_test_image_bytes(color=(100, 150, 200), size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_query_fusion_flow():
    """Verify end-to-end query execution on Optical-SAR pair."""
    opt_bytes = _create_test_image_bytes((120, 200, 100))
    sar_bytes = _create_test_image_bytes((80, 80, 80))

    meta_json = json.dumps({
        "modalities": ["optical", "sar"],
        "session_id": "test_session_fusion",
    })

    response = client.post(
        "/api/query",
        files=[
            ("images", ("optical.png", io.BytesIO(opt_bytes), "image/png")),
            ("images", ("sar.png", io.BytesIO(sar_bytes), "image/png")),
        ],
        data={
            "query": "Does the SAR image confirm the land-cover pattern in the optical image?",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "fusion"
    assert data["answer"] in ("YES", "NO")
    assert data["confidence"] is not None
    assert data["execution_summary"]["task_route"] == "two_image_cross_modal_fusion"
    assert "satquery-optical-sar-fusion-v1" in data["execution_summary"]["models_used"]


def test_query_grounding_flow():
    """Verify end-to-end query execution for region grounding."""
    img_bytes = _create_test_image_bytes((50, 120, 220))

    response = client.post(
        "/api/query",
        files=[
            ("images", ("scene.png", io.BytesIO(img_bytes), "image/png")),
        ],
        data={
            "query": "Locate the smallest contiguous forest region in this image.",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "grounding"
    assert data["visual_evidence"]["type"] == "bbox"
    assert len(data["visual_evidence"]["coordinates"]) == 4
    assert data["execution_summary"]["task_route"] == "single_image_region_grounding"


def test_query_change_vqa_flow():
    """Verify end-to-end query execution for change detection."""
    t1_bytes = _create_test_image_bytes((100, 200, 100))
    t2_bytes = _create_test_image_bytes((150, 100, 50))

    meta_json = json.dumps({
        "dates": ["2022-05-01", "2025-05-01"],
        "session_id": "test_session_change",
    })

    response = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(t1_bytes), "image/png")),
            ("images", ("t2.png", io.BytesIO(t2_bytes), "image/png")),
        ],
        data={
            "query": "Did vegetation decrease between 2022 and 2025?",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "change_vqa"
    assert data["answer"].upper().startswith(("YES", "NO", "[")) or data.get("raw_answer") in ("YES", "NO")
    assert data["execution_summary"]["task_route"] == "bi_temporal_change_analysis"


def test_query_vqa_flow():
    """Verify VQA flow with Person A VLM adapter."""
    img_bytes = _create_test_image_bytes()

    response = client.post(
        "/api/query",
        files=[
            ("images", ("scene.png", io.BytesIO(img_bytes), "image/png")),
        ],
        data={
            "query": "Does this scene contain agricultural land?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "vqa"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]


def test_query_vlm_are_buildings_visible_routing():
    """Verify routing and execution for 'Are buildings visible in this image?'."""
    img_bytes = _create_test_image_bytes()

    response = client.post(
        "/api/query",
        files=[
            ("images", ("scene.png", io.BytesIO(img_bytes), "image/png")),
        ],
        data={
            "query": "Are buildings visible in this image?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "vqa"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]


def test_query_captioning_flow():
    """Verify Captioning flow with Person A VLM adapter."""
    img_bytes = _create_test_image_bytes()

    response = client.post(
        "/api/query",
        files=[
            ("images", ("scene.png", io.BytesIO(img_bytes), "image/png")),
        ],
        data={
            "query": "Describe this satellite image.",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "captioning"
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]


def test_history_endpoint():
    """Verify history retrieval for a session with recorded queries."""
    img_bytes = _create_test_image_bytes()
    session_id = "test_hist_session_123"
    meta_json = json.dumps({"session_id": session_id})

    client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Locate the lake.", "metadata": meta_json},
    )

    hist_resp = client.get(f"/api/history/{session_id}")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) >= 1
    assert history[-1]["query"] == "Locate the lake."
    assert history[-1]["task_detected"] == "grounding"
