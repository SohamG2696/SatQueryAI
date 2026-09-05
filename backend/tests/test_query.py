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
    assert "YES" in data["answer"] or "NO" in data["answer"]
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
    assert len(data["answer"]) > 0
    assert data["execution_summary"]["task_route"] == "bi_temporal_change_analysis"


def test_query_vqa_placeholder_flow():
    """Verify VQA placeholder flow."""
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
    assert len(data["answer"]) > 0


def test_query_captioning_placeholder_flow():
    """Verify Captioning placeholder flow."""
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
    assert len(data["answer"]) > 0


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


def test_query_modality_routing_test_a():
    """Test A: Optical + SAR pair with MCQ query routes to Fusion and executes fusion model (not VLM)."""
    opt_bytes = _create_test_image_bytes((120, 200, 100))
    sar_bytes = _create_test_image_bytes((80, 80, 80))

    response = client.post(
        "/api/query",
        files=[
            ("images", ("optical.png", io.BytesIO(opt_bytes), "image/png")),
            ("images", ("sar.png", io.BytesIO(sar_bytes), "image/png")),
        ],
        data={
            "query": "Which category best describes the image pair: agriculture, forest, water, or urban?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "fusion"
    assert data["execution_summary"]["task_route"] == "two_image_cross_modal_fusion"
    assert "satquery-optical-sar-fusion-v1" in data["execution_summary"]["models_used"]
    assert "satquery-vlm-person-a" not in data["execution_summary"]["models_used"]


def test_query_modality_routing_test_b():
    """Test B: Two optical images with change query route to change_vqa (not Fusion)."""
    opt1_bytes = _create_test_image_bytes((120, 200, 100))
    opt2_bytes = _create_test_image_bytes((130, 210, 110))

    response = client.post(
        "/api/query",
        files=[
            ("images", ("opt1.png", io.BytesIO(opt1_bytes), "image/png")),
            ("images", ("opt2.png", io.BytesIO(opt2_bytes), "image/png")),
        ],
        data={
            "query": "What changed between these two images?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "change_vqa"
    assert data["execution_summary"]["task_route"] != "two_image_cross_modal_fusion"


def test_query_modality_routing_test_c():
    """Test C: Single optical image routes to captioning."""
    opt_bytes = _create_test_image_bytes((100, 150, 200))

    response = client.post(
        "/api/query",
        files=[
            ("images", ("optical.png", io.BytesIO(opt_bytes), "image/png")),
        ],
        data={
            "query": "Describe this satellite image.",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "captioning"
    assert data["execution_summary"]["task_route"] == "single_image_captioning"
    assert any(m in data["execution_summary"]["models_used"] for m in ("satquery-vlm-person-a", "satquery-vlm-caption-placeholder"))


def test_query_modality_routing_test_d():
    """Test D: Optical + SAR pair with built-up regions query routes to Fusion."""
    opt_bytes = _create_test_image_bytes((120, 200, 100))
    sar_bytes = _create_test_image_bytes((80, 80, 80))

    response = client.post(
        "/api/query",
        files=[
            ("images", ("optical.png", io.BytesIO(opt_bytes), "image/png")),
            ("images", ("sar.png", io.BytesIO(sar_bytes), "image/png")),
        ],
        data={
            "query": "Does this area contain built-up regions according to both optical and SAR imagery?",
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "fusion"
    assert data["execution_summary"]["task_route"] == "two_image_cross_modal_fusion"


def test_query_modality_routing_explicit_metadata():
    """Test explicit metadata modalities routing."""
    opt_bytes = _create_test_image_bytes((120, 200, 100))
    sar_bytes = _create_test_image_bytes((80, 80, 80))

    meta_json = json.dumps({"modalities": ["optical", "sar"]})

    response = client.post(
        "/api/query",
        files=[
            ("images", ("img1.png", io.BytesIO(opt_bytes), "image/png")),
            ("images", ("img2.png", io.BytesIO(sar_bytes), "image/png")),
        ],
        data={
            "query": "Which category best describes the scene?",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "fusion"
    assert data["execution_summary"]["task_route"] == "two_image_cross_modal_fusion"

