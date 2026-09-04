"""
SatQuery AI — Full End-to-End System Validation Test Suite.

Validates end-to-end API workflows, routing, execution, response schema,
verification metadata, and error handling across all specialist models:
1. Person A VLM (VQA)
2. Person A Captioning
3. Region Grounding
4. ChangeFormer Bi-Temporal Change Detection
5. Optical-SAR Fusion
6. Multi-Model Agentic Orchestration
7. Routing Isolation
8. Error Handling
9. Verification Layer Metadata
"""

import io
import json
import pytest
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def _create_test_image_bytes(color=(128, 128, 128), size=(224, 224)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# TEST 1 — PERSON A VLM
# ============================================================================
def test_e2e_person_a_vlm():
    """Verify POST /api/query routes 'Are buildings visible in this image?' to Person A VLM."""
    img_bytes = _create_test_image_bytes((100, 150, 80))
    response = client.post(
        "/api/query",
        files=[("images", ("optical.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Are buildings visible in this image?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "vqa"
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert data["confidence"] is not None
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]
    assert "verification" in data
    assert data["verification"]["status"] in ("accepted", "verify_required")
    assert "reason" in data["verification"]
    assert "confidence" in data["verification"]


# ============================================================================
# TEST 2 — PERSON A CAPTIONING
# ============================================================================
def test_e2e_person_a_captioning():
    """Verify POST /api/query routes 'Describe this satellite image.' to Person A VLM."""
    img_bytes = _create_test_image_bytes((80, 120, 160))
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Describe this satellite image."},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "captioning"
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]
    assert "verification" in data
    assert data["verification"]["status"] in ("accepted", "verify_required")


# ============================================================================
# TEST 3 — GROUNDING
# ============================================================================
def test_e2e_grounding():
    """Verify POST /api/query routes 'Find the buildings in this image.' to Region Grounding."""
    img_bytes = _create_test_image_bytes((120, 80, 50))
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Find the buildings in this image."},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "grounding"
    assert "satquery-region-grounding-v1" in data["execution_summary"]["models_used"]
    assert data["visual_evidence"]["type"] == "bbox"
    assert "coordinates" in data["visual_evidence"]
    assert data["visual_evidence"]["coordinate_system"] == "normalized"
    assert "verification" in data


# ============================================================================
# TEST 4 — CHANGE DETECTION
# ============================================================================
def test_e2e_change_detection():
    """Verify POST /api/query routes bi-temporal query to ChangeFormerV6."""
    img1_bytes = _create_test_image_bytes((100, 150, 100))
    img2_bytes = _create_test_image_bytes((150, 100, 50))
    meta_json = json.dumps({"dates": ["2024-01-01", "2026-01-01"]})

    response = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1_bytes), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": "Did anything change between these two images?",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "change_vqa"
    assert any(m in data["execution_summary"]["models_used"] for m in ("ChangeFormerV6", "satquery-change-vqa-v1"))
    assert data["confidence"] is not None
    assert "verification" in data


# ============================================================================
# TEST 5 — OPTICAL-SAR FUSION
# ============================================================================
def test_e2e_optical_sar_fusion():
    """Verify POST /api/query routes cross-modal query to MultiTaskFusionModel."""
    img1_bytes = _create_test_image_bytes((100, 150, 200))
    img2_bytes = _create_test_image_bytes((200, 100, 50))
    meta_json = json.dumps({"modalities": ["optical", "sar"]})

    response = client.post(
        "/api/query",
        files=[
            ("images", ("opt.png", io.BytesIO(img1_bytes), "image/png")),
            ("images", ("sar.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": "Compare the optical and SAR images.",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "fusion"
    assert "satquery-optical-sar-fusion-v1" in data["execution_summary"]["models_used"]
    assert data["confidence"] is not None
    assert "verification" in data


# ============================================================================
# TEST 6 — MULTI-MODEL ORCHESTRATION
# ============================================================================
def test_e2e_multi_model_orchestration():
    """Verify multi-model intent detection, execution trace, and result aggregation."""
    img1_bytes = _create_test_image_bytes((100, 150, 100))
    img2_bytes = _create_test_image_bytes((150, 100, 100))
    meta_json = json.dumps({"dates": ["2024-01-01", "2026-01-01"]})

    query = "Find the buildings in this image and tell me whether they changed between the two dates."

    response = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1_bytes), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": query,
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "multi_model"
    assert data["execution_summary"]["task_route"] == "multi_model_grounding_and_change_vqa"

    models_used = data["execution_summary"]["models_used"]
    assert "satquery-region-grounding-v1" in models_used
    assert any(m in models_used for m in ("ChangeFormerV6", "satquery-change-vqa-v1"))

    assert "[grounding]" in data["answer"]
    assert "[change_vqa]" in data["answer"]
    assert data["visual_evidence"]["type"] == "bbox"

    trace = data["execution_summary"]["parameters"]["execution_trace"]
    assert any("grounding" in t.lower() for t in trace)
    assert any("change" in t.lower() for t in trace)
    assert "verification" in data


# ============================================================================
# TEST 7 — VLM + SPECIALIST ISOLATION
# ============================================================================
def test_e2e_routing_isolation():
    """Verify that Person A VLM addition did NOT alter specialist routing logic."""
    img_bytes = _create_test_image_bytes()
    img2_bytes = _create_test_image_bytes()

    # Route 1: VLM
    r1 = client.post(
        "/api/query",
        files=[("images", ("i1.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Are buildings visible in this image?"},
    )
    assert r1.json()["task_detected"] == "vqa"

    # Route 2: Grounding
    r2 = client.post(
        "/api/query",
        files=[("images", ("i1.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Find the buildings in this image."},
    )
    assert r2.json()["task_detected"] == "grounding"

    # Route 3: Change Detection
    r3 = client.post(
        "/api/query",
        files=[
            ("images", ("i1.png", io.BytesIO(img_bytes), "image/png")),
            ("images", ("i2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": "Did anything change between these two images?",
            "metadata": json.dumps({"dates": ["2024-01-01", "2026-01-01"]}),
        },
    )
    assert r3.json()["task_detected"] == "change_vqa"

    # Route 4: Fusion
    r4 = client.post(
        "/api/query",
        files=[
            ("images", ("i1.png", io.BytesIO(img_bytes), "image/png")),
            ("images", ("i2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": "Compare the optical and SAR images.",
            "metadata": json.dumps({"modalities": ["optical", "sar"]}),
        },
    )
    assert r4.json()["task_detected"] == "fusion"


# ============================================================================
# TEST 8 — ERROR HANDLING
# ============================================================================
def test_e2e_error_handling():
    """Verify error responses for missing images, single image change, empty query, and invalid image ID."""
    img_bytes = _create_test_image_bytes()

    # A. VLM / query with missing image (0 images)
    res_no_img = client.post(
        "/api/query",
        data={"query": "Are buildings visible in this image?"},
    )
    assert res_no_img.status_code in (400, 422)

    # B. Change query with only one image
    res_single_change = client.post(
        "/api/query",
        files=[("images", ("i1.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Did anything change between these two images?"},
    )
    assert res_single_change.status_code in (400, 422)

    # C. Empty query
    res_empty_q = client.post(
        "/api/query",
        files=[("images", ("i1.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "   "},
    )
    assert res_empty_q.status_code in (400, 422)

    # D. Invalid image ID
    res_invalid_id = client.post(
        "/api/query",
        data={
            "query": "Are buildings visible in this image?",
            "image_ids": "non_existent_img_id_99999",
        },
    )
    assert res_invalid_id.status_code in (400, 404, 422)


# ============================================================================
# TEST 9 — VERIFICATION LAYER METADATA
# ============================================================================
def test_e2e_verification_metadata_presence():
    """Verify that all successful query responses contain verification metadata."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Are buildings visible in this image?"},
    )
    assert response.status_code == 200
    data = response.json()

    assert "verification" in data
    v = data["verification"]
    assert "status" in v
    assert "reason" in v
    assert "confidence" in v
    assert v["status"] in ("accepted", "verify_required")
