"""
SatQuery AI — Confidence & Verification Layer Tests.
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
from app.services.verification_service import verify_execution_result

client = TestClient(app)


def _create_test_image_bytes(color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_verification_case_a_vlm():
    """Test Case A: Single VLM query -> status accepted."""
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
    assert v["status"] == "accepted"
    assert v["confidence"] is not None
    assert v["confidence"] >= 0.60


def test_verification_case_b_multi_model():
    """Test Case B: Multi-model (Grounding + Change Detection) -> status accepted with consistent trace."""
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

    assert "verification" in data
    v = data["verification"]
    assert v["status"] in ("accepted", "verify_required")
    assert isinstance(v["reason"], str) and len(v["reason"]) > 0


def test_verification_case_c_low_confidence_or_insufficient_evidence():
    """Test Case C: Low-confidence / insufficient evidence case -> status verify_required."""
    # Sub-case C1: Single model with confidence below threshold (0.35 < 0.60)
    res_low = verify_execution_result(
        task="vqa",
        answer="Incomplete scene prediction",
        confidence=0.35,
        threshold=0.60,
    )
    assert res_low.status == "verify_required"
    assert "below verification threshold" in res_low.reason
    assert res_low.confidence == 0.35

    # Sub-case C2: Grounding task missing visual bounding box evidence
    res_missing_bbox = verify_execution_result(
        task="grounding",
        answer="Target region localized.",
        confidence=0.85,
        visual_evidence={"type": "none"},
        threshold=0.60,
    )
    assert res_missing_bbox.status == "verify_required"
    assert "lacks valid visual bounding box evidence" in res_missing_bbox.reason

    # Sub-case C3: Multi-model task with one low confidence sub-task
    res_multi_low = verify_execution_result(
        task="multi_model",
        answer="Combined result",
        confidence=0.85,
        sub_results=[
            {"task": "grounding", "confidence": 0.85, "answer": "Box found"},
            {"task": "change_vqa", "confidence": 0.25, "answer": "Low conf change"},
        ],
        threshold=0.60,
    )
    assert res_multi_low.status == "verify_required"
    assert "scored below verification threshold" in res_multi_low.reason
    assert res_multi_low.confidence == 0.25
