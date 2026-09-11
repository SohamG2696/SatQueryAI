"""
SatQuery AI — Result Interpreter Integration Tests.

Verifies end-to-end integration of the Result Interpretation Layer
into the /api/query pipeline across all tasks and edge cases.
"""

from __future__ import annotations

import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _create_test_image_bytes(color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. VLM → Interpreter → QueryResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_vlm_interpreter_integration():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Are buildings visible in this image?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "vqa"
    assert "answer" in data
    assert len(data["answer"]) > 0
    # Verify execution trace contains Result Interpretation
    trace = data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Grounding → Interpreter → QueryResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_grounding_interpreter_integration():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Find the buildings in this image."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["visual_evidence"]["type"] in ("bbox", "mask")
    assert "Key findings:" in data["answer"] or "Query target" in data["answer"] or "buildings" in data["answer"].lower()
    trace = data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ChangeFormer → Interpreter → QueryResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_change_former_interpreter_integration():
    img1_bytes = _create_test_image_bytes((100, 150, 100))
    img2_bytes = _create_test_image_bytes((150, 100, 100))
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
    assert "change" in data["answer"].lower()
    trace = data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Fusion → Interpreter → QueryResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_fusion_interpreter_integration():
    img1_bytes = _create_test_image_bytes()
    img2_bytes = _create_test_image_bytes()
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
    assert "optical" in data["answer"].lower() or "sar" in data["answer"].lower() or "fusion" in data["answer"].lower()
    trace = data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. GEE → Interpreter → QueryResponse
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_gee_interpreter_integration():
    with patch("app.services.earth_engine.ee") as mock_ee:
        mock_image = MagicMock()
        mock_image.sample.return_value.first.return_value.get.return_value.getInfo.return_value = 1450.0
        mock_ee.Image.return_value = mock_image

        response = client.post(
            "/api/query",
            data={
                "query": "What is the elevation at lat 30.0 lon 75.0?",
                "metadata": json.dumps({"latitude": 30.0, "longitude": 75.0}),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_detected"] == "gee"
        assert "1450.00" in data["answer"]
        assert "SRTM" in data["answer"] or "elevation" in data["answer"].lower()
        trace = data["execution_summary"]["parameters"].get("execution_trace", [])
        assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Multi-Model → Each Result Interpreted and Combined
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_multi_model_interpreter_integration():
    img1_bytes = _create_test_image_bytes((100, 150, 100))
    img2_bytes = _create_test_image_bytes((150, 100, 100))
    meta_json = json.dumps({"dates": ["2024-01-01", "2026-01-01"]})

    response = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1_bytes), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={
            "query": "Find the buildings in this image and tell me whether they changed between the two dates.",
            "metadata": meta_json,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "multi_model"
    assert len(data["answer"]) > 0
    assert "synthesis" in data or "confidence_by_task" in data or "multi-model" in data["answer"].lower()
    trace = data["execution_summary"]["parameters"].get("execution_trace", [])
    assert any("interpretation" in t.lower() for t in trace)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Interpreter Failure → Original Answer Still Returned
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpreter_failure_fallback():
    img_bytes = _create_test_image_bytes()
    with patch(
        "app.services.result_interpreter.interpret_result",
        side_effect=RuntimeError("Interpreter simulated crash"),
    ):
        response = client.post(
            "/api/query",
            files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
            data={"query": "Are buildings visible in this image?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_detected"] == "vqa"
        # Must still return original VLM answer without crashing
        assert "answer" in data
        assert len(data["answer"]) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. API Compatibility Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_api_schema_compatibility():
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Describe this satellite image."},
    )
    assert response.status_code == 200
    data = response.json()
    # Contract keys must be present
    for field in ("task_detected", "answer", "confidence", "visual_evidence", "execution_summary", "verification"):
        assert field in data
