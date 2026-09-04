"""
SatQuery AI — Multi-Model Orchestration & Specialist Routing Test.
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


def _create_test_image_bytes(color=(128, 128, 128)) -> bytes:
    img = Image.new("RGB", (224, 224), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_multi_model_grounding_and_change_orchestration():
    """Verify multi-model orchestration for Grounding + Change Detection."""
    img1_bytes = _create_test_image_bytes((100, 150, 100))
    img2_bytes = _create_test_image_bytes((150, 100, 100))

    meta_json = json.dumps({
        "dates": ["2024-01-01", "2026-01-01"],
    })

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

    print("\n--- Multi-Model Orchestration Response ---")
    print(f"Task Detected:     {data['task_detected']}")
    print(f"Task Route:        {data['execution_summary']['task_route']}")
    print(f"Models Used:       {data['execution_summary']['models_used']}")
    print(f"Answer:            {data['answer']}")
    print(f"Confidence:        {data['confidence']}")
    print(f"Visual Evidence:   {data['visual_evidence']}")
    print(f"Execution Trace:   {data['execution_summary']['parameters'].get('execution_trace')}")
    print(f"Processing Time:   {data['execution_summary']['processing_time_ms']} ms")

    # Assertions
    assert data["task_detected"] == "multi_model"
    assert data["execution_summary"]["task_route"] == "multi_model_grounding_and_change_vqa"

    # Models called must include region grounding and change vqa
    models_used = data["execution_summary"]["models_used"]
    assert "satquery-region-grounding-v1" in models_used
    assert any(m in models_used for m in ("ChangeFormerV6", "satquery-change-vqa-v1"))

    # Visual evidence from grounding should be present
    assert data["visual_evidence"]["type"] == "bbox"
    assert "coordinates" in data["visual_evidence"]

    # Answer should aggregate sub-task outputs
    assert "[grounding]" in data["answer"]
    assert "[change_vqa]" in data["answer"]

    # Execution trace verification
    trace = data["execution_summary"]["parameters"]["execution_trace"]
    assert any("grounding" in t.lower() for t in trace)
    assert any("change" in t.lower() for t in trace)


def test_specialist_route_vlm():
    """Verify specialist routing 1: 'Are buildings visible in this image?' -> VLM."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Are buildings visible in this image?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "vqa"
    assert "satquery-vlm-person-a" in data["execution_summary"]["models_used"]


def test_specialist_route_grounding():
    """Verify specialist routing 2: 'Find the buildings in this image.' -> Grounding."""
    img_bytes = _create_test_image_bytes()
    response = client.post(
        "/api/query",
        files=[("images", ("img.png", io.BytesIO(img_bytes), "image/png"))],
        data={"query": "Find the buildings in this image."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_detected"] == "grounding"
    assert "satquery-region-grounding-v1" in data["execution_summary"]["models_used"]


def test_specialist_route_change_vqa():
    """Verify specialist routing 3: 'Did anything change between these two images?' -> ChangeFormer."""
    img1_bytes = _create_test_image_bytes()
    img2_bytes = _create_test_image_bytes()
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


def test_specialist_route_fusion():
    """Verify specialist routing 4: 'Compare the optical and SAR images.' -> Fusion."""
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
    assert "satquery-optical-sar-fusion-v1" in data["execution_summary"]["models_used"]
