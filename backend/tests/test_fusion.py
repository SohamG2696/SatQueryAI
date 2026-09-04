"""
Tests for Optical-SAR Fusion Model Adapter.
"""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.fusion import run_module as run_fusion


def test_fusion_adapter_binary():
    """Verify Optical-SAR fusion runs on synthetic optical (4-ch) and SAR (2-ch) images."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    result = run_fusion(
        images=[opt_img, sar_img],
        query="Does the SAR imagery confirm the land cover visible in the optical image?",
        metadata={"modalities": ["optical", "sar"]},
    )

    assert "answer" in result
    assert result["answer"] in ("YES", "NO")
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_name"] == "satquery-optical-sar-fusion-v1"


def test_fusion_adapter_mcq():
    """Verify Optical-SAR fusion runs MCQ prediction."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    result = run_fusion(
        images=[opt_img, sar_img],
        query="Which option best describes the scene: (A) Forest (B) Water (C) Urban (D) Agriculture",
        metadata={"modalities": ["optical", "sar"], "task_hint": "mcq"},
    )

    assert "answer" in result
    assert result["answer"].startswith(("A", "B", "C", "D"))
    assert result["confidence"] is not None
    assert result["parameters"]["task_sub_type"] == "mcq"


def test_fusion_natural_language_mcq():
    """Verify natural language MCQ query detection without task_hint."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    result = run_fusion(
        images=[opt_img, sar_img],
        query="Which category best describes the image: agriculture, forest, water, or urban?",
        metadata={"modalities": ["optical", "sar"]},
    )

    assert "answer" in result
    assert result["parameters"]["task_sub_type"] == "mcq"
    assert result["answer"].startswith(("A", "B", "C", "D"))
    assert result["confidence"] is not None


def test_fusion_binary_query_detection():
    """Verify binary YES/NO task detection."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    result = run_fusion(
        images=[opt_img, sar_img],
        query="Does the target area contain buildings?",
        metadata={"modalities": ["optical", "sar"]},
    )

    assert result["parameters"]["task_sub_type"] == "binary"
    assert result["answer"] in ("YES", "NO")


def test_fusion_adapter_bbox():
    """Verify Optical-SAR fusion runs bounding box prediction."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    result = run_fusion(
        images=[opt_img, sar_img],
        query="Locate the target built-up region in both modalities.",
        metadata={"modalities": ["optical", "sar"], "task_hint": "bbox"},
    )

    assert "visual_evidence" in result
    evidence = result["visual_evidence"]
    assert evidence is not None
    assert evidence["type"] == "bbox"
    assert len(evidence["coordinates"]) == 4


def test_fusion_adapter_insufficient_images():
    """Verify ValueError on single image input."""
    with pytest.raises(ValueError):
        run_fusion(images=[np.zeros((4, 224, 224))], query="Test")


def test_fusion_api_endpoint_bbox_evidence():
    """Verify POST /api/fusion returns visual_evidence and evidence list for BBox queries."""
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # Upload optical image
    opt_buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(100, 150, 200)).save(opt_buf, format="PNG")
    opt_buf.seek(0)
    opt_res = client.post("/api/upload", files={"file": ("opt.png", opt_buf, "image/png")}, data={"modality": "optical"})
    assert opt_res.status_code == 200
    opt_id = opt_res.json()["image_id"]

    # Upload SAR image
    sar_buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(50, 50, 50)).save(sar_buf, format="PNG")
    sar_buf.seek(0)
    sar_res = client.post("/api/upload", files={"file": ("sar.png", sar_buf, "image/png")}, data={"modality": "sar"})
    assert sar_res.status_code == 200
    sar_id = sar_res.json()["image_id"]

    # Call POST /api/fusion
    res = client.post(
        "/api/fusion",
        json={
            "optical_image_id": opt_id,
            "sar_image_id": sar_id,
            "query": "Locate the target built-up region in both modalities.",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["task"] == "fusion"
    assert "visual_evidence" in data
    assert data["visual_evidence"] is not None
    assert data["visual_evidence"]["type"] == "bbox"
    coords = data["visual_evidence"]["coordinates"]
    assert len(coords) == 4
    for c in coords:
        assert 0.0 <= c <= 1.0
    assert len(data["evidence"]) > 0
    assert data["evidence"][0].startswith("bbox:")
