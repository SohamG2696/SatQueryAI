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
    assert result["answer"] in ("A", "B", "C", "D")
    assert result["confidence"] is not None


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


def test_fusion_builtup_regions_query_classification_binary():
    """Verify query 'Does this area contain built-up regions according to both optical and SAR imagery?' is classified as binary, NOT bbox."""
    opt_img = np.random.rand(4, 224, 224).astype(np.float32)
    sar_img = np.random.rand(2, 224, 224).astype(np.float32)

    query = "Does this area contain built-up regions according to both optical and SAR imagery?"
    result = run_fusion(
        images=[opt_img, sar_img],
        query=query,
        metadata={"modalities": ["optical", "sar"]},
    )

    assert result["parameters"]["task_sub_type"] == "binary"
    assert result["answer"] in ("YES", "NO")


def test_fusion_api_endpoint():
    """Verify POST /api/fusion handles validation, image pair resolution, and status codes."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.upload import UPLOADED_IMAGES

    client = TestClient(app)

    # 1. Reject empty image ID
    resp = client.post("/api/fusion", json={"optical_image_id": "", "sar_image_id": "sar1", "query": "test"})
    assert resp.status_code == 400

    # 2. Reject identical image IDs
    resp = client.post("/api/fusion", json={"optical_image_id": "img1", "sar_image_id": "img1", "query": "test"})
    assert resp.status_code == 400

    # 3. Reject non-existent image ID
    resp = client.post("/api/fusion", json={"optical_image_id": "nonexistent_opt", "sar_image_id": "nonexistent_sar", "query": "test"})
    assert resp.status_code == 404

    # 4. Valid uploaded image pair returns 200 with task="fusion" and never routes to VLM
    # Register dummy files
    dummy_opt = Path("test_opt.png")
    dummy_sar = Path("test_sar.png")
    from PIL import Image
    Image.new("RGB", (224, 224)).save(dummy_opt)
    Image.new("RGB", (224, 224)).save(dummy_sar)

    UPLOADED_IMAGES["dummy_opt_id"] = dummy_opt
    UPLOADED_IMAGES["dummy_sar_id"] = dummy_sar

    try:
        resp = client.post(
            "/api/fusion",
            json={
                "optical_image_id": "dummy_opt_id",
                "sar_image_id": "dummy_sar_id",
                "query": "Does this area contain built-up regions according to both optical and SAR imagery?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["task"] == "fusion"
        assert data["models_used"] == ["satquery-optical-sar-fusion-v1"]
        assert data["parameters"]["task_sub_type"] == "binary"
        assert data["answer"] in ("YES", "NO")
    finally:
        if dummy_opt.exists():
            dummy_opt.unlink()
        if dummy_sar.exists():
            dummy_sar.unlink()
        UPLOADED_IMAGES.pop("dummy_opt_id", None)
        UPLOADED_IMAGES.pop("dummy_sar_id", None)

