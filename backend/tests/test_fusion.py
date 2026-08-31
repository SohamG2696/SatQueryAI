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
