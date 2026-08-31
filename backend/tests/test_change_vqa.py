"""
Tests for Bi-Temporal Change-VQA Model Adapter.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.change_vqa import run_module as run_change_vqa


def test_change_vqa_adapter():
    """Verify bi-temporal change evaluation on before and after image pair."""
    t1_img = np.random.rand(4, 224, 224).astype(np.float32)
    t2_img = np.random.rand(4, 224, 224).astype(np.float32)

    result = run_change_vqa(
        images=[t1_img, t2_img],
        query="Did the amount of vegetation increase between 2022 and 2025?",
        metadata={"dates": ["2022-01-01", "2025-01-01"]},
    )

    assert "answer" in result
    assert result["answer"] in ("YES", "NO")
    assert "confidence" in result
    assert result["model_name"] == "satquery-change-vqa-v1"
    assert "parameters" in result


def test_change_vqa_adapter_single_image_error():
    """Verify error when only one image is supplied."""
    with pytest.raises(ValueError):
        run_change_vqa(images=[np.zeros((3, 224, 224))], query="Did vegetation increase?")
