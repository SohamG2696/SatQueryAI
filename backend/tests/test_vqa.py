"""
Tests for Person A VLM VQA and Captioning Adapters.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.vqa import run_module as run_vqa
from app.models.captioning import run_module as run_captioning


def test_vqa_adapter():
    """Verify VQA module returns Person A VLM prediction."""
    img = np.random.rand(224, 224, 3).astype(np.uint8)
    result = run_vqa(images=[img], query="Are buildings visible in this satellite image?")

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert result["model_name"] == "satquery-vlm-person-a"
    assert result["parameters"]["status"] == "ready"


def test_captioning_adapter():
    """Verify Captioning module returns Person A VLM prediction."""
    img = np.random.rand(224, 224, 3).astype(np.uint8)
    result = run_captioning(images=[img], query="Describe this satellite scene.")

    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert result["model_name"] == "satquery-vlm-person-a"
    assert result["parameters"]["status"] == "ready"
