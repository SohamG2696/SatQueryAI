"""
Tests for VQA and Captioning Placeholder Adapters.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.vqa import run_module as run_vqa
from app.models.captioning import run_module as run_captioning


def test_vqa_placeholder():
    """Verify VQA placeholder returns controlled non-crashing status."""
    img = np.random.rand(3, 224, 224).astype(np.float32)
    result = run_vqa(images=[img], query="Is there agriculture in this image?")

    assert "answer" in result
    assert "[VLM_PENDING]" in result["answer"]
    assert result["confidence"] is None
    assert result["parameters"]["status"] == "not_ready"


def test_captioning_placeholder():
    """Verify Captioning placeholder returns controlled non-crashing status."""
    img = np.random.rand(3, 224, 224).astype(np.float32)
    result = run_captioning(images=[img], query="Describe this satellite scene.")

    assert "answer" in result
    assert "[VLM_PENDING]" in result["answer"]
    assert result["confidence"] is None
    assert result["parameters"]["status"] == "not_ready"
