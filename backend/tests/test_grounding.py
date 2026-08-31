"""
Tests for Region Grounding Model Adapter.
"""

import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.grounding import run_module as run_grounding


def test_grounding_adapter():
    """Verify single image region grounding produces normalized bounding boxes."""
    img = np.random.rand(3, 224, 224).astype(np.float32)

    result = run_grounding(
        images=[img],
        query="Locate the smallest contiguous forest region.",
    )

    assert "answer" in result
    assert "confidence" in result
    assert result["model_name"] == "satquery-region-grounding-v1"

    evidence = result.get("visual_evidence")
    assert evidence is not None
    assert evidence["type"] == "bbox"
    coords = evidence["coordinates"]
    assert len(coords) == 4
    x1, y1, x2, y2 = coords
    assert 0.0 <= x1 <= x2 <= 1.0
    assert 0.0 <= y1 <= y2 <= 1.0


def test_grounding_adapter_empty_images():
    """Verify error on empty image list."""
    with pytest.raises(ValueError):
        run_grounding(images=[], query="Locate water")
