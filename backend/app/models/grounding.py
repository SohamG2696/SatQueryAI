"""
SatQuery AI — Region Grounding Model Adapter.

Implements the standard model contract:
    run_module(images, query, metadata) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for models package resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config import settings
from app.services.image_service import prepare_optical_tensor
from app.utils.device import get_device
from models.grounding.inference import GroundingInferenceEngine

_ENGINE: GroundingInferenceEngine | None = None


def get_grounding_engine() -> GroundingInferenceEngine:
    """Retrieve or lazily initialize the singleton GroundingInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        device = get_device()
        weights_path = Path(settings.grounding_model_path)
        if not weights_path.exists():
            weights_path = _PROJECT_ROOT / settings.grounding_model_path

        vocab_path = Path("models/fusion/vocabulary.json")
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "models/fusion/vocabulary.json"
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "datasets/processed/vocabulary.json"

        _ENGINE = GroundingInferenceEngine(
            weights_path=weights_path if weights_path.exists() else None,
            vocab_path=vocab_path,
            device=device,
        )
    return _ENGINE


def run_module(
    images: list[Any],
    query: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute spatial region grounding on a single satellite image.

    Parameters
    ----------
    images : list
        Single satellite image [image_source].
    query : str
        Spatial query (e.g. "locate the smallest forest region", "highlight water body").
    metadata : dict | None
        Optional metadata.

    Returns
    -------
    dict
        Standard contract response with bounding-box visual evidence.
    """
    if not images:
        raise ValueError("Region Grounding requires at least one satellite image.")

    engine = get_grounding_engine()
    image_src = images[0]
    tensor = prepare_optical_tensor(image_src, target_size=(224, 224), device=engine.device)

    result = engine.predict(tensor, query)

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.82),
        "visual_evidence": result.get("visual_evidence"),
        "model_name": "satquery-region-grounding-v1",
        "parameters": {
            "query": query,
            "coordinate_system": "normalized",
        },
    }
