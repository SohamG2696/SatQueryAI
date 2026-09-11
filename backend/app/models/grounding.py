"""
SatQuery AI — Region Grounding Model Adapter.

Implements the standard model contract:
    run_module(images, query, metadata) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
        
        raw_path = Path(settings.grounding_model_path)
        if raw_path.is_absolute() and raw_path.exists():
            weights_path = raw_path
        else:
            weights_path = (_PROJECT_ROOT / settings.grounding_model_path).resolve()

        if not weights_path.exists():
            alt_path = (_PROJECT_ROOT / "models" / "grounding" / "weights" / "spatial_grounding_model.pth").resolve()
            if alt_path.exists():
                weights_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Grounding model weights not found at '{weights_path}' or '{alt_path}'. "
                    "Random weights fallbacks are disabled."
                )

        vocab_path = Path("models/fusion/vocabulary.json")
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "models/fusion/vocabulary.json"
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "datasets/processed/vocabulary.json"

        _ENGINE = GroundingInferenceEngine(
            weights_path=weights_path,
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

    orig_q = (metadata or {}).get("original_query") or (metadata or {}).get("query")
    effective_q = query
    if orig_q and any(k in orig_q.lower() for k in ("bounding box", "bbox", "smallest", "largest", "single", "contiguous", "bounding", "box")):
        effective_q = orig_q

    result = engine.predict(image_src, effective_q)

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.82),
        "visual_evidence": result.get("visual_evidence"),
        "model_name": result.get("model_name", "satquery-region-grounding-v1"),
        "parameters": {
            "query": query,
            "normalized_query": result.get("normalized_query"),
            "coordinate_system": result.get("coordinate_system", "normalized"),
            "requested_target": result.get("requested_target"),
            "model_target": result.get("model_target"),
            "semantic_fallback": result.get("semantic_fallback"),
            "fallback_reason": result.get("fallback_reason"),
            "warning": result.get("warning"),
            "confidence_type": result.get("confidence_type"),
            "classification_method": result.get("classification_method"),
            "target_coverage_pct": result.get("target_coverage_pct"),
        },
    }
