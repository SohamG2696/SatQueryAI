"""
SatQuery AI — Bi-Temporal Change-VQA Model Adapter.

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
from models.change_vqa.inference import ChangeVQAInferenceEngine

_ENGINE: ChangeVQAInferenceEngine | None = None


def get_change_vqa_engine() -> ChangeVQAInferenceEngine:
    """Retrieve or lazily initialize the singleton ChangeVQAInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        device = get_device()
        weights_path = Path(settings.change_vqa_model_path)
        if not weights_path.exists():
            weights_path = _PROJECT_ROOT / settings.change_vqa_model_path

        vocab_path = Path("models/fusion/vocabulary.json")
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "models/fusion/vocabulary.json"
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "datasets/processed/vocabulary.json"

        _ENGINE = ChangeVQAInferenceEngine(
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
    """Execute bi-temporal Change-VQA analysis.

    Parameters
    ----------
    images : list
        Expected two images: [before_image, after_image].
    query : str
        Question about changes (e.g. "Did vegetation increase between 2022 and 2025?").
    metadata : dict | None
        Optional metadata including acquisition dates.

    Returns
    -------
    dict
        Standard contract response.
    """
    if len(images) < 2:
        raise ValueError("Change-VQA requires two bi-temporal images (before and after).")

    engine = get_change_vqa_engine()
    meta = metadata or {}
    dates = meta.get("dates")

    t1_tensor = prepare_optical_tensor(images[0], target_size=(224, 224), device=engine.device)
    t2_tensor = prepare_optical_tensor(images[1], target_size=(224, 224), device=engine.device)

    result = engine.predict(t1_tensor, t2_tensor, query, dates=dates)

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.78),
        "visual_evidence": result.get("visual_evidence", {"type": "none"}),
        "model_name": "satquery-change-vqa-v1",
        "parameters": result.get("parameters", {}),
    }
