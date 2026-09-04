"""
SatQuery AI — Bi-Temporal Change-VQA Model Adapter.

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
from app.utils.device import get_device
from models.change_vqa.inference import ChangeVQAInferenceEngine

_ENGINE: ChangeVQAInferenceEngine | None = None


def get_change_vqa_engine() -> ChangeVQAInferenceEngine:
    """Retrieve or lazily initialize the singleton ChangeVQAInferenceEngine (ChangeFormerV6)."""
    global _ENGINE
    if _ENGINE is None:
        device = get_device()

        # Resolve checkpoint path
        weights_path = Path(settings.change_vqa_model_path)
        if not weights_path.exists():
            weights_path = _PROJECT_ROOT / settings.change_vqa_model_path
        if not weights_path.exists():
            # Last-resort: look for any .pth in the weights folder
            fallback = _PROJECT_ROOT / "models" / "change_vqa" / "weights" / "checkpoint_best.pth"
            if fallback.exists():
                weights_path = fallback
            else:
                raise FileNotFoundError(
                    f"[ChangeVQA] Checkpoint not found at '{settings.change_vqa_model_path}' "
                    f"or '{_PROJECT_ROOT / settings.change_vqa_model_path}'. "
                    "No silent fallback to random weights."
                )

        vocab_path = _PROJECT_ROOT / "datasets" / "processed" / "vocabulary.json"

        _ENGINE = ChangeVQAInferenceEngine(
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
    """Execute bi-temporal ChangeFormerV6 change detection analysis.

    Parameters
    ----------
    images : list
        Two images [before, after] — Path, PIL.Image, np.ndarray, or bytes.
    query : str
        Question about the change (e.g. 'Did vegetation increase?').
    metadata : dict | None
        Optional dict; may include 'dates': ['YYYY-MM-DD', 'YYYY-MM-DD'].

    Returns
    -------
    dict with keys: answer, confidence, visual_evidence, model_name, parameters.
    """
    if len(images) < 2:
        raise ValueError("Change-VQA requires two bi-temporal images (before and after).")

    engine = get_change_vqa_engine()
    meta = metadata or {}
    dates = meta.get("dates")

    image_before = images[0]
    image_after = images[1]

    print(f"[ChangeVQA adapter] before={image_before!r}")
    print(f"[ChangeVQA adapter] after ={image_after!r}")

    result = engine.predict(image_before, image_after, query, dates=dates)

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.0),
        "visual_evidence": result.get("visual_evidence", {"type": "none"}),
        "model_name": "ChangeFormerV6",
        "parameters": result.get("parameters", {}),
        # also expose top-level convenience fields
        "raw_answer": result.get("raw_answer"),
        "change_ratio": result.get("change_ratio"),
        "global_change_ratio": result.get("global_change_ratio"),
        "category": result.get("category"),
        "question_type": result.get("question_type"),
        "change_mask_base64": result.get("visual_evidence", {}).get("change_mask_base64"),
        "has_grounding": False,
    }
