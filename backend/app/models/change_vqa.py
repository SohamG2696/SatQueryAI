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

try:
    from app.config import settings
    _DEFAULT_WEIGHTS_PATH = settings.change_vqa_model_path
except Exception:
    _DEFAULT_WEIGHTS_PATH = "models/change_vqa/weights/checkpoint_best.pth"

try:
    from app.utils.device import get_device
except Exception:
    def get_device():
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

from models.change_vqa.inference import ChangeVQAInferenceEngine

_ENGINE: ChangeVQAInferenceEngine | None = None


def get_change_vqa_engine() -> ChangeVQAInferenceEngine:
    """Retrieve or lazily initialize the singleton ChangeVQAInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        try:
            device = get_device()
        except Exception:
            import torch
            device = torch.device("xpu") if hasattr(torch, "xpu") and torch.xpu.is_available() else torch.device("cpu")

        weights_path = Path(_DEFAULT_WEIGHTS_PATH)
        if not weights_path.exists():
            weights_path = _PROJECT_ROOT / _DEFAULT_WEIGHTS_PATH
        if not weights_path.exists():
            weights_path = _PROJECT_ROOT.parent / "checkpoints" / "full" / "checkpoint_best.pth"

        _ENGINE = ChangeVQAInferenceEngine(
            weights_path=weights_path if weights_path.exists() else None,
            device=device,
        )
    return _ENGINE


def run_module(
    images: list[Any],
    query: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute bi-temporal Change-VQA analysis using ChangeFormerV6 + SemanticGrounder.

    Parameters
    ----------
    images : list
        Expected two images: [before_image, after_image] (str, Path, or PIL.Image).
    query : str
        Question about changes (e.g. "Have the areas of buildings changed?").
    metadata : dict | None
        Optional metadata including acquisition dates.

    Returns
    -------
    dict
        Full Person B response schema with category change statistics and change mask.
    """
    if len(images) < 2:
        raise ValueError("Change-VQA requires two bi-temporal images (before and after).")

    engine = get_change_vqa_engine()
    meta = metadata or {}
    dates = meta.get("dates")

    # Pass original raw image file references directly into the inference engine
    # ChangeAnalyzer handles image loading and resizing to 256x256 internally.
    result = engine.predict(images[0], images[1], query, dates=dates)

    return {
        "answer": result["answer"],
        "raw_answer": result.get("raw_answer"),
        "confidence": result.get("confidence"),
        "change_ratio": result.get("change_ratio"),
        "global_change_ratio": result.get("global_change_ratio"),
        "category": result.get("category"),
        "category_change_ratio": result.get("category_change_ratio"),
        "has_grounding": result.get("has_grounding", False),
        "change_mask_base64": result.get("change_mask_base64"),
        "question_type": result.get("question_type"),
        "model_name": result.get("model", "ChangeFormerV6"),
        "task": result.get("task", "change_detection"),
        "metrics": result.get("metrics"),
        "category_metrics": result.get("category_metrics"),
        "all_category_metrics": result.get("all_category_metrics"),
        "visual_evidence": {
            "type": "change_mask" if result.get("change_mask_base64") else "none",
            "mask_base64": result.get("change_mask_base64"),
        },
        "parameters": result.get("parameters", {}),
    }
