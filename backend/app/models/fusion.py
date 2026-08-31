"""
SatQuery AI — Optical-SAR Fusion Model Adapter.

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

import torch

from app.config import settings
from app.services.image_service import prepare_optical_tensor, prepare_sar_tensor
from app.utils.device import get_device
from models.fusion.inference import FusionInferenceEngine

_ENGINE: FusionInferenceEngine | None = None


def get_fusion_engine() -> FusionInferenceEngine:
    """Retrieve or lazily initialize the singleton FusionInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        device = get_device()
        weights_path = Path(settings.fusion_model_path)
        if not weights_path.exists():
            # Check relative to project root
            weights_path = _PROJECT_ROOT / settings.fusion_model_path
        if not weights_path.exists():
            alt_path = Path("datasets/processed/multitask_fusion_model_final.pth")
            if alt_path.exists():
                weights_path = alt_path
            elif (_PROJECT_ROOT / "datasets/processed/multitask_fusion_model_final.pth").exists():
                weights_path = _PROJECT_ROOT / "datasets/processed/multitask_fusion_model_final.pth"

        vocab_path = Path("models/fusion/vocabulary.json")
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "models/fusion/vocabulary.json"
        if not vocab_path.exists():
            vocab_path = _PROJECT_ROOT / "datasets/processed/vocabulary.json"

        _ENGINE = FusionInferenceEngine(
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
    """Execute Optical-SAR cross-modal fusion analysis.

    Parameters
    ----------
    images : list
        Expected two images: [optical_image, sar_image].
    query : str
        Natural-language question or instruction.
    metadata : dict | None
        Optional metadata including modality labels, image IDs, etc.

    Returns
    -------
    dict
        Standard contract response:
        {
            "answer": str,
            "confidence": float,
            "visual_evidence": dict | None,
            "model_name": str,
            "parameters": dict
        }
    """
    if len(images) < 2:
        raise ValueError("Optical-SAR Fusion requires two co-registered images (Optical and SAR).")

    engine = get_fusion_engine()

    meta = metadata or {}
    modalities = meta.get("modalities", ["optical", "sar"])

    # Determine optical and SAR assignment
    if len(modalities) >= 2 and modalities[0] == "sar" and modalities[1] == "optical":
        sar_src = images[0]
        optical_src = images[1]
    else:
        optical_src = images[0]
        sar_src = images[1]

    optical_tensor = prepare_optical_tensor(optical_src, target_size=(224, 224), device=engine.device)
    sar_tensor = prepare_sar_tensor(sar_src, target_size=(224, 224), device=engine.device)

    task_hint = meta.get("task_hint")
    result = engine.predict(optical_tensor, sar_tensor, query, task_hint=task_hint)

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence"),
        "visual_evidence": result.get("visual_evidence"),
        "model_name": "satquery-optical-sar-fusion-v1",
        "parameters": {
            "task_sub_type": result.get("task_sub_type", "binary"),
            "probabilities": result.get("probabilities"),
            "query": query,
        },
    }
