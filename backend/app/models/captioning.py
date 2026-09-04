"""
SatQuery AI — Image Captioning Model Adapter.

Connects single-image scene description / captioning queries to Person A's LLaVA-OneVision VLM adapter.
"""

from __future__ import annotations

from typing import Any
from app.models.vqa import _to_pil_image
from models.vlm.vlm_adapter import get_vlm_adapter


def run_module(
    images: list[Any],
    query: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute single-image Scene Description / Captioning using Person A's VLM model."""
    if not images:
        raise ValueError("Image Captioning requires at least one satellite image.")

    q = query.strip() if query and query.strip() else "Describe this satellite image in detail."

    pil_img = _to_pil_image(images[0])
    adapter = get_vlm_adapter()
    res = adapter.predict(image=pil_img, question=q)

    return {
        "answer": res["prediction"],
        "confidence": 0.90,
        "visual_evidence": {
            "type": "none",
        },
        "model_name": "satquery-vlm-person-a",
        "parameters": {
            "query": res["question"],
            "raw_prediction": res["raw_prediction"],
            "inference_time_s": res["inference_time_s"],
            "model": res["model"],
            "status": "ready",
        },
    }
