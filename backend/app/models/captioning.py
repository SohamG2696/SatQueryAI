"""
SatQuery AI — Image Captioning Model Adapter (Placeholder).

Implements the standard model contract for satellite scene description.
Currently returns controlled unavailable status until the general VLM module is ready.
"""

from __future__ import annotations

from typing import Any


def run_module(
    images: list[Any],
    query: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute single-image Scene Description / Captioning.

    Parameters
    ----------
    images : list
        Single satellite image.
    query : str
        Optional description instruction.
    metadata : dict | None
        Optional metadata.

    Returns
    -------
    dict
        Controlled placeholder response compliant with the standard model contract.
    """
    if not images:
        raise ValueError("Image Captioning requires at least one satellite image.")

    return {
        "answer": (
            "[VLM_PENDING] Remote-sensing Vision-Language Captioning Model "
            "is scheduled for integration in the next milestone. "
            "The image was successfully received and validated."
        ),
        "confidence": None,
        "visual_evidence": {
            "type": "none",
        },
        "model_name": "satquery-vlm-caption-placeholder",
        "parameters": {
            "query": query,
            "status": "not_ready",
        },
    }
