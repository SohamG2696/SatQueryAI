"""
SatQuery AI — Visual Question Answering (VQA) Model Adapter (Placeholder).

Implements the standard model contract for single-image VQA.
Currently returns controlled unavailable status until the general VLM module is ready.
"""

from __future__ import annotations

from typing import Any


def run_module(
    images: list[Any],
    query: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute single-image Visual Question Answering.

    Parameters
    ----------
    images : list
        Single satellite image.
    query : str
        Natural-language question.
    metadata : dict | None
        Optional metadata.

    Returns
    -------
    dict
        Controlled placeholder response compliant with the standard model contract.
    """
    if not images:
        raise ValueError("VQA requires at least one satellite image.")

    return {
        "answer": (
            "[VLM_PENDING] General remote-sensing Vision-Language Model (VQA) "
            "is scheduled for integration in the next milestone. "
            "The query was successfully received and validated."
        ),
        "confidence": None,
        "visual_evidence": {
            "type": "none",
        },
        "model_name": "satquery-vlm-vqa-placeholder",
        "parameters": {
            "query": query,
            "status": "not_ready",
        },
    }
