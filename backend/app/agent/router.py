"""
SatQuery AI — Deterministic Task Router.

Maps input configuration (image count, modalities, dates, query intent)
to the appropriate specialist remote-sensing task and structured route name.
"""

from __future__ import annotations

from typing import Tuple

from .intent_classifier import classify_query_intent


def route_request(
    query: str,
    image_count: int,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
) -> Tuple[str, str]:
    """Route an incoming request to the canonical specialist task and route name.

    Parameters
    ----------
    query : str
        User's question or instruction.
    image_count : int
        Number of uploaded satellite images provided.
    modalities : list[str] | None
        Detected or specified modalities per image.
    dates : list[str] | None
        Acquisition dates per image.

    Returns
    -------
    Tuple[str, str]
        (canonical_task_name, structured_task_route)
        e.g. ("fusion", "two_image_cross_modal_fusion")

    Raises
    ------
    ValueError
        If input parameters are invalid or unsupported.
    """
    if image_count < 1:
        raise ValueError("At least one satellite image is required.")

    mods = [m.lower().strip() for m in (modalities or [])]
    dates = dates or []
    intent = classify_query_intent(query)

    # ── CASE 5: Cross-Modal Fusion (2 images, optical + SAR) ──────
    has_optical = any("optical" in m or "rgb" in m for m in mods)
    has_sar = any("sar" in m or "radar" in m for m in mods)
    if image_count >= 2 and (has_optical and has_sar or intent == "fusion"):
        return "fusion", "two_image_cross_modal_fusion"

    # ── CASE 4: Bi-Temporal Change Detection (2 images, temporal) ─
    has_diff_dates = (len(dates) >= 2 and dates[0] != dates[1])
    if image_count >= 2 and (has_diff_dates or intent == "change"):
        return "change_vqa", "bi_temporal_change_analysis"

    # Fallback for 2 images if modalities same and no dates given
    if image_count >= 2:
        if intent == "grounding":
            return "grounding", "multi_image_region_grounding"
        return "change_vqa", "bi_temporal_change_analysis"

    # ── CASE 3: Region Grounding (1 image, spatial localization) ──
    if intent == "grounding":
        return "grounding", "single_image_region_grounding"

    # ── CASE 2: Image Captioning (1 image, scene description) ─────
    if intent in ("captioning", "empty"):
        return "captioning", "single_image_captioning"

    # ── CASE 1: Visual Question Answering (1 image, question) ─────
    return "vqa", "single_image_vqa"
