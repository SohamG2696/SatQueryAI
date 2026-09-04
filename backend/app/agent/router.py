"""
SatQuery AI — Deterministic Task Router.

Maps input configuration (image count, modalities, dates, query intent)
to the appropriate specialist remote-sensing task and structured route name.
"""

from __future__ import annotations

from typing import List, Tuple

from .intent_classifier import detect_all_intents, classify_query_intent


def get_route_info(
    query: str,
    image_count: int,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
) -> Tuple[str, str, List[str]]:
    """Determine routing for query and images, returning (task, task_route, tasks_list)."""
    if image_count < 1:
        raise ValueError("At least one satellite image is required.")

    mods = [m.lower().strip() for m in (modalities or [])]
    dates = dates or []
    intents = detect_all_intents(query)

    has_optical = any("optical" in m or "rgb" in m for m in mods)
    has_sar = any("sar" in m or "radar" in m for m in mods)
    has_diff_dates = (len(dates) >= 2 and dates[0] != dates[1])

    # 1. Multi-Model Route (query contains multiple distinct intents)
    if len(intents) > 1:
        task_route = f"multi_model_{'_and_'.join(intents)}"
        return "multi_model", task_route, intents

    # Single intent detected
    intent = intents[0] if intents else "captioning"

    # 2. Cross-Modal Fusion (explicit fusion intent or optical+SAR modalities)
    if (has_optical and has_sar) or intent == "fusion":
        return "fusion", "two_image_cross_modal_fusion", ["fusion"]

    # 3. Bi-Temporal Change Detection (explicit change intent or different dates)
    if has_diff_dates or intent == "change_vqa":
        return "change_vqa", "bi_temporal_change_analysis", ["change_vqa"]

    # 4. Region Grounding
    if intent == "grounding":
        route = "multi_image_region_grounding" if image_count >= 2 else "single_image_region_grounding"
        return "grounding", route, ["grounding"]

    # 5. Image Captioning
    if intent in ("captioning", "empty"):
        return "captioning", "single_image_captioning", ["captioning"]

    # 6. Fallback VQA
    return "vqa", "single_image_vqa", ["vqa"]


def route_request(
    query: str,
    image_count: int,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
) -> Tuple[str, str]:
    """Route request to canonical task name and route name (backward compatible)."""
    task, task_route, _ = get_route_info(
        query=query,
        image_count=image_count,
        modalities=modalities,
        dates=dates,
    )
    return task, task_route
