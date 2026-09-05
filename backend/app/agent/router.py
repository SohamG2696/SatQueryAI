"""
SatQuery AI — Task-Aware Deterministic Task Router.

Maps structured AnalysisPlan to the appropriate specialist remote-sensing task
and route string.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def get_route_info(
    query: str,
    image_count: int = 0,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
    plan: Any | None = None,
) -> Tuple[str, str, List[str]]:
    """Determine routing for query and images, returning (task, task_route, tasks_list)."""
    if plan is None:
        from app.agent.planner import build_analysis_plan
        meta = {"modalities": modalities, "dates": dates}
        plan = build_analysis_plan(query, image_count=image_count, metadata=meta)

    primary = plan.primary_task
    sub_tasks = getattr(plan, "sub_tasks", [primary])

    if primary == "gee":
        return "gee", "google_earth_engine_fetch", ["gee"]

    if primary == "multi_model":
        task_route = f"multi_model_{'_and_'.join(sub_tasks)}"
        return "multi_model", task_route, sub_tasks

    if primary == "fusion":
        return "fusion", "two_image_cross_modal_fusion", ["fusion"]

    if primary in ("change_vqa", "change"):
        return "change_vqa", "bi_temporal_change_analysis", ["change_vqa"]

    if primary == "grounding":
        route = "multi_image_region_grounding" if image_count >= 2 else "single_image_region_grounding"
        return "grounding", route, ["grounding"]

    if primary == "captioning":
        return "captioning", "single_image_captioning", ["captioning"]

    return "vqa", "single_image_vqa", ["vqa"]


def route_request(
    query: str,
    image_count: int = 0,
    modalities: list[str] | None = None,
    dates: list[str] | None = None,
) -> Tuple[str, str]:
    """Route request to canonical task name and route name (backward compatible)."""
    from app.agent.planner import build_analysis_plan, validate_plan_inputs
    meta = {"modalities": modalities, "dates": dates}
    plan = build_analysis_plan(query, image_count=image_count, metadata=meta)
    task, task_route, _ = get_route_info(
        query=query,
        image_count=image_count,
        modalities=modalities,
        dates=dates,
        plan=plan,
    )
    validate_plan_inputs(plan, image_count, modalities)
    return task, task_route
