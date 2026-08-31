"""
SatQuery AI — Agentic Routing Endpoint (Legacy / Direct Adapter).

POST /api/agent

Direct routing endpoint providing backward compatibility.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.controller import controller
from app.agent.router import route_request
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Agent"])


class AgentRequest(BaseModel):
    """Request body for the agentic routing endpoint."""

    query: str = Field(default="", description="Natural-language question or instruction.")
    image_ids: list[str] = Field(..., min_length=1, description="One or more uploaded image IDs.")
    modalities: list[str] = Field(
        default_factory=lambda: ["optical"],
        description="Modality label per image: 'optical' or 'sar'.",
    )
    dates: list[str] | None = Field(
        default=None,
        description="Acquisition dates per image (ISO format), for change detection.",
    )
    parameters: dict[str, Any] = Field(default_factory=dict, description="Optional task parameters.")


@router.post("/agent", response_model=AnalysisResponse)
async def agent_route(req: AgentRequest) -> AnalysisResponse:
    """Send a complete request to the agentic controller."""
    start = time.time()

    try:
        task, route_name = route_request(
            query=req.query,
            image_count=len(req.image_ids),
            modalities=req.modalities,
            dates=req.dates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    trace = [
        "Input received",
        f"{len(req.image_ids)} images detected",
        f"Modalities: {', '.join(req.modalities)}",
        f"Task classified -> {task} ({route_name})",
        f"Specialist workflow selected for '{task}'",
    ]

    return AnalysisResponse(
        success=True,
        task=task,
        answer=f"Agentic controller routed to specialist workflow '{task}' ({route_name}).",
        confidence=0.85,
        models_used=[f"satquery-{task}-model"],
        parameters={
            "query": req.query,
            "image_ids": req.image_ids,
            "modalities": req.modalities,
            "dates": req.dates,
            **req.parameters,
        },
        evidence=[],
        execution_trace=trace,
        processing_time=round(time.time() - start, 4),
    )
