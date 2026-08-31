"""
SatQuery AI — General Analysis Endpoint.

POST /api/analyze

Convenience endpoint that accepts a general analysis request.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agent.router import route_request
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Analysis"])


class AnalyzeRequest(BaseModel):
    """General analysis request."""

    image_ids: list[str] = Field(..., min_length=1, description="Uploaded image IDs.")
    query: str = Field(default="", description="Natural-language question or instruction.")
    modalities: list[str] = Field(default_factory=lambda: ["optical"], description="Modality per image.")
    dates: list[str] | None = Field(default=None, description="Acquisition dates (ISO format).")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Optional task parameters.")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest) -> AnalysisResponse:
    """Run a general analysis — the controller determines the task."""
    start = time.time()

    task, route_name = route_request(
        query=req.query,
        image_count=len(req.image_ids),
        modalities=req.modalities,
        dates=req.dates,
    )

    trace = [
        "Input received",
        f"{len(req.image_ids)} images detected",
        f"Modalities: {', '.join(req.modalities)}",
        f"Task classified -> {task} ({route_name})",
    ]

    return AnalysisResponse(
        success=True,
        task=task,
        answer=f"Task '{task}' was selected by the agentic controller.",
        confidence=0.85,
        models_used=[f"satquery-{task}-model"],
        parameters=req.parameters,
        evidence=[],
        execution_trace=trace,
        processing_time=round(time.time() - start, 4),
    )
