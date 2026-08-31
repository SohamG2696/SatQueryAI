"""
SatQuery AI — Region Grounding Endpoint.

POST /api/grounding
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.models.grounding import run_module as run_grounding
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Grounding"])


class GroundingRequest(BaseModel):
    """Request body for the grounding endpoint."""

    image_id: str = Field(..., description="ID of the uploaded satellite image.")
    query: str = Field(..., min_length=1, description="Spatial query, e.g. 'highlight the water body'.")


@router.post("/grounding", response_model=AnalysisResponse)
async def region_grounding(req: GroundingRequest) -> AnalysisResponse:
    """Identify and highlight a region in a satellite image based on a text query."""
    start = time.time()

    img_path = UPLOADED_IMAGES.get(req.image_id)
    if not img_path:
        found = list(settings.upload_path.rglob(f"{req.image_id}.*"))
        img_path = found[0] if found else None

    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image '{req.image_id}' not found.")

    result = run_grounding(images=[img_path], query=req.query)

    ev_list = []
    if result.get("visual_evidence"):
        ev_list.append(str(result["visual_evidence"]))

    return AnalysisResponse(
        success=True,
        task="grounding",
        answer=result["answer"],
        confidence=result.get("confidence"),
        models_used=[result.get("model_name", "satquery-region-grounding-v1")],
        parameters=result.get("parameters", {}),
        evidence=ev_list,
        execution_trace=[
            "Satellite image loaded",
            "Spatial grounding network localized target region",
            "Bounding box coordinates returned",
        ],
        processing_time=round(time.time() - start, 4),
    )
