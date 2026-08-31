"""
SatQuery AI — Image Captioning Endpoint.

POST /api/caption
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.models.captioning import run_module as run_captioning
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Captioning"])


class CaptionRequest(BaseModel):
    """Request body for the captioning endpoint."""

    image_id: str = Field(..., description="ID of the uploaded satellite image.")


@router.post("/caption", response_model=AnalysisResponse)
async def generate_caption(req: CaptionRequest) -> AnalysisResponse:
    """Generate a scene description for a satellite image."""
    start = time.time()

    img_path = UPLOADED_IMAGES.get(req.image_id)
    if not img_path:
        found = list(settings.upload_path.rglob(f"{req.image_id}.*"))
        img_path = found[0] if found else None

    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image '{req.image_id}' not found.")

    result = run_captioning(images=[img_path])

    return AnalysisResponse(
        success=True,
        task="caption",
        answer=result["answer"],
        confidence=result.get("confidence"),
        models_used=[result.get("model_name", "satquery-vlm-caption-placeholder")],
        parameters={"image_id": req.image_id},
        evidence=[],
        execution_trace=[
            "Image loaded",
            "Captioning adapter invoked",
            "Placeholder response formatted",
        ],
        processing_time=round(time.time() - start, 4),
    )
