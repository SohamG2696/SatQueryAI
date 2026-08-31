"""
SatQuery AI — Visual Question Answering Endpoint.

POST /api/vqa
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.models.vqa import run_module as run_vqa
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["VQA"])


class VQARequest(BaseModel):
    """Request body for the VQA endpoint."""

    image_id: str = Field(..., description="ID of the uploaded satellite image.")
    question: str = Field(..., min_length=1, description="Natural-language question about the image.")


@router.post("/vqa", response_model=AnalysisResponse)
async def visual_question_answering(req: VQARequest) -> AnalysisResponse:
    """Answer a natural-language question about a satellite image."""
    start = time.time()

    img_path = UPLOADED_IMAGES.get(req.image_id)
    if not img_path:
        found = list(settings.upload_path.rglob(f"{req.image_id}.*"))
        img_path = found[0] if found else None

    if not img_path:
        raise HTTPException(status_code=404, detail=f"Image '{req.image_id}' not found.")

    result = run_vqa(images=[img_path], query=req.question)

    return AnalysisResponse(
        success=True,
        task="vqa",
        answer=result["answer"],
        confidence=result.get("confidence"),
        models_used=[result.get("model_name", "satquery-vlm-vqa-placeholder")],
        parameters=result.get("parameters", {}),
        evidence=[],
        execution_trace=[
            "Image loaded",
            "VQA adapter invoked",
            "Placeholder response formatted",
        ],
        processing_time=round(time.time() - start, 4),
    )
