"""
SatQuery AI — Bi-Temporal Change Detection Endpoint.

POST /api/change
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.models.change_vqa import run_module as run_change_vqa
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Change Detection"])


class ChangeRequest(BaseModel):
    """Request body for the change detection endpoint."""

    image_id_before: str = Field(..., description="ID of the earlier (before) image.")
    image_id_after: str = Field(..., description="ID of the later (after) image.")
    date_before: str = Field(default="", description="Acquisition date of the before image (ISO format).")
    date_after: str = Field(default="", description="Acquisition date of the after image (ISO format).")
    query: str = Field(default="What changed?", description="Optional question about the changes.")


@router.post("/change", response_model=AnalysisResponse)
async def change_detection(req: ChangeRequest) -> AnalysisResponse:
    """Detect and describe changes between two satellite images."""
    start = time.time()

    path_before = UPLOADED_IMAGES.get(req.image_id_before)
    if not path_before:
        found = list(settings.upload_path.rglob(f"{req.image_id_before}.*"))
        path_before = found[0] if found else None

    path_after = UPLOADED_IMAGES.get(req.image_id_after)
    if not path_after:
        found = list(settings.upload_path.rglob(f"{req.image_id_after}.*"))
        path_after = found[0] if found else None

    if not path_before or not path_after:
        raise HTTPException(
            status_code=404,
            detail=f"Images not found: before='{req.image_id_before}', after='{req.image_id_after}'",
        )

    dates = [req.date_before, req.date_after] if (req.date_before and req.date_after) else None
    result = run_change_vqa(
        images=[path_before, path_after],
        query=req.query,
        metadata={"dates": dates},
    )

    return AnalysisResponse(
        success=True,
        task="change",
        answer=result["answer"],
        confidence=result.get("confidence"),
        models_used=[result.get("model_name", "satquery-change-vqa-v1")],
        parameters=result.get("parameters", {}),
        evidence=[],
        execution_trace=[
            "Bi-temporal pair loaded",
            "Change-VQA siamese network executed",
            "Multi-temporal difference classified",
        ],
        processing_time=round(time.time() - start, 4),
    )
