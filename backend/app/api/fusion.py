"""
SatQuery AI — Optical-SAR Cross-Modal Fusion Endpoint.

POST /api/fusion
"""

from __future__ import annotations

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.models.fusion import run_module as run_fusion
from app.schemas.response import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Fusion"])


class FusionRequest(BaseModel):
    """Request body for the optical-SAR fusion endpoint."""

    optical_image_id: str = Field(..., description="ID of the uploaded optical image.")
    sar_image_id: str = Field(..., description="ID of the uploaded SAR image.")
    query: str = Field(
        default="Confirm land cover features using both modalities.",
        description="Question / instruction for the fused analysis.",
    )


@router.post("/fusion", response_model=AnalysisResponse)
async def optical_sar_fusion(req: FusionRequest) -> AnalysisResponse:
    """Analyze co-registered optical and SAR images through cross-modal fusion."""
    start = time.time()

    # Resolve paths
    opt_path = UPLOADED_IMAGES.get(req.optical_image_id)
    if not opt_path:
        found = list(settings.upload_path.rglob(f"{req.optical_image_id}.*"))
        opt_path = found[0] if found else None

    sar_path = UPLOADED_IMAGES.get(req.sar_image_id)
    if not sar_path:
        found = list(settings.upload_path.rglob(f"{req.sar_image_id}.*"))
        sar_path = found[0] if found else None

    if not opt_path or not sar_path:
        raise HTTPException(
            status_code=404,
            detail=f"Image pair not found: optical='{req.optical_image_id}', sar='{req.sar_image_id}'",
        )

    result = run_fusion(
        images=[opt_path, sar_path],
        query=req.query,
        metadata={"modalities": ["optical", "sar"]},
    )

    visual_ev = result.get("visual_evidence")
    evidence_list = []
    if visual_ev:
        if isinstance(visual_ev, dict) and visual_ev.get("type") == "bbox":
            coords = visual_ev.get("coordinates", [])
            evidence_list.append(f"bbox:{coords}")
        else:
            evidence_list.append(str(visual_ev))

    return AnalysisResponse(
        success=True,
        task="fusion",
        answer=result["answer"],
        confidence=result.get("confidence"),
        models_used=[result.get("model_name", "satquery-optical-sar-fusion-v1")],
        parameters=result.get("parameters", {}),
        visual_evidence=visual_ev,
        evidence=evidence_list,
        execution_trace=[
            "Optical and SAR inputs validated",
            "Cross-modal fusion encoder executed",
            "Prediction computed",
        ],
        processing_time=round(time.time() - start, 4),
    )
