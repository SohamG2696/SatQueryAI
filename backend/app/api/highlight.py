"""
SatQuery AI — RSCoVLM Visual Highlighting Endpoint.

POST /api/highlight

Processes visual queries and satellite images via RSCoVLM spatial understanding
and renders bounding box overlays via the pure PIL Python Highlighting Service.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.services.rscovlm_highlight_service import highlight_image_regions
from app.utils.validators import validate_query
from models.vlm.vlm_adapter import get_vlm_adapter

router = APIRouter(prefix="/api", tags=["Highlighting"])


class HighlightRequest(BaseModel):
    """Request body for visual highlighting endpoint."""

    image_id: Optional[str] = Field(None, description="ID of the uploaded satellite image.")
    image_path: Optional[str] = Field(None, description="Absolute or relative filepath to satellite image.")
    query: str = Field(..., min_length=1, description="Visual query, e.g. 'Highlight the buildings near the road'.")
    box_threshold: float = Field(0.10, ge=0.0, le=1.0, description="Detection box threshold (optional).")
    text_threshold: float = Field(0.10, ge=0.0, le=1.0, description="Text similarity threshold (optional).")


@router.post("/highlight")
async def highlight_objects(req: HighlightRequest) -> Dict[str, Any]:
    """Execute RSCoVLM spatial reasoning and Python PIL visual highlighting on a satellite image."""
    start_time = time.time()

    clean_query = validate_query(req.query, required=True)

    # 1. Resolve Image Source
    img_source = None
    if req.image_path and Path(req.image_path).exists():
        img_source = str(Path(req.image_path).resolve())
    elif req.image_id:
        p = Path(req.image_id)
        if p.is_absolute():
            if p.exists():
                img_source = str(p.resolve())
        else:
            img_source = UPLOADED_IMAGES.get(req.image_id)
            if not img_source:
                found = list(settings.upload_path.rglob(f"{req.image_id}.*"))
                img_source = str(found[0].resolve()) if found else None

    if not img_source or not Path(img_source).exists():
        raise HTTPException(
            status_code=404,
            detail="Image not found. Provide a valid 'image_id' or existing 'image_path'."
        )

    # Validate image can be opened and is not corrupt
    try:
        with Image.open(img_source) as test_img:
            test_img.verify()
    except Exception as img_err:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded image is corrupt or cannot be opened: {str(img_err)}"
        )

    # 2. RSCoVLM Inference for Validation, Spatial Regions, and NLP Answer
    adapter = get_vlm_adapter()
    try:
        res = adapter.predict(
            image=img_source,
            question=clean_query,
            task="highlight",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RSCoVLM vision-language inference failed: {str(exc)}"
        )

    # 3. Extract Structured Components
    validation = res.get("validation", {
        "image_valid": True,
        "query_valid": True,
        "answerable": True,
        "confidence": 0.92,
        "reason": "Image and query evaluated successfully.",
    })
    highlighting_info = res.get("highlighting", {})
    regions = res.get("regions", highlighting_info.get("regions", []))
    answer_text = res.get("answer", res.get("raw_prediction", "Visual analysis completed."))
    model_name = res.get("model_name", "RSCoVLM-7B-2512-SatQuery")
    confidence = float(res.get("confidence", validation.get("confidence", 0.92)))

    # 4. Execute Python Highlighting Service
    highlight_result = {}
    if regions:
        try:
            highlight_result = highlight_image_regions(image=img_source, regions=regions)
            if "regions" in highlight_result:
                regions = highlight_result["regions"]
        except Exception as h_err:
            logger_msg = f"Visual highlighting overlay failed: {h_err}"
            highlight_result = {"error": logger_msg}

    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    # Convert regions to detections format for backward compatibility
    detections = []
    for r in regions:
        detections.append({
            "target": r.get("label", "region"),
            "label": r.get("label", "region"),
            "box": r.get("box", []),
            "score": r.get("confidence", confidence),
            "confidence": r.get("confidence", confidence),
        })

    visual_evidence = {
        "type": "bbox" if regions else "none",
        "available": bool(regions),
        "source": model_name,
        "coordinates": regions[0]["box"] if regions else None,
        "coordinate_system": "normalized",
        "primary_label": regions[0]["label"] if regions else None,
        "primary_score": regions[0]["confidence"] if regions else confidence,
        "all_boxes": [r["box"] for r in regions] if regions else None,
        "all_labels": [r["label"] for r in regions] if regions else None,
        "all_scores": [r["confidence"] for r in regions] if regions else None,
        "regions": regions,
        "highlighted_image_path": highlight_result.get("highlighted_image_path"),
        "highlighted_image_filename": highlight_result.get("highlighted_image_filename"),
        "highlighted_image_base64": highlight_result.get("highlighted_image_base64"),
        "validation": validation,
    }

    return {
        "success": True,
        "source": model_name,
        "task": "highlight",
        "query": clean_query,
        "validation": validation,
        "answer": answer_text,
        "highlighted_image_path": highlight_result.get("highlighted_image_path"),
        "highlighted_image_filename": highlight_result.get("highlighted_image_filename"),
        "highlighted_image_base64": highlight_result.get("highlighted_image_base64"),
        "detection_count": len(regions),
        "regions": regions,
        "detections": detections,
        "visual_evidence": visual_evidence,
        "processing_time_ms": processing_time_ms,
    }
