"""
SatQuery AI — Main Query Endpoint.

POST /api/query

The primary entry point for multi-modal remote sensing vision-language analysis.
Accepts 1 or 2 satellite images (via direct upload or image_ids), natural-language query,
and optional metadata. Returns evidence-grounded QueryResponse.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, List, Optional
from PIL import Image

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.controller import controller
from app.api.history import record_history
from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.schemas.response import QueryResponse
from app.utils.logging import log_error
from app.utils.validators import validate_query

router = APIRouter(prefix="/api", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def execute_query(
    query: str = Form(..., description="Natural language question or spatial instruction"),
    images: Optional[List[UploadFile]] = File(None, description="One or two satellite image files"),
    image_ids: Optional[str] = Form(None, description="Comma-separated image IDs (e.g. 'img_123,img_456')"),
    metadata: Optional[str] = Form(None, description="JSON string with modalities, dates, session_id, parameters"),
) -> QueryResponse:
    """Execute multi-modal query across specialist vision-language models."""
    clean_query = validate_query(query, required=True)

    # 1. Parse Metadata JSON
    meta_dict: dict[str, Any] = {}
    if metadata and metadata.strip():
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON format in metadata field.",
            )

    # 2. Resolve Image Sources
    image_sources: List[Any] = []

    # Priority A: Direct file uploads in current request
    if images:
        for f in images:
            content = await f.read()
            if len(content) > 0:
                try:
                    pil_img = Image.open(io.BytesIO(content))
                    image_sources.append(pil_img)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to decode uploaded image file '{f.filename}': {str(exc)}",
                    )

    # Priority B: Referenced Image IDs from previous uploads
    if image_ids and image_ids.strip():
        id_list = [i.strip() for i in image_ids.split(",") if i.strip()]
        for i_id in id_list:
            if i_id in UPLOADED_IMAGES:
                image_sources.append(UPLOADED_IMAGES[i_id])
            else:
                # Search upload directory for matching file
                found = list(settings.upload_path.rglob(f"{i_id}.*"))
                if found:
                    image_sources.append(found[0])
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Referenced image ID '{i_id}' not found.",
                    )

    if not image_sources:
        raise HTTPException(
            status_code=400,
            detail="At least one satellite image (uploaded file or valid image_id) is required.",
        )

    # 3. Process via Agentic Controller
    try:
        response = controller.process_query(
            images=image_sources,
            query=clean_query,
            metadata=meta_dict,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        log_error("POST /api/query", exc)
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(exc)}")

    # 4. Record to Session History if applicable
    session_id = meta_dict.get("session_id")
    if session_id:
        record_history(
            session_id=session_id,
            query=clean_query,
            task_detected=response.task_detected,
            answer=response.answer,
            confidence=response.confidence,
            visual_evidence=response.visual_evidence.model_dump() if response.visual_evidence else None,
        )

    return response
