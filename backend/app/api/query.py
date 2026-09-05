"""
SatQuery AI — Main Query Endpoint.

POST /api/query

The primary entry point for multi-modal remote sensing vision-language analysis.
Accepts 1 or 2 satellite images (via direct upload or image_ids), natural-language query,
and optional metadata. Returns evidence-grounded QueryResponse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.controller import controller
from app.api.history import record_history
from app.api.upload import UPLOADED_IMAGES
from app.config import settings
from app.schemas.response import QueryResponse
from app.services.image_service import load_image_array
from app.services.metadata_service import detect_modality
from app.utils.logging import log_error
from app.utils.validators import validate_query

router = APIRouter(prefix="/api", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def execute_query(
    query: str = Form(..., description="Natural language question or spatial instruction"),
    images: Optional[List[Union[UploadFile, str]]] = File(None, description="One or two satellite image files"),
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

    # 2. Resolve Image Sources and Filenames
    image_sources: List[Any] = []
    image_filenames: List[str] = []

    # Priority A: Direct file uploads in current request
    if images:
        for f in images:
            if hasattr(f, "read") and getattr(f, "filename", None):
                content = await f.read()
                if len(content) > 0:
                    image_sources.append(content)
                    image_filenames.append(str(f.filename))

    # Priority B: Referenced Image IDs from previous uploads
    if image_ids and image_ids.strip():
        id_list = [i.strip() for i in image_ids.split(",") if i.strip()]
        for i_id in id_list:
            if i_id in UPLOADED_IMAGES:
                img_path = UPLOADED_IMAGES[i_id]
                image_sources.append(img_path)
                image_filenames.append(getattr(img_path, "name", str(i_id)))
            else:
                # Search upload directory for matching file
                found = list(settings.upload_path.rglob(f"{i_id}.*"))
                if found:
                    image_sources.append(found[0])
                    image_filenames.append(found[0].name)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Referenced image ID '{i_id}' not found.",
                    )

    # 3. Derive / Supplement Modality Metadata for Routing if images present
    existing_mods = meta_dict.get("modalities")
    if not isinstance(existing_mods, list):
        existing_mods = []

    resolved_modalities: List[str] = []
    for idx, src in enumerate(image_sources):
        fname = image_filenames[idx] if idx < len(image_filenames) else ""
        exp_mod = str(existing_mods[idx]).strip() if idx < len(existing_mods) and existing_mods[idx] else None

        num_bands = 3
        try:
            arr, _ = load_image_array(src)
            num_bands = arr.shape[0]
        except Exception:
            pass

        mod_result = detect_modality(
            filename=fname,
            explicit_modality=exp_mod,
            bands=num_bands,
        )
        resolved_modalities.append(mod_result["modality"])

    if resolved_modalities:
        meta_dict["modalities"] = resolved_modalities

    # 4. Process via Agentic Controller
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
