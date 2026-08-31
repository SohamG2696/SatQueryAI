"""
SatQuery AI — Satellite Image Upload Endpoint.

POST /api/upload

Accepts GeoTIFF / TIFF / PNG / JPG files, extracts geospatial and sensor metadata,
saves securely into organized folders, and returns an image reference ID.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas.upload import DetectedMetadata, UploadResponse
from app.services.image_service import load_image_array
from app.services.metadata_service import detect_modality, extract_timestamp_from_filename
from app.utils.validators import sanitize_filename, validate_file_extension, validate_file_size

router = APIRouter(prefix="/api", tags=["Upload"])

# In-memory image registry mapping image_id to saved local filepath
UPLOADED_IMAGES: dict[str, Path] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Satellite image file (.tif, .tiff, .png, .jpg, .npz)"),
    modality: Optional[str] = Form(None, description="Optional explicit modality ('optical' or 'sar')"),
) -> UploadResponse:
    """Upload a remote sensing satellite image for multi-modal analysis."""
    original_name = sanitize_filename(file.filename or "image.tif")
    validate_file_extension(original_name)

    contents = await file.read()
    size_bytes = len(contents)
    validate_file_size(size_bytes)

    # 1. Parse Image & Extract Geospatial Metadata
    try:
        arr, meta = load_image_array(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode satellite image file: {str(exc)}",
        )

    # 2. Modality & Timestamp Detection
    mod_info = detect_modality(
        filename=original_name,
        explicit_modality=modality,
        bands=meta.get("bands", 3),
        raster_metadata=meta,
    )
    detected_mod = mod_info["modality"]
    timestamp = extract_timestamp_from_filename(original_name)

    # 3. Secure File Saving
    image_id = f"img_{uuid.uuid4().hex[:12]}"
    suffix = Path(original_name).suffix.lower()
    save_filename = f"{image_id}{suffix}"

    target_subfolder = "sar" if detected_mod == "sar" else "optical"
    save_dir = settings.upload_path / target_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_filename

    with open(save_path, "wb") as f:
        f.write(contents)

    # Register in memory
    UPLOADED_IMAGES[image_id] = save_path

    detected_meta = DetectedMetadata(
        modality=detected_mod,
        detection_method=mod_info["detection_method"],
        width=meta.get("width", 0),
        height=meta.get("height", 0),
        bands=meta.get("bands", 0),
        crs=meta.get("crs"),
        bounds=meta.get("bounds"),
        timestamp=timestamp,
        format=meta.get("format", suffix.lstrip(".")),
    )

    return UploadResponse(
        success=True,
        image_id=image_id,
        filename=original_name,
        size_bytes=size_bytes,
        content_type=file.content_type or "application/octet-stream",
        metadata_detected=detected_meta,
        message=f"Satellite image successfully uploaded and indexed as {image_id}.",
    )
