"""
SatQuery AI — Satellite Acquisition API Router.

Provides endpoints for real satellite imagery acquisition (Sentinel-2 via CDSE).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.sentinel_service import sentinel_service

router = APIRouter(prefix="/api/satellite", tags=["Satellite Acquisition"])


class Sentinel2Request(BaseModel):
    """Request payload for Sentinel-2 L2A imagery acquisition."""

    bbox: List[float] = Field(
        ...,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84 (EPSG:4326)",
        examples=[[72.82, 18.95, 72.84, 18.97]],
    )
    date_from: str = Field(
        default="2025-01-01T00:00:00Z",
        description="Start ISO date string",
    )
    date_to: str = Field(
        default="2025-01-31T23:59:59Z",
        description="End ISO date string",
    )
    width: int = Field(default=256, ge=64, le=2048)
    height: int = Field(default=256, ge=64, le=2048)
    max_cloud_coverage: int = Field(default=30, ge=0, le=100)


class Sentinel2Response(BaseModel):
    """Standardized response metadata for satellite acquisition."""

    success: bool
    image_id: Optional[str] = None
    source: str = "Copernicus Data Space"
    satellite: str = "Sentinel-2"
    product: str = "Sentinel-2 L2A"
    bbox: List[float] = Field(default_factory=list)
    date_from: str = ""
    date_to: str = ""
    image_url: Optional[str] = None
    width: int = 256
    height: int = 256
    size_bytes: int = 0
    message: Optional[str] = None
    detail: Optional[str] = None


@router.post(
    "/sentinel2",
    response_model=Sentinel2Response,
    status_code=status.HTTP_200_OK,
    summary="Acquire Real Sentinel-2 L2A Imagery",
    description="Acquires real Sentinel-2 L2A imagery from Copernicus Data Space for the specified bounding box.",
)
async def acquire_sentinel2_imagery(payload: Sentinel2Request) -> Sentinel2Response:
    if len(payload.bbox) != 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bbox must contain exactly 4 numbers: [min_lon, min_lat, max_lon, max_lat]",
        )

    min_lon, min_lat, max_lon, max_lat = payload.bbox
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Longitude values must be between -180 and 180",
        )
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Latitude values must be between -90 and 90",
        )
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_lon must be < max_lon and min_lat must be < max_lat",
        )

    res = sentinel_service.acquire_sentinel2_image(
        bbox=payload.bbox,
        date_from=payload.date_from,
        date_to=payload.date_to,
        width=payload.width,
        height=payload.height,
        max_cloud_coverage=payload.max_cloud_coverage,
    )

    return Sentinel2Response(**res)


@router.get(
    "/image/{image_id}",
    response_class=FileResponse,
    summary="Retrieve Acquired Satellite Image",
    description="Returns the acquired satellite PNG file safely.",
)
async def get_acquired_satellite_image(image_id: str):
    # Sanitize image_id to prevent directory traversal
    if not re.match(r"^[a-zA-Z0-9_-]+$", image_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image_id parameter",
        )

    temp_dir = settings.upload_path / "temporary"
    file_path = temp_dir / f"sentinel2_{image_id}.png"

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Acquired satellite image '{image_id}' not found",
        )

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=f"sentinel2_{image_id}.png",
    )
