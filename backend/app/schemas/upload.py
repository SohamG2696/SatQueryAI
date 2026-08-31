"""
SatQuery AI — Image Upload Schemas.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class DetectedMetadata(BaseModel):
    """Geospatial, sensor, and dimensional metadata extracted from uploaded images."""

    modality: str = Field(default="optical", description="Sensor modality: 'optical' or 'sar'.")
    detection_method: str = Field(default="heuristic", description="Method used: 'metadata', 'filename', or 'heuristic'.")
    width: int = Field(default=0, description="Image width in pixels.")
    height: int = Field(default=0, description="Image height in pixels.")
    bands: int = Field(default=0, description="Number of spectral/polarimetric bands.")
    crs: Optional[str] = Field(default=None, description="Coordinate Reference System (e.g. 'EPSG:32633').")
    bounds: Optional[List[float]] = Field(default=None, description="Geographic/projected bounding extent.")
    timestamp: Optional[str] = Field(default=None, description="Extracted acquisition date/time in ISO format.")
    format: str = Field(default="", description="Image format (e.g. 'geotiff', 'png', 'jpeg').")


class UploadResponse(BaseModel):
    """Response returned upon successful satellite image upload."""

    success: bool = True
    image_id: str = Field(..., description="Unique persistent identifier for the image.")
    filename: str = Field(..., description="Original filename uploaded.")
    size_bytes: int = Field(..., ge=0, description="File size in bytes.")
    content_type: str = Field(default="", description="MIME content type.")
    metadata_detected: DetectedMetadata = Field(..., description="Extracted geospatial and sensor attributes.")
    message: str = Field(default="File uploaded successfully.")
