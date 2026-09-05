"""SatQuery AI — GEE Pydantic Schemas."""

from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class GEERequest(BaseModel):
    latitude: float
    longitude: float


class GEENaturalQueryRequest(BaseModel):
    query: str
    latitude: float
    longitude: float


class GEEImageResponse(BaseModel):
    image_id: str
    date: str
    cloud_percentage: Optional[float] = None
    thumbnail_url: str
    explanation: str
    dataset: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    metadata: Dict[str, Any] = {}


class GEEElevationResponse(BaseModel):
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    elevation_mean: Optional[float] = None
    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None
    buffer_m: float = 0.0
    explanation: str


class GEEIndexResponse(BaseModel):
    latitude: float
    longitude: float
    index_name: str
    index_value: float
    date: Optional[str] = None
    dataset: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    explanation: str


class GEEQueryResponse(BaseModel):
    intent: str
    result: Any
    explanation: str
