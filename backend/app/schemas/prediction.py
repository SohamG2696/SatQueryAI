"""SatQuery AI — Land Cover Future Prediction Pydantic Schemas."""

from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    region_id: str = Field(
        ...,
        description="Unique region identifier (e.g., 'region_01', 'region_02')",
        json_schema_extra={"example": "region_01"},
    )
    target_year: int = Field(
        ...,
        description="Target forecast year (must be > 2025, e.g., 2027, 2030)",
        json_schema_extra={"example": 2027},
    )


class PredictionResponse(BaseModel):
    region_id: str
    forecast_year: int
    built_up_pct: float
    vegetation_pct: float
    water_pct: float
    method: str
    confidence: str
    r2_scores: Dict[str, float]
    annual_slopes: Dict[str, float]
    interpretation_notes: List[str] = Field(default_factory=list)
    disclaimer: str
