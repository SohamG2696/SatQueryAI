"""
SatQuery AI — Execution and Evidence Schemas.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class VisualEvidence(BaseModel):
    """Structured visual evidence returned for grounded regions, change maps, or attention."""

    type: str = Field(
        default="none",
        description="Type of visual evidence: 'bbox', 'change_map', 'mask', 'none'.",
    )
    coordinates: Optional[List[Any]] = Field(
        default=None,
        description="Bounding box coordinates (e.g. [x1, y1, x2, y2] in normalized space).",
    )
    coordinate_system: Optional[str] = Field(
        default="normalized",
        description="Coordinate convention: 'normalized' (0-1) or 'pixel'.",
    )
    data: Optional[Any] = Field(
        default=None,
        description="Optional auxiliary data (e.g., mask array, heatmap values).",
    )


class ExecutionSummary(BaseModel):
    """Factual, auditable execution trace and performance metrics."""

    models_used: List[str] = Field(
        default_factory=list,
        description="List of specialist model names that participated in inference.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific parameters and evaluation variables.",
    )
    task_route: str = Field(
        default="",
        description="Structured route identified by the agentic controller.",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total execution latency in milliseconds.",
    )
