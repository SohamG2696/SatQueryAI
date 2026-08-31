"""
SatQuery AI — Standardized Response Schemas.

Defines the unified QueryResponse contract consumed by the React frontend,
ensuring consistent rendering across VQA, Captioning, Grounding, Change Analysis, and Fusion.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from .execution import ExecutionSummary, VisualEvidence
from .upload import UploadResponse


class QueryResponse(BaseModel):
    """Primary unified response returned by the POST /api/query endpoint."""

    task_detected: str = Field(
        ...,
        description="Task identified by agent controller: 'vqa', 'captioning', 'grounding', 'change_vqa', 'fusion'.",
    )
    answer: str = Field(
        ...,
        description="Factual natural-language answer, prediction, or description.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Defensible model confidence score (0.0 - 1.0) or null if indeterminate.",
    )
    visual_evidence: Optional[VisualEvidence] = Field(
        default=None,
        description="Visual grounding evidence (bounding boxes, change maps, masks).",
    )
    execution_summary: ExecutionSummary = Field(
        ...,
        description="Factual execution trace including models used, route, and processing latency.",
    )


class AnalysisResponse(BaseModel):
    """Unified response maintained for backward compatibility across legacy specialist routes."""

    success: bool = True
    task: str
    answer: str = ""
    confidence: Optional[float] = None
    models_used: List[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    processing_time: float = 0.0
