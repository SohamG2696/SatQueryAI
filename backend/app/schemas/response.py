"""
SatQuery AI — Standardized Response Schemas.

Defines the unified QueryResponse contract consumed by the React frontend,
ensuring consistent rendering across VQA, Captioning, Grounding, Change Analysis, and Fusion.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field

from .execution import ExecutionSummary, VisualEvidence
from app.services.verification_service import VerificationDetails


class QueryResponse(BaseModel):
    """Primary unified response returned by the POST /api/query endpoint."""

    task_detected: str = Field(
        ...,
        description="Task identified by agent controller: 'vqa', 'captioning', 'scene_description', 'grounding', 'change_vqa', 'fusion', 'gee', 'multi_model', 'invalid'.",
    )
    valid: Optional[bool] = Field(
        default=True,
        description="Indicates whether the user query is a valid supported remote sensing task.",
    )
    canonical_task: Optional[str] = Field(
        default=None,
        description="Normalized canonical task name ('scene_description', 'grounding', 'vqa', 'change_vqa', 'fusion', etc.).",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Detected user intent category.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation or justification when a query is invalid or routed.",
    )
    target: Optional[str] = Field(
        default=None,
        description="Extracted spatial grounding or analysis target entity.",
    )
    operation: Optional[str] = Field(
        default=None,
        description="Extracted operation type ('locate', 'scene_description', 'index_calculation', etc.).",
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
    confidence_by_task: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-task confidence mapping for multi-model execution.",
    )
    synthesis: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured multi-model synthesis summary, findings, and evidence quality.",
    )
    visual_evidence: Optional[VisualEvidence] = Field(
        default=None,
        description="Visual grounding evidence (bounding boxes, change maps, masks).",
    )
    execution_summary: ExecutionSummary = Field(
        ...,
        description="Factual execution trace including models used, route, and processing latency.",
    )
    verification: Optional[VerificationDetails] = Field(
        default=None,
        description="Verification layer metadata ('accepted' or 'verify_required').",
    )


class AnalysisResponse(BaseModel):
    """Unified response maintained for backward compatibility across legacy specialist routes."""

    success: bool = True
    task: str
    answer: str = ""
    confidence: Optional[float] = None
    models_used: List[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    visual_evidence: Optional[VisualEvidence] = None
    evidence: List[str] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    processing_time: float = 0.0
