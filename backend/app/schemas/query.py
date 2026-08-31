"""
SatQuery AI — Query Request Schemas.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class QueryMetadata(BaseModel):
    """Optional metadata payload passed alongside query requests."""

    modalities: Optional[List[str]] = Field(
        default=None,
        description="Explicit modality tags per image: e.g. ['optical'], ['optical', 'sar'].",
    )
    dates: Optional[List[str]] = Field(
        default=None,
        description="Acquisition timestamps per image in ISO format (for change detection).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for history tracking.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution overrides or parameters.",
    )
