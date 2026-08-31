"""
SatQuery AI — Input and Security Validation Utility.

Validates filenames, file sizes, image compatibility, query constraints,
and prevents directory traversal vulnerabilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from fastapi import HTTPException

from app.config import settings

# Additional allowed formats for internal and test execution
_TEST_FORMATS = {".npz"}


def validate_file_extension(filename: str) -> str:
    """Ensure the file has an allowed remote sensing or image extension."""
    suffix = Path(filename).suffix.lower()
    allowed = set(settings.allowed_extensions_list) | _TEST_FORMATS

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{suffix}'. "
                f"Allowed formats: {', '.join(sorted(allowed))}"
            ),
        )
    return suffix


def validate_file_size(size_bytes: int) -> None:
    """Ensure the file size is within the allowed limits."""
    if size_bytes <= 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes).",
        )

    if size_bytes > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size ({size_bytes / (1024 * 1024):.2f} MB) exceeds maximum "
                f"limit of {settings.max_upload_size_mb} MB."
            ),
        )


def validate_query(query: str | None, required: bool = True) -> str:
    """Validate query text."""
    clean_query = (query or "").strip()
    if required and not clean_query:
        raise HTTPException(
            status_code=400,
            detail="A non-empty query string is required for this operation.",
        )
    if len(clean_query) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Query string is too long (maximum 1,000 characters).",
        )
    return clean_query


def validate_image_count(image_count: int, task: str) -> None:
    """Ensure the number of images matches the task requirement."""
    if image_count < 1:
        raise HTTPException(
            status_code=400,
            detail="At least one satellite image is required.",
        )

    if task in ("change", "change_vqa", "fusion") and image_count < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{task}' requires exactly two satellite images.",
        )

    if task in ("vqa", "caption", "captioning", "grounding") and image_count > 1:
        # Multi-image for single image tasks - allowable, but warning / restriction
        pass


def sanitize_filename(filename: str) -> str:
    """Sanitize filename and strip potential path traversal components."""
    clean_name = os.path.basename(filename)
    clean_name = "".join(c for c in clean_name if c.isalnum() or c in (".", "_", "-"))
    return clean_name or "uploaded_image"
