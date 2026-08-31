"""
SatQuery AI — Confidence Calibration Service.

Provides defensible confidence scoring and threshold checks for model outputs.
"""

from __future__ import annotations

from typing import Any

from app.config import settings


def calibrate_confidence(
    raw_confidence: float | None,
    task: str,
    threshold: float | None = None,
) -> Tuple[float | None, bool]:
    """Validate and calibrate model confidence score.

    Parameters
    ----------
    raw_confidence : float | None
        Raw model probability or confidence output.
    task : str
        The task being executed.
    threshold : float | None
        Optional threshold override (defaults to settings.confidence_threshold).

    Returns
    -------
    Tuple[float | None, bool]
        (calibrated_confidence, is_above_threshold)
    """
    thresh = threshold if threshold is not None else settings.confidence_threshold

    if raw_confidence is None:
        return None, True

    # Clamp to valid [0.0, 1.0] range
    conf = max(0.0, min(1.0, float(raw_confidence)))
    is_confident = (conf >= thresh)

    return round(conf, 4), is_confident
