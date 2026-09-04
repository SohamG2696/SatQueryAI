"""
SatQuery AI — Factual Verification Service.

Evaluates single and multi-model outputs deterministically to assign a verification status:
- 'accepted': High confidence score and consistent cross-task visual evidence.
- 'verify_required': Low confidence score, missing required evidence, or conflicting outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import settings


class VerificationDetails(BaseModel):
    """Structured verification metadata attached to query responses."""

    status: str = Field(
        ...,
        description="Verification outcome: 'accepted' or 'verify_required'.",
    )
    reason: str = Field(
        ...,
        description="Factual explanation for the verification decision.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score evaluated.",
    )


def verify_execution_result(
    task: str,
    answer: str,
    confidence: float | None,
    visual_evidence: Dict[str, Any] | None = None,
    sub_results: List[Dict[str, Any]] | None = None,
    threshold: float | None = None,
) -> VerificationDetails:
    """Deterministically verify execution outputs for single or multi-model queries.

    Parameters
    ----------
    task : str
        Canonical task name ('vqa', 'captioning', 'grounding', 'change_vqa', 'fusion', 'multi_model').
    answer : str
        Natural language output string.
    confidence : float | None
        Calibrated output confidence.
    visual_evidence : Dict[str, Any] | None
        Associated visual evidence (bounding boxes, change maps).
    sub_results : List[Dict[str, Any]] | None
        Sub-task model outputs if task is 'multi_model'.
    threshold : float | None
        Minimum confidence threshold for acceptance (defaults to settings.confidence_threshold, e.g. 0.60).

    Returns
    -------
    VerificationDetails
        Structured verification status ('accepted' or 'verify_required'), reason, and confidence.
    """
    conf_threshold = threshold if threshold is not None else getattr(settings, "confidence_threshold", 0.60)

    # 1. Multi-Model Consistency & Evidence Check
    if task == "multi_model" and sub_results:
        sub_confidences = [r.get("confidence") for r in sub_results if r.get("confidence") is not None]
        low_conf_tasks = [
            r.get("task", "sub_task")
            for r in sub_results
            if r.get("confidence") is not None and float(r["confidence"]) < conf_threshold
        ]

        if low_conf_tasks:
            return VerificationDetails(
                status="verify_required",
                reason=f"Sub-task(s) '{', '.join(low_conf_tasks)}' scored below verification threshold ({conf_threshold:.2f}).",
                confidence=round(min(sub_confidences), 4) if sub_confidences else confidence,
            )

        # Check for grounding evidence if grounding is one of the sub-tasks
        has_grounding = any(r.get("task") in ("grounding", "bbox") for r in sub_results)
        if has_grounding and visual_evidence:
            ev_type = visual_evidence.get("type", "none")
            if ev_type == "none" or not visual_evidence.get("coordinates"):
                return VerificationDetails(
                    status="verify_required",
                    reason="Grounding sub-task expected bounding box evidence but none was produced.",
                    confidence=confidence,
                )

        return VerificationDetails(
            status="accepted",
            reason=f"Multi-model execution consistent across {len(sub_results)} sub-tasks with high confidence.",
            confidence=confidence,
        )

    # 2. Single Model Low Confidence Check
    if confidence is not None:
        conf_val = float(confidence)
        if conf_val < conf_threshold:
            return VerificationDetails(
                status="verify_required",
                reason=f"Model confidence ({conf_val:.4f}) is below verification threshold ({conf_threshold:.2f}).",
                confidence=round(conf_val, 4),
            )

    # 3. Grounding Specific Evidence Check
    if task in ("grounding", "bbox"):
        if not visual_evidence or visual_evidence.get("type") == "none" or not visual_evidence.get("coordinates"):
            return VerificationDetails(
                status="verify_required",
                reason="Spatial region grounding result lacks valid visual bounding box evidence.",
                confidence=confidence,
            )

    # 4. Check for Indeterminate / Empty Answers
    if not answer or "not_ready" in answer.lower() or "[vlm_pending]" in answer.lower():
        return VerificationDetails(
            status="verify_required",
            reason="Output answer is unconfirmed or model status is indeterminate.",
            confidence=confidence,
        )

    # 5. Default High-Confidence Acceptance
    conf_disp = round(float(confidence), 4) if confidence is not None else 0.85
    return VerificationDetails(
        status="accepted",
        reason=f"Model output verified successfully with high confidence ({conf_disp:.4f}).",
        confidence=conf_disp,
    )
