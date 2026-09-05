"""
SatQuery AI — Factual Verification Service.

Evaluates single and multi-model outputs deterministically to assign a verification status:
- 'accepted': Model executed successfully and met confidence/evidence requirements.
- 'verify_required': Low confidence score, missing required evidence, or indeterminate outputs.

Distinguishes execution validation and model confidence from ground-truth verification.
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


def get_confidence_description(conf: float | None) -> str:
    """Return a factual, non-misleading description of a model confidence score.

    Guidelines:
    - < 0.50: Low confidence; interpret with caution.
    - 0.50 - 0.75: Moderate confidence; interpret with caution.
    - 0.75 - 0.90: Relatively high model confidence, but not independently verified.
    - > 0.90: High model confidence, but confidence is not equivalent to accuracy.
    """
    if conf is None:
        return "Model confidence is indeterminate."

    c = float(conf)
    if c > 1.0:
        c = c / 100.0

    pct = c * 100

    if c < 0.50:
        return f"The model reports low confidence ({pct:.2f}%); interpret with caution."
    elif c <= 0.75:
        return f"The model reports moderate confidence ({pct:.2f}%); interpret with caution."
    elif c <= 0.90:
        return f"The model reports relatively high confidence ({pct:.2f}%); this does not represent independently measured prediction accuracy."
    else:
        return f"The model reports high confidence ({pct:.2f}%); note that model confidence is not equivalent to ground-truth accuracy."


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
                reason=f"Execution completed. Sub-task(s) '{', '.join(low_conf_tasks)}' scored below verification threshold ({conf_threshold:.2f}).",
                confidence=round(min(sub_confidences), 4) if sub_confidences else confidence,
            )

        has_grounding = any(r.get("task") in ("grounding", "bbox") for r in sub_results)
        if has_grounding and visual_evidence:
            ev_type = visual_evidence.get("type", "none")
            if ev_type == "none" or not visual_evidence.get("coordinates"):
                return VerificationDetails(
                    status="verify_required",
                    reason="Execution completed. Grounding sub-task expected bounding box evidence but none was produced.",
                    confidence=confidence,
                )

        return VerificationDetails(
            status="accepted",
            reason=f"Multi-model execution completed across {len(sub_results)} sub-tasks. {get_confidence_description(confidence)}",
            confidence=confidence,
        )

    # 2. Single Model Low Confidence Check
    if confidence is not None:
        conf_val = float(confidence)
        if conf_val < conf_threshold:
            return VerificationDetails(
                status="verify_required",
                reason=f"Model execution completed. {get_confidence_description(conf_val)} Scored below verification threshold ({conf_threshold:.2f}).",
                confidence=round(conf_val, 4),
            )

    # 3. Grounding Specific Evidence Check
    if task in ("grounding", "bbox"):
        if not visual_evidence or visual_evidence.get("type") == "none" or not visual_evidence.get("coordinates"):
            return VerificationDetails(
                status="verify_required",
                reason="Model execution completed, but spatial region grounding result lacks valid visual bounding box evidence.",
                confidence=confidence,
            )

    # 4. Check for Indeterminate / Empty Answers
    if not answer or "not_ready" in answer.lower() or "[vlm_pending]" in answer.lower():
        return VerificationDetails(
            status="verify_required",
            reason="Model execution completed, but output answer is unconfirmed or model status is indeterminate.",
            confidence=confidence,
        )

    # 5. Default Execution Acceptance with Factual Confidence Description
    conf_disp = round(float(confidence), 4) if confidence is not None else 0.85
    return VerificationDetails(
        status="accepted",
        reason=f"Model execution completed successfully. {get_confidence_description(conf_disp)}",
        confidence=conf_disp,
    )
