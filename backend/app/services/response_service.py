"""
SatQuery AI — Response Assembler Service.

Normalizes specialist model outputs into the unified QueryResponse contract
with calibrated confidence scores, factual execution summaries, and visual evidence.
"""

from __future__ import annotations

from typing import Any, Dict

from app.schemas.execution import ExecutionSummary, VisualEvidence
from app.schemas.response import QueryResponse
from app.services.confidence_service import calibrate_confidence
from app.services.verification_service import verify_execution_result


def assemble_query_response(
    task: str,
    task_route: str,
    model_output: Dict[str, Any],
) -> QueryResponse:
    """Normalize and assemble model output into QueryResponse.

    Parameters
    ----------
    task : str
        Canonical task name ('vqa', 'captioning', 'grounding', 'change_vqa', 'fusion').
    task_route : str
        Structured route string (e.g. 'two_image_cross_modal_fusion').
    model_output : Dict[str, Any]
        Raw dictionary returned from the model adapter.

    Returns
    -------
    QueryResponse
        Fully assembled and validated Pydantic response.
    """
    raw_answer = model_output.get("answer", "")
    raw_confidence = model_output.get("confidence")
    model_name = model_output.get("model_name", f"satquery-{task}-model")
    raw_evidence = model_output.get("visual_evidence")
    proc_time_ms = float(model_output.get("processing_time_ms", 0.0))
    params = dict(model_output.get("parameters", {}))

    # 1. Calibrate Confidence
    calibrated_conf, _ = calibrate_confidence(raw_confidence, task=task)

    # 2. Assemble Visual Evidence
    visual_evidence = None
    if isinstance(raw_evidence, dict):
        visual_evidence = VisualEvidence(
            type=raw_evidence.get("type", "none"),
            coordinates=raw_evidence.get("coordinates"),
            coordinate_system=raw_evidence.get("coordinate_system", "normalized"),
            data=raw_evidence.get("data"),
        )
    elif raw_evidence is None:
        visual_evidence = VisualEvidence(type="none")

    # 3. Evaluate Verification Status
    verification = verify_execution_result(
        task=task,
        answer=raw_answer,
        confidence=calibrated_conf,
        visual_evidence=raw_evidence if isinstance(raw_evidence, dict) else None,
    )
    params["verification"] = verification.model_dump()

    # 4. Assemble Execution Summary
    execution_summary = ExecutionSummary(
        models_used=[model_name],
        parameters=params,
        task_route=task_route,
        processing_time_ms=proc_time_ms,
    )

    return QueryResponse(
        task_detected=task,
        answer=raw_answer,
        confidence=calibrated_conf,
        visual_evidence=visual_evidence,
        execution_summary=execution_summary,
        verification=verification,
    )
