"""
SatQuery AI — Agentic Controller.

The central orchestrator of SatQuery AI:
1. Validates inputs and image integrity
2. Evaluates query intent and multi-modal properties
3. Routes deterministically to specialist model workflows (single or multi-model)
4. Invokes cached model adapters via InferenceService
5. Normalizes and returns a structured, evidence-grounded QueryResponse
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.agent.model_registry import registry
from app.agent.router import get_route_info
from app.schemas.execution import ExecutionSummary, VisualEvidence
from app.schemas.response import QueryResponse
from app.services.inference_service import inference_service
from app.services.response_service import assemble_query_response
from app.services.verification_service import verify_execution_result
from app.utils.logging import log_request, log_error
from app.utils.validators import validate_query


class AgenticController:
    """Central agent controller coordinating specialist remote sensing workflows."""

    def process_query(
        self,
        images: List[Any],
        query: str,
        metadata: Dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Process an incoming multi-modal query end-to-end.

        Parameters
        ----------
        images : List[Any]
            List of 1 or 2 satellite image sources (file paths, arrays, bytes).
        query : str
            User query or question.
        metadata : Dict[str, Any] | None
            Optional metadata (modalities, dates, parameters).

        Returns
        -------
        QueryResponse
            Standardized response.
        """
        meta = metadata or {}
        modalities = meta.get("modalities")
        dates = meta.get("dates")

        # 1. Input Validation
        clean_query = validate_query(query, required=False)
        image_count = len(images)

        if image_count < 1:
            raise ValueError("At least one satellite image is required.")

        # 2. Routing Decision
        task, task_route, tasks_list = get_route_info(
            query=clean_query,
            image_count=image_count,
            modalities=modalities,
            dates=dates,
        )

        # 3. Validate image requirements per task
        for t in tasks_list:
            if t in ("change_vqa", "change") and image_count < 2:
                raise ValueError("Bi-temporal change detection requires at least 2 images (before and after).")
            if t in ("fusion", "cross_modal") and image_count < 2:
                raise ValueError("Optical-SAR cross-modal fusion requires at least 2 images (optical and SAR).")

        # 4. Execute Inference

        # --- Case A: Single Specialist Model Execution ---
        if len(tasks_list) == 1:
            single_task = tasks_list[0]
            model_entry = registry.get_entry(single_task)

            log_request(
                task=single_task,
                model_name=model_entry.model_name,
                query=clean_query,
                image_count=image_count,
            )

            model_output = inference_service.run_inference(
                task=single_task,
                images=images,
                query=clean_query,
                metadata=meta,
            )

            response = assemble_query_response(
                task=single_task,
                task_route=task_route,
                model_output=model_output,
            )

            # Enrich parameters with factual execution trace
            trace = [
                f"Intent detected: {single_task}",
                f"Loaded {image_count} satellite image(s)",
                f"Executed specialist model '{model_output.get('model_name')}'",
                "Result generated successfully",
            ]
            response.execution_summary.parameters["execution_trace"] = trace
            return response

        # --- Case B: Multi-Model Sequential Execution ---
        trace = [
            f"Multi-model intent detected: {', '.join(tasks_list)}",
            f"Loaded {image_count} satellite image(s)",
        ]

        models_used: List[str] = []
        total_time_ms = 0.0
        confidences: List[float] = []
        answers: List[str] = []
        evidences: List[Dict[str, Any]] = []
        sub_results: List[Dict[str, Any]] = []
        combined_params: Dict[str, Any] = dict(meta)

        for sub_task in tasks_list:
            sub_entry = registry.get_entry(sub_task)

            log_request(
                task=sub_task,
                model_name=sub_entry.model_name,
                query=clean_query,
                image_count=image_count,
            )

            out = inference_service.run_inference(
                task=sub_task,
                images=images,
                query=clean_query,
                metadata=meta,
            )

            sub_out = dict(out)
            sub_out["task"] = sub_task
            sub_results.append(sub_out)

            m_name = out.get("model_name", sub_entry.model_name)
            if m_name not in models_used:
                models_used.append(m_name)

            total_time_ms += float(out.get("processing_time_ms", 0.0))
            if out.get("confidence") is not None:
                confidences.append(float(out["confidence"]))

            ans = out.get("answer", "")
            if ans:
                answers.append(f"[{sub_task}] {ans}")

            ev = out.get("visual_evidence")
            if isinstance(ev, dict) and ev.get("type") != "none":
                evidences.append(ev)

            trace.append(f"Executed '{m_name}' for sub-task '{sub_task}'")

        combined_answer = " | ".join(answers) if answers else "Multi-model evaluation completed."
        combined_conf = round(min(confidences), 4) if confidences else 0.85

        primary_evidence = VisualEvidence(type="none")
        if evidences:
            first_ev = evidences[0]
            primary_evidence = VisualEvidence(
                type=first_ev.get("type", "none"),
                coordinates=first_ev.get("coordinates"),
                coordinate_system=first_ev.get("coordinate_system", "normalized"),
                data=first_ev.get("data"),
            )

        # Evaluate Multi-Model Verification
        verification = verify_execution_result(
            task="multi_model",
            answer=combined_answer,
            confidence=combined_conf,
            visual_evidence=primary_evidence.model_dump() if primary_evidence else None,
            sub_results=sub_results,
        )

        trace.append("Combined multi-model outputs successfully")
        combined_params["execution_trace"] = trace
        combined_params["sub_tasks"] = tasks_list
        combined_params["verification"] = verification.model_dump()

        execution_summary = ExecutionSummary(
            models_used=models_used,
            parameters=combined_params,
            task_route=task_route,
            processing_time_ms=round(total_time_ms, 2),
        )

        return QueryResponse(
            task_detected="multi_model",
            answer=combined_answer,
            confidence=combined_conf,
            visual_evidence=primary_evidence,
            execution_summary=execution_summary,
            verification=verification,
        )


# Singleton controller instance
controller = AgenticController()
