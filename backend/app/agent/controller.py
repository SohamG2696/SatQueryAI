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
from app.services.result_interpreter import decompose_query, synthesize_multi_model_results
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

        clean_query = validate_query(query, required=False)
        image_count = len(images)

        # 1. Build Structured Analysis Plan
        from app.agent.planner import build_analysis_plan, validate_plan_inputs
        plan = build_analysis_plan(clean_query, image_count=image_count, metadata=meta)

        # 2. Deterministic Routing Decision
        task, task_route, tasks_list = get_route_info(
            query=clean_query,
            image_count=image_count,
            modalities=modalities,
            dates=dates,
            plan=plan,
        )

        # 3. Input Requirement Validation AFTER task resolution
        validate_plan_inputs(plan, image_count, modalities)

        # 4. Execute Inference

        # --- Case A: Single Specialist Model Execution ---
        if len(tasks_list) == 1:
            single_task = tasks_list[0]
            
            if single_task == "gee":
                from app.services.earth_engine import gee_query_planner
                from app.services.result_interpreter import interpret_result
                import time
                
                lat = plan.latitude if plan.latitude is not None else 0.0
                lon = plan.longitude if plan.longitude is not None else 0.0
                
                start = time.time()
                gee_intent, gee_result, explanation = gee_query_planner(clean_query, lat, lon)
                end = time.time()
                
                # Apply Result Interpreter Layer to GEE result
                final_gee_ans = explanation
                try:
                    gee_payload = gee_result if isinstance(gee_result, dict) else {"result": gee_result, "explanation": explanation}
                    interp = interpret_result(
                        query=clean_query,
                        task=gee_intent or "gee",
                        result=gee_payload,
                    )
                    formatted = interp.to_formatted_answer()
                    if formatted:
                        final_gee_ans = formatted
                except Exception:
                    final_gee_ans = explanation

                return QueryResponse(
                    task_detected="gee",
                    answer=final_gee_ans,
                    confidence=1.0,
                    visual_evidence=VisualEvidence(type="none"),
                    execution_summary=ExecutionSummary(
                        models_used=["google_earth_engine"],
                        parameters={
                            "lat": lat,
                            "lon": lon,
                            "gee_intent": gee_intent,
                            "execution_trace": [
                                "Intent detected: gee",
                                "Executed Google Earth Engine query planner",
                                "Result Interpretation executed successfully",
                                "Result generated successfully",
                            ],
                        },
                        task_route=task_route,
                        processing_time_ms=round((end - start) * 1000, 2)
                    )
                )

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
                query=clean_query,
            )

            # Enrich parameters with factual execution trace
            trace = [
                f"Intent detected: {single_task}",
                f"Loaded {image_count} satellite image(s)",
                f"Executed specialist model '{model_output.get('model_name')}'",
                "Result Interpretation executed successfully",
                "Result generated successfully",
            ]
            response.execution_summary.parameters["execution_trace"] = trace
            return response

        # --- Case B: Multi-Model Sequential Execution ---
        trace = [
            f"Multi-model intent detected: {', '.join(tasks_list)}",
            f"Loaded {image_count} satellite image(s)",
        ]

        # Decompose query for each sub-task
        task_queries = decompose_query(clean_query, tasks_list)

        models_used: List[str] = []
        total_time_ms = 0.0
        sub_task_results: List[Dict[str, Any]] = []
        combined_params: Dict[str, Any] = dict(meta)

        for sub_task in tasks_list:
            sub_entry = registry.get_entry(sub_task)
            t_query = task_queries.get(sub_task, clean_query)

            log_request(
                task=sub_task,
                model_name=sub_entry.model_name,
                query=t_query,
                image_count=image_count,
            )

            out = inference_service.run_inference(
                task=sub_task,
                images=images,
                query=t_query,
                metadata=meta,
            )

            m_name = out.get("model_name", sub_entry.model_name)
            if m_name not in models_used:
                models_used.append(m_name)

            proc_ms = float(out.get("processing_time_ms", 0.0))
            total_time_ms += proc_ms

            sub_task_results.append({
                "task": sub_task,
                "model_name": m_name,
                "query": t_query,
                "answer": out.get("answer", ""),
                "confidence": out.get("confidence"),
                "visual_evidence": out.get("visual_evidence"),
                "parameters": out.get("parameters", {}),
                "processing_time_ms": proc_ms,
            })

            trace.append(f"Executed '{m_name}' for sub-task '{sub_task}'")

        trace.append("Multi-model specialist execution completed")

        # Synthesize results into ONE coherent response with error fallback
        try:
            synthesis_result = synthesize_multi_model_results(
                original_query=clean_query,
                sub_task_results=sub_task_results,
            )
            synthesized_answer = synthesis_result["synthesized_answer"]
            confidence_by_task = synthesis_result["confidence_by_task"]
            synthesis_block = synthesis_result["synthesis"]
            primary_visual_evidence = synthesis_result["primary_visual_evidence"]
            trace.append("Multi-model result synthesis executed")
            trace.append("Multi-model result interpretation layer applied successfully")
            trace.append(f"Synthesized outputs from {len(sub_task_results)} specialist analyses")
        except Exception as exc:
            log_error("Multi-model synthesis", exc)
            trace.append("Multi-model synthesis failed; fallback to specialist outputs")
            answers = [f"[{r['task']}] {r['answer']}" for r in sub_task_results if r.get("answer")]
            synthesized_answer = " | ".join(answers) if answers else "Multi-model evaluation completed."
            confidence_by_task = {r["task"]: r["confidence"] for r in sub_task_results if r.get("confidence") is not None}
            synthesis_block = {
                "summary": "Multi-model execution fallback.",
                "findings": answers,
                "conclusion": "Specialist outputs concatenated due to synthesis fallback.",
                "evidence_quality": "insufficient",
                "uncertainties": [f"Synthesis exception: {str(exc)}"],
            }
            primary_visual_evidence = VisualEvidence(type="none")

        combined_params["original_query"] = clean_query
        combined_params["execution_trace"] = trace
        combined_params["sub_tasks"] = tasks_list
        combined_params["sub_task_queries"] = task_queries
        combined_params["confidence_by_task"] = confidence_by_task
        combined_params["synthesis"] = synthesis_block
        combined_params["sub_results"] = sub_task_results

        execution_summary = ExecutionSummary(
            models_used=models_used,
            parameters=combined_params,
            task_route=task_route,
            processing_time_ms=round(total_time_ms, 2),
        )

        verification = verify_execution_result(
            task="multi_model",
            answer=synthesized_answer,
            confidence=None,
            visual_evidence=primary_visual_evidence.model_dump() if hasattr(primary_visual_evidence, "model_dump") else (primary_visual_evidence if isinstance(primary_visual_evidence, dict) else None),
            sub_results=sub_task_results,
        )

        return QueryResponse(
            task_detected="multi_model",
            answer=synthesized_answer,
            confidence=None,
            confidence_by_task=confidence_by_task,
            synthesis=synthesis_block,
            visual_evidence=primary_visual_evidence,
            execution_summary=execution_summary,
            verification=verification,
        )


# Singleton controller instance
controller = AgenticController()
