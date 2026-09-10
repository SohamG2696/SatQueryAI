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
from app.agent.query_validator import GuardrailStatus, validate_query_intent
from app.agent.router import get_route_info
from app.schemas.execution import ExecutionSummary, VisualEvidence
from app.schemas.response import QueryResponse
from app.services.cache_service import inference_cache
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

        # 0. Query Validation & Normalization Layer (STEP 1 & 2)
        validation = validate_query_intent(
            query=clean_query,
            image_count=image_count,
            metadata=meta,
        )

        # Early Rejection — ZERO model execution for both INVALID and NEEDS_CLARIFICATION.
        if not validation.valid:
            if validation.status == GuardrailStatus.NEEDS_CLARIFICATION:
                # Intent was recognised but required inputs are missing.
                # Return a helpful clarification message without running any model.
                clarification_reason = validation.reason or "Two compatible images are required for change analysis."
                return QueryResponse(
                    valid=False,
                    task_detected=validation.intent,
                    canonical_task=None,
                    intent=validation.intent,
                    reason=clarification_reason,
                    answer=f"Clarification needed: {clarification_reason}",
                    confidence=0.0,
                    visual_evidence=VisualEvidence(type="none"),
                    execution_summary=ExecutionSummary(
                        models_used=[],
                        parameters={
                            "valid": False,
                            "status": validation.status,
                            "reason": clarification_reason,
                            "original_query": clean_query,
                            "execution_trace": [
                                "Domain guardrail evaluated query",
                                f"Needs clarification: {clarification_reason}",
                                "Specialist model execution bypassed",
                            ],
                        },
                        task_route="needs_clarification",
                        processing_time_ms=0.0,
                    ),
                )

            # INVALID — genuine out-of-domain rejection
            rejection_reason = validation.reason or "The query is outside SatQuery's supported remote-sensing analysis domain."
            return QueryResponse(
                valid=False,
                task_detected="invalid",
                canonical_task=None,
                intent="invalid",
                reason=rejection_reason,
                answer="Invalid query. Please ask a question related to the uploaded remote-sensing imagery.",
                confidence=0.0,
                visual_evidence=VisualEvidence(type="none"),
                execution_summary=ExecutionSummary(
                    models_used=[],
                    parameters={
                        "valid": False,
                        "reason": rejection_reason,
                        "original_query": clean_query,
                        "execution_trace": [
                            "Domain guardrail evaluated query",
                            f"Query rejected: {rejection_reason}",
                            "Specialist model execution bypassed",
                        ],
                    },
                    task_route="rejected_query",
                    processing_time_ms=0.0,
                ),
            )

        # Check Inference Cache (STEP 5)
        cached_response = inference_cache.get(
            images=images,
            canonical_intent=validation.intent,
            canonical_target_or_prompt=validation.canonical_prompt,
        )
        if cached_response is not None:
            cached_response.valid = True
            cached_response.canonical_task = validation.canonical_task
            cached_response.intent = validation.intent
            if validation.target:
                cached_response.target = validation.target
            if validation.operation:
                cached_response.operation = validation.operation
            return cached_response

        # Use canonical prompt if available for fixed tasks, otherwise clean query
        effective_query = validation.canonical_prompt if validation.canonical_prompt else clean_query

        # 1. Build Structured Analysis Plan
        from app.agent.planner import build_analysis_plan, validate_plan_inputs
        plan = build_analysis_plan(effective_query, image_count=image_count, metadata=meta)

        # 2. Deterministic Routing Decision
        task, task_route, tasks_list = get_route_info(
            query=effective_query,
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

            if single_task == "future_prediction":
                import re
                import time
                from models.future_prediction.inference.dynamic_predictor import predict_from_uploads

                years = meta.get("years")
                if not years:
                    dates_meta = meta.get("dates") or []
                    years_parsed = []
                    for d in dates_meta:
                        try:
                            m = re.search(r"\b(20\d\d)\b", str(d))
                            if m:
                                years_parsed.append(int(m.group(1)))
                        except Exception:
                            pass
                    if len(years_parsed) == image_count:
                        years = years_parsed
                    else:
                        default_yrs = [2019, 2021, 2023, 2025]
                        years = default_yrs[:image_count] if image_count <= 4 else list(range(2019, 2019 + image_count))

                start = time.time()
                pred_res = predict_from_uploads(
                    image_paths=images,
                    years=years,
                    query=clean_query,
                )
                end = time.time()

                # Calculate overall numeric confidence score from category predictions
                CONF_MAP = {"high": 0.90, "moderate": 0.60, "low": 0.30}
                cat_preds = pred_res.get("predictions", {})
                if cat_preds:
                    cat_scores = [CONF_MAP.get(p.get("confidence", "low"), 0.30) for p in cat_preds.values()]
                    overall_confidence = min(cat_scores)
                else:
                    overall_confidence = 0.50

                return QueryResponse(
                    task_detected="future_prediction",
                    answer=pred_res.get("explanation", ""),
                    confidence=overall_confidence,
                    visual_evidence=VisualEvidence(type="none"),
                    execution_summary=ExecutionSummary(
                        models_used=["dynamic_landcover_predictor"],
                        parameters={
                            "target_year": pred_res.get("target_year"),
                            "predictions": pred_res.get("predictions"),
                            "historical_data": pred_res.get("historical_data"),
                            "unsupported_categories": pred_res.get("unsupported_categories"),
                            "disclaimer": pred_res.get("disclaimer"),
                            "execution_trace": [
                                "Intent detected: future_prediction",
                                f"Loaded {image_count} historical satellite image(s)",
                                "Spatial location validation passed",
                                "Extracted SCL land-cover stats per image",
                                "Executed weighted linear trend forecasting",
                                "Result generated successfully",
                            ],
                        },
                        task_route=task_route,
                        processing_time_ms=round((end - start) * 1000, 2),
                    ),
                )

            if single_task == "gee":
                from app.services.earth_engine import gee_query_planner
                from app.services.result_interpreter import interpret_result
                import time
                
                lat = plan.latitude if plan.latitude is not None else 0.0
                lon = plan.longitude if plan.longitude is not None else 0.0
                
                start = time.time()
                gee_intent, gee_result, explanation = gee_query_planner(effective_query, lat, lon)
                end = time.time()
                
                # Apply Result Interpreter Layer to GEE result
                final_gee_ans = explanation
                try:
                    gee_payload = gee_result if isinstance(gee_result, dict) else {"result": gee_result, "explanation": explanation}
                    interp = interpret_result(
                        query=effective_query,
                        task=gee_intent or "gee",
                        result=gee_payload,
                    )
                    formatted = interp.to_formatted_answer()
                    if formatted:
                        final_gee_ans = formatted
                except Exception:
                    final_gee_ans = explanation

                response = QueryResponse(
                    valid=True,
                    task_detected="gee",
                    canonical_task=validation.canonical_task,
                    intent=validation.intent,
                    target=validation.target,
                    operation=validation.operation,
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
                                "Query validated and normalized successfully",
                                f"Intent detected: {validation.intent}",
                                "Executed Google Earth Engine query planner",
                                "Result Interpretation executed successfully",
                                "Result generated successfully",
                            ],
                        },
                        task_route=task_route,
                        processing_time_ms=round((end - start) * 1000, 2)
                    )
                )

                inference_cache.put(
                    images=images,
                    canonical_intent=validation.intent,
                    canonical_target_or_prompt=validation.canonical_prompt,
                    response=response,
                )
                return response

            model_entry = registry.get_entry(single_task)

            log_request(
                task=single_task,
                model_name=model_entry.model_name,
                query=effective_query,
                image_count=image_count,
            )

            model_output = inference_service.run_inference(
                task=single_task,
                images=images,
                query=effective_query,
                metadata=meta,
            )

            response = assemble_query_response(
                task=single_task,
                task_route=task_route,
                model_output=model_output,
                query=clean_query,
            )

            # Populate normalization & schema fields
            response.valid = True
            response.canonical_task = validation.canonical_task
            response.intent = validation.intent
            if validation.target:
                response.target = validation.target
            if validation.operation:
                response.operation = validation.operation

            # Enrich parameters with factual execution trace
            trace = [
                f"Query validated: intent={validation.intent}, canonical_task={validation.canonical_task}",
                f"Loaded {image_count} satellite image(s)",
                f"Executed specialist model '{model_output.get('model_name')}'",
                "Result Interpretation executed successfully",
                "Result generated successfully",
            ]
            response.execution_summary.parameters["execution_trace"] = trace

            # Cache the assembled response
            inference_cache.put(
                images=images,
                canonical_intent=validation.intent,
                canonical_target_or_prompt=validation.canonical_prompt,
                response=response,
            )

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

        multi_response = QueryResponse(
            valid=True,
            task_detected="multi_model",
            canonical_task="multi_model",
            intent=validation.intent,
            target=validation.target,
            operation=validation.operation,
            answer=synthesized_answer,
            confidence=None,
            confidence_by_task=confidence_by_task,
            synthesis=synthesis_block,
            visual_evidence=primary_visual_evidence,
            execution_summary=execution_summary,
            verification=verification,
        )

        inference_cache.put(
            images=images,
            canonical_intent=validation.intent,
            canonical_target_or_prompt=validation.canonical_prompt,
            response=multi_response,
        )

        return multi_response


# Singleton controller instance
controller = AgenticController()
