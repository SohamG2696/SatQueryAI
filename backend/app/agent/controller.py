"""
SatQuery AI — Agentic Controller.

The central orchestrator of SatQuery AI:
1. Validates inputs and image integrity
2. Evaluates query intent and multi-modal properties (image count, modalities, timestamps)
3. Routes deterministically to the appropriate specialist workflow
4. Invokes the cached model adapter via InferenceService
5. Normalizes and returns a structured, evidence-grounded QueryResponse
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.agent.model_registry import registry
from app.agent.router import route_request
from app.schemas.response import QueryResponse
from app.services.inference_service import inference_service
from app.services.response_service import assemble_query_response
from app.utils.logging import log_request, log_error
from app.utils.validators import validate_query, validate_image_count


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

        # 1. Validation
        clean_query = validate_query(query, required=False)
        image_count = len(images)

        # 2. Routing Decision
        task, task_route = route_request(
            query=clean_query,
            image_count=image_count,
            modalities=modalities,
            dates=dates,
        )

        validate_image_count(image_count, task)
        model_entry = registry.get_entry(task)

        # 3. Log incoming request
        log_request(
            task=task,
            model_name=model_entry.model_name,
            query=clean_query,
            image_count=image_count,
        )

        # 4. Model Inference via Cached Service
        model_output = inference_service.run_inference(
            task=task,
            images=images,
            query=clean_query,
            metadata=meta,
        )

        # 5. Response Assembly
        response = assemble_query_response(
            task=task,
            task_route=task_route,
            model_output=model_output,
        )

        return response


# Singleton controller instance
controller = AgenticController()
