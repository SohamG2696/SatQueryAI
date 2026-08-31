"""
SatQuery AI — Inference Execution Service.

Singleton service managing model caching in memory, warm-up, and execution timing.
Avoids reloading heavy PyTorch weights on every API call.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch

from app.agent.model_registry import registry
from app.utils.device import get_device, sync_device
from app.utils.logging import log_inference, log_error


class InferenceService:
    """Singleton inference coordinator with model caching."""

    def __init__(self):
        self.device = get_device()
        self._warmed_up = False

    def warm_up(self) -> None:
        """Pre-load ready model weights on server startup."""
        if self._warmed_up:
            return
        try:
            from app.models.fusion import get_fusion_engine
            get_fusion_engine()

            from app.models.grounding import get_grounding_engine
            get_grounding_engine()

            from app.models.change_vqa import get_change_vqa_engine
            get_change_vqa_engine()

            self._warmed_up = True
        except Exception as e:
            log_error("InferenceService.warm_up", e)

    @torch.no_grad()
    def run_inference(
        self,
        task: str,
        images: List[Any],
        query: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Execute specialist model inference with performance timing.

        Parameters
        ----------
        task : str
            Canonical task name (e.g. 'fusion', 'grounding', 'change_vqa', 'vqa', 'captioning').
        images : List[Any]
            Loaded image arrays, tensors, or file references.
        query : str
            Query string.
        metadata : Dict[str, Any] | None
            Task-specific metadata.

        Returns
        -------
        Dict[str, Any]
            Standard model adapter result enriched with processing_time_ms.
        """
        adapter = registry.get_adapter(task)
        model_entry = registry.get_entry(task)

        sync_device(self.device)
        start_time = time.perf_counter()

        try:
            result = adapter(images=images, query=query, metadata=metadata)
        except Exception as exc:
            log_error(f"Inference execution for task '{task}'", exc)
            raise exc

        sync_device(self.device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        result["processing_time_ms"] = round(elapsed_ms, 2)
        if "model_name" not in result:
            result["model_name"] = model_entry.model_name

        log_inference(
            model_name=result["model_name"],
            duration_ms=elapsed_ms,
            confidence=result.get("confidence"),
        )

        return result


# Singleton inference service
inference_service = InferenceService()
