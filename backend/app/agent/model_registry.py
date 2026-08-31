"""
SatQuery AI — Specialist Model Registry.

Decouples the controller from internal model architectures and tracks
model metadata, readiness status, adapters, and input/output contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from app.models.fusion import run_module as run_fusion
from app.models.grounding import run_module as run_grounding
from app.models.change_vqa import run_module as run_change_vqa
from app.models.vqa import run_module as run_vqa
from app.models.captioning import run_module as run_captioning


@dataclass
class ModelEntry:
    task_name: str
    model_name: str
    version: str
    availability: str  # "READY" | "NOT_READY"
    adapter: Callable[..., dict[str, Any]]
    expected_input: str
    output_type: str
    description: str


class ModelRegistry:
    """Registry maintaining active and placeholder specialist remote sensing models."""

    def __init__(self):
        self._registry: Dict[str, ModelEntry] = {
            "fusion": ModelEntry(
                task_name="fusion",
                model_name="satquery-optical-sar-fusion-v1",
                version="1.0.0",
                availability="READY",
                adapter=run_fusion,
                expected_input="2 co-registered images (1 Optical + 1 SAR)",
                output_type="binary_or_mcq_or_bbox",
                description="Multi-task cross-modal Optical-SAR fusion model (trained on Sentinel-1 & 2).",
            ),
            "grounding": ModelEntry(
                task_name="grounding",
                model_name="satquery-region-grounding-v1",
                version="1.0.0",
                availability="READY",
                adapter=run_grounding,
                expected_input="1 satellite image + spatial text query",
                output_type="bounding_box",
                description="Spatial localization and region grounding network producing normalized bounding boxes.",
            ),
            "change_vqa": ModelEntry(
                task_name="change_vqa",
                model_name="satquery-change-vqa-v1",
                version="1.0.0",
                availability="READY",
                adapter=run_change_vqa,
                expected_input="2 bi-temporal satellite images (T1, T2) + query",
                output_type="change_classification",
                description="Bi-temporal change reasoning model analyzing multi-temporal environmental dynamics.",
            ),
            "vqa": ModelEntry(
                task_name="vqa",
                model_name="satquery-vlm-vqa-placeholder",
                version="0.1.0",
                availability="NOT_READY",
                adapter=run_vqa,
                expected_input="1 satellite image + natural-language question",
                output_type="text_answer",
                description="General Remote Sensing Vision-Language Model for open-ended VQA (placeholder).",
            ),
            "captioning": ModelEntry(
                task_name="captioning",
                model_name="satquery-vlm-caption-placeholder",
                version="0.1.0",
                availability="NOT_READY",
                adapter=run_captioning,
                expected_input="1 satellite image",
                output_type="text_description",
                description="Remote Sensing Vision-Language Model for scene description (placeholder).",
            ),
        }

    def get_entry(self, task: str) -> ModelEntry:
        """Retrieve model entry by canonical task name."""
        norm_task = task.lower().strip()
        # Aliases
        if norm_task in ("change", "change_detection"):
            norm_task = "change_vqa"
        elif norm_task in ("caption", "description"):
            norm_task = "captioning"
        elif norm_task in ("cross_modal_fusion", "optical_sar_fusion"):
            norm_task = "fusion"

        if norm_task not in self._registry:
            raise KeyError(f"Unknown task '{task}'. Available tasks: {list(self._registry.keys())}")
        return self._registry[norm_task]

    def get_adapter(self, task: str) -> Callable[..., dict[str, Any]]:
        """Get the callable adapter function for a task."""
        return self.get_entry(task).adapter

    def is_available(self, task: str) -> bool:
        """Check if a specialist model is currently READY."""
        return self.get_entry(task).availability == "READY"

    def get_status_dict(self) -> dict[str, str]:
        """Return status dictionary for all models (used in /health)."""
        return {
            "fusion": "ready" if self.is_available("fusion") else "not_ready",
            "grounding": "ready" if self.is_available("grounding") else "not_ready",
            "change_vqa": "ready" if self.is_available("change_vqa") else "not_ready",
            "vqa": "ready" if self.is_available("vqa") else "not_ready",
            "captioning": "ready" if self.is_available("captioning") else "not_ready",
        }

    def list_models(self) -> List[dict[str, Any]]:
        """List metadata for all registered models."""
        return [
            {
                "task": entry.task_name,
                "model_name": entry.model_name,
                "version": entry.version,
                "status": entry.availability,
                "expected_input": entry.expected_input,
                "output_type": entry.output_type,
                "description": entry.description,
            }
            for entry in self._registry.values()
        ]


# Singleton model registry instance
registry = ModelRegistry()
