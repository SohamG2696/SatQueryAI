"""
RSCoVLM Answer Synthesis Service.
"""

import logging
from typing import Any, Dict, Optional, Union

from PIL import Image

from models.vlm.rscovlm.rscovlm_config import RSCoVLMConfig
from models.vlm.rscovlm.rscovlm_utils import validate_rscovlm_response

logger = logging.getLogger("satquery.rscovlm.answer_service")


class RSCoVLMAnswerService:
    """Detailed reasoning and response synthesis service powered by RSCoVLM."""

    def __init__(self, config: Optional[RSCoVLMConfig] = None):
        self.config = config or RSCoVLMConfig()

    def generate_answer(
        self,
        predict_fn,
        image: Union[str, Image.Image],
        question: str,
        specialist_evidence: Optional[Dict[str, Any]] = None,
        task: str = "vqa",
    ) -> Dict[str, Any]:
        """
        Generate detailed, evidence-grounded response using RSCoVLM.
        Passes original question directly to predict_fn without double prompt formatting.
        """
        try:
            res = predict_fn(
                image=image,
                question=question,
                specialist_evidence=specialist_evidence,
                task=task,
            )
            if isinstance(res, dict):
                raw_pred = res.get("raw_prediction", res.get("prediction", res.get("answer", "")))
                answer_text = res.get("answer", res.get("prediction", raw_pred))
                confidence = res.get("confidence", 0.95)
                model_name = res.get("model_name", res.get("model", self.config.model_name))
                vlm_type = res.get("vlm_type", "RSCoVLM")
                inference_time_s = res.get("inference_time_s", 0.0)
            else:
                raw_pred = str(res)
                answer_text = str(res)
                confidence = 0.95
                model_name = self.config.model_name
                vlm_type = "RSCoVLM"
                inference_time_s = 0.0

            # Extract structured validation, highlighting, and regions
            validation = res.get("validation", {
                "image_valid": True,
                "query_valid": True,
                "answerable": True,
                "confidence": confidence,
                "reason": "Evaluated by RSCoVLM.",
            }) if isinstance(res, dict) else {}

            highlighting = res.get("highlighting", {}) if isinstance(res, dict) else {}
            regions = res.get("regions", highlighting.get("regions", [])) if isinstance(res, dict) else []

            # Perform Python image highlighting if regions are present
            highlight_res = None
            if regions:
                try:
                    from app.services.rscovlm_highlight_service import highlight_image_regions
                    highlight_res = highlight_image_regions(image=image, regions=regions)
                    if highlight_res and "regions" in highlight_res:
                        regions = highlight_res["regions"]
                except Exception as h_err:
                    logger.warning(f"Failed to generate highlighted image overlay: {h_err}")

            # Apply strict evidence-grounding guard verification exactly once
            validated_answer = validate_rscovlm_response(
                answer=answer_text,
                specialist_evidence=specialist_evidence,
            )

            # Build standardized visual evidence dictionary
            visual_evidence = {
                "type": "bbox" if regions else "none",
                "available": bool(regions),
                "source": model_name,
                "coordinates": regions[0]["box"] if regions else None,
                "coordinate_system": "normalized",
                "primary_label": regions[0]["label"] if regions else None,
                "primary_score": regions[0]["confidence"] if regions else confidence,
                "all_boxes": [r["box"] for r in regions] if regions else None,
                "all_labels": [r["label"] for r in regions] if regions else None,
                "all_scores": [r["confidence"] for r in regions] if regions else None,
                "regions": regions,
                "highlighted_image_path": highlight_res.get("highlighted_image_path") if highlight_res else None,
                "highlighted_image_filename": highlight_res.get("highlighted_image_filename") if highlight_res else None,
                "highlighted_image_base64": highlight_res.get("highlighted_image_base64") if highlight_res else None,
            }

            return {
                "answer": validated_answer,
                "confidence": confidence,
                "model_name": model_name,
                "vlm_type": vlm_type,
                "validation": validation,
                "highlighting": highlighting,
                "regions": regions,
                "visual_evidence": visual_evidence,
                "highlighted_image_path": highlight_res.get("highlighted_image_path") if highlight_res else None,
                "highlighted_image_filename": highlight_res.get("highlighted_image_filename") if highlight_res else None,
                "highlighted_image_base64": highlight_res.get("highlighted_image_base64") if highlight_res else None,
                "raw_prediction": raw_pred,
                "inference_time_s": inference_time_s,
                "specialist_evidence_used": bool(specialist_evidence),
            }

        except Exception as e:
            logger.error(f"RSCoVLM answer synthesis error: {e}", exc_info=True)
            raise e
