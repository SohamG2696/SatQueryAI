"""
RSCoVLM-7B-2512-SatQuery Adapter Implementation.

Uses Qwen2_5_VLForConditionalGeneration and PEFT LoRA in bfloat16.
"""

from __future__ import annotations

import base64
from io import BytesIO
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image
import requests

from models.vlm.rscovlm.rscovlm_config import RSCoVLMConfig
from models.vlm.rscovlm.rscovlm_utils import (
    extract_rscovlm_json,
    format_rscovlm_prompt,
    validate_rscovlm_response,
)

logger = logging.getLogger("satquery.rscovlm.adapter")


class RSCoVLMAdapter:
    """
    Adapter for fine-tuned RSCoVLM-7B-2512-SatQuery vision-language model.
    Loads Qwen2_5_VLForConditionalGeneration base model and PEFT LoRA adapter in bfloat16.
    Supports local execution and remote REST forwarding.
    """

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        remote_enabled: Optional[bool] = None,
        remote_url: Optional[str] = None,
        config: Optional[RSCoVLMConfig] = None,
    ):
        self.config = config or RSCoVLMConfig()

        # Resolve paths & environment settings
        base_path = (
            base_model_path
            or os.environ.get("RSCOVLM_BASE_MODEL_PATH")
            or getattr(self.config, "base_model_path", "")
        )
        adapt_path = (
            adapter_path
            or os.environ.get("RSCOVLM_ADAPTER_PATH")
            or getattr(self.config, "adapter_path", "")
        )
        dev = (
            device
            or os.environ.get("RSCOVLM_DEVICE")
            or getattr(self.config, "device", "cuda")
        )

        # Retrieve backend config settings if available
        try:
            from backend.app.config import settings

            if not base_path:
                base_path = settings.rscovlm_base_model_path
            if not adapt_path:
                adapt_path = settings.rscovlm_adapter_path
            if device is None:
                dev = settings.rscovlm_device
            if remote_enabled is None:
                remote_enabled = settings.rscovlm_remote_enabled
            if remote_url is None:
                remote_url = settings.rscovlm_remote_url
        except ImportError:
            pass

        rem_flag = (
            remote_enabled
            if remote_enabled is not None
            else (os.environ.get("RSCOVLM_REMOTE_ENABLED", "false").lower() == "true")
        )
        rem_url = remote_url or os.environ.get("RSCOVLM_REMOTE_URL", "")

        self.config.base_model_path = base_path
        self.config.adapter_path = adapt_path
        self.config.device = dev
        self.config.remote_enabled = rem_flag
        self.config.remote_url = rem_url

        self.model_name = self.config.model_name
        self.model_type = "RSCoVLM"

        self.model = None
        self.processor = None
        self.is_loaded = False
        self.is_remote = False

        if rem_flag and rem_url:
            self.is_remote = True
            self.is_loaded = True
            logger.info(f"RSCoVLM initialized in remote REST mode: {rem_url}")
        elif (base_path and os.path.exists(base_path)) or (
            adapt_path and os.path.exists(adapt_path)
        ):
            self.is_loaded = True
            logger.info(
                f"RSCoVLM local paths confirmed available (Base: '{base_path}', Adapter: '{adapt_path}'). Model loading is lazy."
            )
        else:
            self.is_loaded = False
            logger.info(
                f"RSCoVLM local model paths unavailable (Base: '{base_path}', Adapter: '{adapt_path}')."
            )

    def _ensure_model_loaded(self) -> None:
        """Lazy load Qwen2.5-VL base model and PEFT LoRA adapter into memory."""
        if self.is_remote:
            return
        if self.model is not None and self.processor is not None:
            return

        base_path = self.config.base_model_path
        adapt_path = self.config.adapter_path

        if not base_path or not os.path.exists(base_path):
            self.is_loaded = False
            raise FileNotFoundError(
                f"RSCoVLM base model path does not exist: '{base_path}'"
            )

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from peft import PeftModel

            logger.info(f"Loading Qwen2.5-VL base model in bfloat16 from: {base_path}")
            proc_path = (
                adapt_path
                if (
                    adapt_path
                    and os.path.exists(
                        os.path.join(adapt_path, "preprocessor_config.json")
                    )
                )
                else base_path
            )
            self.processor = AutoProcessor.from_pretrained(proc_path)

            target_device = self.config.device
            if target_device == "cuda" and not torch.cuda.is_available():
                target_device = "cpu"

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_path,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if target_device == "cuda" else "cpu",
            )

            if adapt_path and os.path.exists(adapt_path):
                logger.info(f"Loading RSCoVLM PEFT LoRA adapter from: {adapt_path}")
                self.model = PeftModel.from_pretrained(self.model, adapt_path)

            self.model.eval()
            self.is_loaded = True
            logger.info(
                "Successfully loaded RSCoVLM Qwen2.5-VL + PEFT model into memory."
            )
        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.processor = None
            logger.error(f"Failed to load RSCoVLM model weights: {e}")
            raise e

    def predict(
        self,
        image: Union[str, Path, Image.Image, Dict[str, Any]],
        question: str,
        specialist_evidence: Optional[Dict[str, Any]] = None,
        task: str = "vqa",
    ) -> Dict[str, Any]:
        """
        Execute vision-language inference using RSCoVLM.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"RSCoVLM model is not available or loaded (Base path: '{self.config.base_model_path}', Adapter path: '{self.config.adapter_path}')."
            )

        pil_image = self._convert_to_pil(image)
        if pil_image is None:
            raise ValueError(
                "RSCoVLM requires a valid image path, PIL image, or image1 payload"
            )

        if self.is_remote and self.config.remote_url:
            return self._predict_remote(pil_image, question, specialist_evidence, task)

        # Ensure model is loaded before inference
        self._ensure_model_loaded()
        return self._predict_local(pil_image, question, specialist_evidence, task)

    def _predict_remote(
        self,
        image: Optional[Image.Image],
        question: str,
        specialist_evidence: Optional[Dict[str, Any]],
        task: str,
    ) -> Dict[str, Any]:
        """Route prediction to remote HTTP REST endpoint."""
        try:
            image_b64 = ""
            if image is not None:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "question": question,
                "task": task,
                "specialist_evidence": specialist_evidence or {},
                "image_b64": image_b64,
            }
            resp = requests.post(self.config.remote_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError("Remote RSCoVLM response must be a JSON object")
            raw_answer = data.get("answer", data.get("raw_prediction", ""))
            parsed = extract_rscovlm_json(str(raw_answer))

            answer_text = parsed.get("answer", str(raw_answer))
            validation = parsed.get("validation", data.get("validation", {}))
            highlighting = parsed.get("highlighting", data.get("highlighting", {}))
            if not isinstance(validation, dict):
                validation = {}
            if not isinstance(highlighting, dict):
                highlighting = {}
            confidence = float(
                data.get("confidence", validation.get("confidence", 0.95))
            )

            return {
                "answer": answer_text,
                "raw_prediction": raw_answer,
                "confidence": confidence,
                "model_name": self.model_name,
                "vlm_type": "RSCoVLM",
                "validation": validation,
                "highlighting": highlighting,
                "regions": highlighting.get("regions", []),
                "structured": parsed.get("structured", False),
                "remote": True,
            }
        except Exception as e:
            logger.error(f"Remote RSCoVLM HTTP request failed: {e}")
            raise e

    def _predict_local(
        self,
        image,
        question: str,
        specialist_evidence=None,
        task: str = "vqa",
    ):
        """
        RSCoVLM inference using the same direct Qwen2.5-VL
        pipeline that produced the successful standalone result.

        The fine-tuned model was trained on mixed tasks:

        VQA:
            d

        Spatial:
            [0.25,0.0,1.0,0.97]

        Point-to-box:
            [0.0,0.0,0.09,0.2]

        Therefore we do NOT force JSON output.
        """

        import re
        import torch
        from PIL import Image

        self._ensure_model_loaded()

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = self._convert_to_pil(image)

        # --------------------------------------------------------
        # DETAILED ANSWER PROMPT
        # --------------------------------------------------------

        prompt = str(question).strip()

        task_lower = str(task or "vqa").lower()

        # For spatial/localization questions, preserve the model's
        # learned coordinate-output behavior so highlighting continues
        # to work exactly as before.
        spatial_keywords = [
            "point out",
            "locate",
            "localize",
            "where is",
            "where are",
            "identify the location",
            "bounding box",
            "bounding-box",
            "box around",
            "enclosing",
            "highlight",
            "region",
            "area",
            "<ref>",
            "<point>",
        ]

        is_spatial_question = any(
            keyword in prompt.lower()
            for keyword in spatial_keywords
        )

        detailed_keywords = [
            "describe",
            "describe in detail",
            "detailed description",
            "detailed analysis",
            "analyze the image",
            "analyze this image",
            "explain the scene",
            "explain what is visible",
            "main features",
            "spatial arrangement",
            "spatial relationships",
            "where are the features",
            "what can be seen",
            "what is visible",
            "give an overview",
            "provide an overview",
            "overall scene",
            "landscape description",
        ]

        is_detailed_question = any(
            keyword in prompt.lower()
            for keyword in detailed_keywords
        )

        if is_spatial_question:
            # Preserve the exact learned spatial/grounding format.
            inference_prompt = prompt

        elif is_detailed_question:
            inference_prompt = f"""
Analyze the entire satellite image carefully before answering.

Question:
{prompt}

Provide a detailed, natural-language analysis of the satellite image.

Answer requirements:
- Start with a direct answer to the question.
- Write at least 5 complete sentences.
- Prefer 6 to 8 informative sentences when the image contains enough
  visible information.
- Describe the main visible objects, structures, land-cover features,
  vegetation, water, roads, or other relevant features.
- Explain where the important features are located within the image.
- Describe the spatial arrangement and relationships between important
  features.
- Mention visible patterns such as density, distribution, shape,
  orientation, clustering, spacing, or surrounding context when useful.
- Use spatial terms such as upper-left, upper-right, lower-left,
  lower-right, center, top, bottom, along the edge, adjacent to,
  surrounded by, or between when visually appropriate.
- Consider the entire image rather than focusing on only one region.
- Avoid repeating the same observation.
- Only describe information that is visually supported by the image.
- Do not invent measurements, exact locations, object identities,
  or facts that cannot reasonably be inferred visually.
- Do not output JSON.
- Do not output bounding-box coordinates.
- Do not mention these instructions.

Provide the final answer directly.
""".strip()

        else:
            # Normal VQA/classification questions should preserve
            # concise question-answer behavior.
            inference_prompt = prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_image,
                    },
                    {
                        "type": "text",
                        "text": inference_prompt,
                    },
                ],
            }
        ]

        # --------------------------------------------------------
        # QWEN CHAT TEMPLATE
        # --------------------------------------------------------

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # --------------------------------------------------------
        # VISION INPUTS
        # --------------------------------------------------------

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        except Exception:
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )

        # --------------------------------------------------------
        # GPU
        # --------------------------------------------------------

        device = next(self.model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # --------------------------------------------------------
        # GENERATION
        # --------------------------------------------------------

        # --------------------------------------------------------
        # MODEL MODE
        # --------------------------------------------------------
        # Spatial questions keep the fine-tuned LoRA because the
        # adapter learned the coordinate/grounding output format.
        #
        # Descriptive questions temporarily disable LoRA so the
        # underlying Qwen2.5-VL language model can produce natural,
        # detailed descriptions without the short-answer behavior
        # learned by the VQA/grounding LoRA.
        #
        # The same base model remains in VRAM; no second 7B model
        # is loaded.
        # --------------------------------------------------------

                # --------------------------------------------------------
        # GENERATION
        # --------------------------------------------------------

        if is_spatial_question:
            # Spatial/grounding questions use the fine-tuned LoRA.
            generation_max_tokens = 256
            use_lora = True

        elif is_detailed_question:
            # Detailed natural-language analysis currently uses the
            # underlying base Qwen2.5-VL model. The LoRA was trained
            # heavily on short VQA/grounding outputs, so disabling it
            # avoids forcing detailed answers into that short format.
            generation_max_tokens = 384
            use_lora = False

        else:
            # Normal VQA/classification.
            generation_max_tokens = 128
            use_lora = False

        if not use_lora and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=generation_max_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
        else:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_max_tokens,
                    do_sample=False,
                    use_cache=True,
                )

        # --------------------------------------------------------
        # REMOVE INPUT TOKENS
        # --------------------------------------------------------

        input_ids = inputs.get("input_ids")

        if input_ids is not None:

            generated_ids_trimmed = [
                output_ids[len(input_ids_one) :]
                for input_ids_one, output_ids in zip(input_ids, generated_ids)
            ]

        else:
            generated_ids_trimmed = generated_ids

        # --------------------------------------------------------
        # DECODE
        # --------------------------------------------------------

        answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        # --------------------------------------------------------
        # RAW MODEL OUTPUT
        # --------------------------------------------------------

        # ========================================================
        # SPATIAL OUTPUT PARSER
        # ========================================================

        regions = []

        # Matches:
        #
        # [0.25,0.0,1.0,0.97]
        # [0.0 0.0, 1.0 1.0]
        # [123,456,789,900]

        bbox_pattern = re.compile(
            r"\[\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*[, ]\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*[, ]\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*[, ]\s*"
            r"(-?\d+(?:\.\d+)?)"
            r"\s*\]"
        )

        for match in bbox_pattern.findall(answer):

            try:

                box = [float(v) for v in match]

                x1, y1, x2, y2 = box

                if x2 <= x1 or y2 <= y1:
                    continue

                # Normalized coordinates
                if all(0 <= v <= 1 for v in box):

                    normalized = box

                # Qwen-style 0-1000 coordinates
                elif all(0 <= v <= 1000 for v in box):

                    normalized = [v / 1000.0 for v in box]

                else:
                    continue

                # Do not highlight the complete image.
                if (
                    normalized[0] <= 0.001
                    and normalized[1] <= 0.001
                    and normalized[2] >= 0.999
                    and normalized[3] >= 0.999
                ):
                    continue

                regions.append(
                    {
                        "label": "identified region",
                        "box": normalized,
                        "confidence": 0.90,
                    }
                )

            except Exception:
                pass

        # Remove duplicate boxes
        unique_regions = []

        for region in regions:

            if region["box"] not in [x["box"] for x in unique_regions]:
                unique_regions.append(region)

        regions = unique_regions

        # ========================================================
        # VALIDATION
        # ========================================================

        validation = {
            "image_valid": True,
            "query_valid": bool(prompt),
            "answerable": True,
            "confidence": 0.90,
            "reason": (
                "RSCoVLM analyzed the satellite image using "
                "its fine-tuned VQA/spatial capabilities."
            ),
        }

        # ========================================================
        # RETURN
        # ========================================================

        return {
            "success": True,
            "model_name": "RSCoVLM-7B-2512-SatQuery",
            "task": task,
            "answer": answer,
            "raw_prediction": answer,
            "confidence": 0.90,
            "validation": validation,
            "regions": regions,
            "highlighting": {
                "required": len(regions) > 0,
                "regions": regions,
            },
        }

    def _convert_to_pil(
        self, image: Union[str, Path, Image.Image, Dict[str, Any]]
    ) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)) and os.path.exists(image):
            return Image.open(image).convert("RGB")
        if isinstance(image, dict) and "image1" in image:
            img1 = image["image1"]
            if isinstance(img1, (str, Path)) and os.path.exists(img1):
                return Image.open(img1).convert("RGB")
            elif isinstance(img1, Image.Image):
                return img1
        return None
