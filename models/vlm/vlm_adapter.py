"""
SatQuery AI — Person A LLaVA-OneVision VLM Adapter.

Provides a reusable inference adapter for Person A's Remote-Sensing VLM model.
Loads merged weights (Qwen2-0.5B-Instruct + SigLIP) once and performs VQA inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Union

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


class VLMAdapter:
    """Inference adapter managing Person A's LLaVA-OneVision VLM model."""

    _instance: VLMAdapter | None = None

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        device: torch.device | str | None = None,
    ):
        if weights_dir is None:
            weights_dir = Path(__file__).parent / "weights"
        self.weights_dir = Path(weights_dir)

        if not self.weights_dir.exists():
            raise FileNotFoundError(f"VLM weights directory not found at: {self.weights_dir}")

        # Select execution device
        if device is not None:
            self.device = str(device)
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
        else:
            self.device = "cpu"

        print(f"[VLMAdapter] Loading processor and model on device: {self.device}")
        load_start = time.time()
        try:
            self.processor = AutoProcessor.from_pretrained(self.weights_dir)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.weights_dir,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device != "cpu" else None,
            )
            if self.device == "cpu":
                self.model.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load VLM model from {self.weights_dir}: {e}") from e

        load_dur = time.time() - load_start
        print(f"[VLMAdapter] Model loaded successfully in {load_dur:.2f}s.")

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        question: str,
    ) -> dict[str, Any]:
        """
        Run VQA inference given an image and a natural-language question.

        Returns:
            dict with keys: question, raw_prediction, prediction, inference_time_s, model
        """
        # Validate question
        if not question or not str(question).strip():
            raise ValueError("Question cannot be empty or blank.")
        question_str = str(question).strip()

        # Validate & load image
        if image is None:
            raise ValueError("Image is required but None was provided.")

        pil_image: Image.Image
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            try:
                pil_image = Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, Exception) as e:
                raise ValueError(f"Invalid image file at {img_path}: {e}") from e
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL Image or file path.")

        # Build prompt using chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question_str},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Preprocess inputs
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        # Run inference
        t0 = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=50)
        infer_duration = time.time() - t0

        # Decode response
        generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_prediction = self.processor.decode(generated_tokens, skip_special_tokens=True)
        prediction = raw_prediction.strip()

        return {
            "question": question_str,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "inference_time_s": round(infer_duration, 4),
            "model": "llava-onevision-person-a",
        }


_vlm_adapter_instance: VLMAdapter | None = None


def get_vlm_adapter(
    weights_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> VLMAdapter:
    """Get or create singleton VLMAdapter instance."""
    global _vlm_adapter_instance
    if _vlm_adapter_instance is None:
        _vlm_adapter_instance = VLMAdapter(weights_dir=weights_dir, device=device)
    return _vlm_adapter_instance
