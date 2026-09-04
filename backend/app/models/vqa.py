"""
SatQuery AI — Visual Question Answering (VQA) Model Adapter.

Connects single-image VQA queries to Person A's LLaVA-OneVision VLM adapter.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

from models.vlm.vlm_adapter import get_vlm_adapter


def _to_pil_image(source: Any) -> Image.Image:
    """Convert various image formats (Path, str, bytes, PIL, numpy) into RGB PIL Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {p}")
        return Image.open(p).convert("RGB")
    if isinstance(source, (bytes, io.BytesIO)):
        buf = io.BytesIO(source) if isinstance(source, bytes) else source
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8 and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr[:, :, :3]).convert("RGB")
    raise ValueError(f"Unsupported image input type for VLM: {type(source)}")


def run_module(
    images: list[Any],
    query: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute single-image Visual Question Answering using Person A's VLM model."""
    if not images:
        raise ValueError("VQA requires at least one satellite image.")

    pil_img = _to_pil_image(images[0])
    adapter = get_vlm_adapter()
    res = adapter.predict(image=pil_img, question=query)

    return {
        "answer": res["prediction"],
        "confidence": 0.90,
        "visual_evidence": {
            "type": "none",
        },
        "model_name": "satquery-vlm-person-a",
        "parameters": {
            "query": res["question"],
            "raw_prediction": res["raw_prediction"],
            "inference_time_s": res["inference_time_s"],
            "model": res["model"],
            "status": "ready",
        },
    }
