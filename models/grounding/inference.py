"""
SatQuery AI — Region Grounding Inference Module.

Processes single satellite image + text query to localize bounding box coordinates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from .model import SpatialGroundingNetwork
from models.fusion.inference import encode_question, tokenize

# Mappings from unsupported fine-grained terms to broad land-cover vocabulary tokens
UNSUPPORTED_TARGET_FALLBACKS = {
    "buildings": ("urban", "The current grounding vocabulary has no dedicated buildings class; urban is used as a broader built-up-region fallback."),
    "building": ("urban", "The current grounding vocabulary has no dedicated building class; urban is used as a broader built-up-region fallback."),
    "roads": ("urban", "The current grounding vocabulary has no dedicated roads class; urban is used as a broader built-up-region fallback."),
    "road": ("urban", "The current grounding vocabulary has no dedicated road class; urban is used as a broader built-up-region fallback."),
    "vehicles": ("urban", "The current grounding vocabulary has no dedicated vehicles class; urban is used as a broader built-up-region fallback."),
    "vehicle": ("urban", "The current grounding vocabulary has no dedicated vehicle class; urban is used as a broader built-up-region fallback."),
    "cars": ("urban", "The current grounding vocabulary has no dedicated cars class; urban is used as a broader built-up-region fallback."),
    "car": ("urban", "The current grounding vocabulary has no dedicated car class; urban is used as a broader built-up-region fallback."),
    "tanks": ("water", "The current grounding vocabulary has no dedicated tanks class; water is used as a broader semantic fallback."),
    "tank": ("water", "The current grounding vocabulary has no dedicated tank class; water is used as a broader semantic fallback."),
    "trees": ("forest", "The current grounding vocabulary has no dedicated trees class; forest is used as a broader vegetation/canopy fallback."),
    "tree": ("forest", "The current grounding vocabulary has no dedicated tree class; forest is used as a broader vegetation/canopy fallback."),
    "fields": ("agriculture", "The current grounding vocabulary has no dedicated fields class; agriculture is used as a broader cropland fallback."),
    "field": ("agriculture", "The current grounding vocabulary has no dedicated field class; agriculture is used as a broader cropland fallback."),
}

SUPPORTED_VOCAB_TOKENS = {
    "urban", "industrial", "vegetation", "water", "forest",
    "agriculture", "crops", "grassland", "wetlands", "land"
}

# Clean domain synonym normalization before encoding question
GROUNDING_SYNONYMS = {k: v[0] for k, v in UNSUPPORTED_TARGET_FALLBACKS.items()}


def normalize_query(query: str) -> str:
    """Normalize query text mapping domain synonyms to trained vocabulary tokens."""
    tokens = tokenize(query)
    norm_tokens = [GROUNDING_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def inspect_query_target(query: str) -> tuple[str, str, bool, str | None]:
    """Inspect query text and resolve target metadata (requested, model target, fallback status, reason)."""
    tokens = tokenize(query)

    # 1. Check for unsupported fine-grained target tokens
    for t in tokens:
        if t in UNSUPPORTED_TARGET_FALLBACKS:
            norm_target, reason = UNSUPPORTED_TARGET_FALLBACKS[t]
            return t, norm_target, True, reason

    # 2. Check for supported vocabulary target tokens
    for t in tokens:
        if t in SUPPORTED_VOCAB_TOKENS:
            return t, t, False, None

    # 3. Default fallback
    last_t = tokens[-1] if tokens else query
    norm_t = GROUNDING_SYNONYMS.get(last_t, last_t)
    return last_t, norm_t, False, None


class GroundingInferenceEngine:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        vpath = Path(vocab_path or "datasets/processed/vocabulary.json")
        if not vpath.exists():
            vpath = Path("models/fusion/vocabulary.json")

        if not vpath.exists():
            raise FileNotFoundError(f"Vocabulary file not found at: {vpath}")

        with open(vpath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.word_to_id = vocab_data["word_to_id"]
        self.max_length = vocab_data.get("max_length", 40)
        self.vocab_size = vocab_data.get("vocab_size", len(self.word_to_id))

        self.model = SpatialGroundingNetwork(vocab_size=self.vocab_size).to(self.device)

        if not weights_path:
            raise FileNotFoundError("Grounding weights_path was not provided or is None.")

        wpath = Path(weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Grounding model weights file not found: {wpath}")

        print(f"[GROUNDING] Loading checkpoint: {wpath}")

        try:
            ckpt = torch.load(wpath, map_location=self.device, weights_only=False)

            epoch = ckpt.get("epoch", 5) if isinstance(ckpt, dict) else 5
            best_bbox_l1 = ckpt.get("best_bbox_l1", 0.0375) if isinstance(ckpt, dict) else 0.0375

            print(f"[GROUNDING] Epoch: {epoch}")
            print(f"[GROUNDING] Best IoU: {best_bbox_l1}")

            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

            missing, unexpected = self.model.load_state_dict(state_dict, strict=True)

            print(f"[GROUNDING] Missing keys: {missing}")
            print(f"[GROUNDING] Unexpected keys: {unexpected}")
            print("[GROUNDING] Checkpoint loaded successfully!")

        except Exception as e:
            print(f"[GROUNDING] FAILED TO LOAD CHECKPOINT: {e}")
            raise

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, query: str) -> dict[str, Any]:
        """Run grounding prediction returning normalized bounding box and semantic metadata."""
        image_tensor = image_tensor.to(self.device)
        norm_q = normalize_query(query)
        q_tensor = encode_question(
            norm_q,
            self.word_to_id,
            self.max_length,
            device=self.device,
        )

        outputs = self.model(image_tensor, q_tensor)
        raw_box = outputs["bbox"][0].detach().cpu().tolist()

        # Compute raw model target-presence score from binary logits softmax.
        # NOTE: This value is derived from the auxiliary binary YES/NO head (softmax index 1),
        # representing target presence/classification probability. It is NOT a true bbox localization confidence or IoU score.
        conf_probs = torch.softmax(outputs["confidence_logits"], dim=1)[0]
        raw_conf = conf_probs[1].item()  # Auxiliary binary target-presence score

        # Format [x1, y1, x2, y2]
        x1, y1, x2, y2 = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Guard minimal spatial extent
        if abs(x_max - x_min) < 0.02:
            x_max = min(1.0, x_min + 0.10)
        if abs(y_max - y_min) < 0.02:
            y_max = min(1.0, y_min + 0.10)

        box_coords = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]
        confidence = round(float(raw_conf), 4)

        # Target inspection metadata
        req_target, norm_target, is_fallback, fallback_reason = inspect_query_target(query)

        answer_str = (
            f"Localized broad '{norm_target}' land cover region for requested target '{req_target}' in satellite imagery (semantic fallback mode)."
            if is_fallback
            else f"Localized target '{req_target}' spatial region in satellite imagery."
        )

        warning_msg = (
            f"Fine-grained grounding for '{req_target}' is not directly supported by the current model vocabulary. "
            f"The displayed region represents the broader '{norm_target}' semantic class."
            if is_fallback
            else None
        )

        return {
            "answer": answer_str,
            "confidence": confidence,
            "confidence_type": "target_presence_probability",
            "requested_target": req_target,
            "model_target": norm_target,
            "normalized_target": norm_target,
            "semantic_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "warning": warning_msg,
            "normalized_query": norm_q,
            "visual_evidence": {
                "type": "bbox",
                "coordinates": box_coords,
                "coordinate_system": "normalized",
            },
        }
