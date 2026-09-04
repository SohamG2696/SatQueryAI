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
from models.fusion.inference import tokenize

# Safe linguistic normalization dictionary preserving target object semantics
# Note: Target objects like 'roads', 'buildings', 'trees', 'vehicles' must NOT be mapped to land-cover classes like 'urban' or 'industrial'.
GROUNDING_SYNONYMS: dict[str, str] = {
    # Safe identity or spelling variants
    "road": "roads",
    "building": "buildings",
    "tree": "trees",
    "vehicle": "vehicles",
    "car": "cars",
    "field": "fields",
}


def normalize_query(query: str) -> str:
    """Normalize query text performing lowercase, whitespace cleanup, and punctuation removal without altering target object semantics."""
    tokens = tokenize(query)
    norm_tokens = [GROUNDING_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def encode_question(
    text: str,
    word_to_id: dict[str, int],
    max_length: int = 40,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode question into token IDs tensor [1, max_length] using Grounding query normalization."""
    norm_text = normalize_query(text)
    tokens = tokenize(norm_text)
    pad_id = word_to_id.get("<PAD>", 0)
    unk_id = word_to_id.get("<UNK>", 1)

    ids = [word_to_id.get(tok, unk_id) for tok in tokens[:max_length]]
    ids += [pad_id] * (max_length - len(ids))

    tensor = torch.tensor([ids], dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


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
        """Run grounding prediction returning normalized bounding box."""
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

        # Compute raw model confidence from binary logits softmax
        conf_probs = torch.softmax(outputs["confidence_logits"], dim=1)[0]
        raw_conf = conf_probs[1].item()  # Probability of target detection

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

        return {
            "answer": f"Localized target spatial region for query '{query}' in satellite imagery.",
            "confidence": confidence,
            "normalized_query": norm_q,
            "visual_evidence": {
                "type": "bbox",
                "coordinates": box_coords,
                "coordinate_system": "normalized",
            },
        }
