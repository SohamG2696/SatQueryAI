"""
SatQuery AI — Optical-SAR Fusion Inference Module.

Loads trained weights and runs multi-task cross-modal inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from .model import MultiTaskFusionModel

# Domain synonym normalization mapping for query tokenization
FUSION_SYNONYMS = {
    "roads": "urban",
    "road": "urban",
    "buildings": "industrial",
    "building": "industrial",
    "tanks": "water",
    "tank": "water",
    "vehicles": "urban",
    "vehicle": "urban",
    "cars": "urban",
    "car": "urban",
    "trees": "forest",
    "tree": "forest",
    "fields": "agriculture",
    "field": "agriculture",
}


def tokenize(text: str) -> list[str]:
    """Tokenize query string into lower-case word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


def normalize_query(query: str) -> str:
    """Normalize query text mapping domain synonyms to trained vocabulary tokens."""
    tokens = tokenize(query)
    norm_tokens = [FUSION_SYNONYMS.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def encode_question(
    text: str,
    word_to_id: dict[str, int],
    max_length: int = 40,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode question into token IDs tensor [1, max_length]."""
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


class FusionInferenceEngine:
    """Inference engine managing the MultiTaskFusionModel."""

    def __init__(
        self,
        weights_path: str | Path,
        vocab_path: str | Path,
        device: torch.device,
    ):
        self.device = device
        self.weights_path = Path(weights_path)
        self.vocab_path = Path(vocab_path)

        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab_data = json.load(f)

        self.word_to_id = self.vocab_data["word_to_id"]
        self.max_length = self.vocab_data.get("max_length", 40)
        self.vocab_size = self.vocab_data.get("vocab_size", len(self.word_to_id))

        self.model = MultiTaskFusionModel(vocab_size=self.vocab_size).to(self.device)

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")

        print(f"[FUSION] Loading checkpoint: {self.weights_path}")
        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)

        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        print(f"[FUSION] Missing keys: {missing}, Unexpected keys: {unexpected}")
        print("[FUSION] Checkpoint loaded successfully with strict=True!")

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        optical_tensor: torch.Tensor,
        sar_tensor: torch.Tensor,
        query: str,
        task_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run multi-task fusion prediction."""
        optical_tensor = optical_tensor.to(self.device)
        sar_tensor = sar_tensor.to(self.device)
        
        norm_q = normalize_query(query)
        q_tensor = encode_question(
            query,
            self.word_to_id,
            self.max_length,
            device=self.device,
        )

        outputs = self.model(
            optical=optical_tensor,
            sar=sar_tensor,
            question=q_tensor,
        )

        query_lower = query.lower()

        # Task 1: Bounding Box
        if task_hint == "bbox" or any(k in query_lower for k in ("where", "locate", "highlight", "find", "bbox", "region", "coordinates")):
            raw_box = outputs["bbox"][0].detach().cpu().tolist()
            x1, y1, x2, y2 = raw_box[0], raw_box[1], raw_box[2], raw_box[3]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            box_coords = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]
            conf = float(torch.softmax(outputs["binary"], dim=1)[0][1].item())

            return {
                "task_sub_type": "bbox",
                "answer": f"Localized target spatial region for query '{query}' in optical-SAR imagery.",
                "confidence": round(conf, 4),
                "normalized_query": norm_q,
                "visual_evidence": {
                    "type": "bbox",
                    "coordinates": box_coords,
                    "coordinate_system": "normalized",
                },
            }

        # Task 2: MCQ
        if task_hint == "mcq" or any(k in query_lower for k in ("option", "which of the following", "(a)", "(b)", "a)", "b)")):
            probs = torch.softmax(outputs["mcq"], dim=1)[0]
            pred_idx = probs.argmax().item()
            conf = probs[pred_idx].item()
            pred_option = chr(ord("A") + pred_idx)
            return {
                "task_sub_type": "mcq",
                "answer": pred_option,
                "confidence": round(conf, 4),
                "normalized_query": norm_q,
                "visual_evidence": None,
                "probabilities": {chr(ord("A") + i): round(p.item(), 4) for i, p in enumerate(probs)},
            }

        # Task 3: Binary YES/NO (Default)
        probs = torch.softmax(outputs["binary"], dim=1)[0]
        pred_idx = probs.argmax().item()
        conf = probs[pred_idx].item()
        answer = "YES" if pred_idx == 1 else "NO"

        return {
            "task_sub_type": "binary",
            "answer": answer,
            "confidence": round(conf, 4),
            "normalized_query": norm_q,
            "visual_evidence": None,
            "probabilities": {
                "NO": round(probs[0].item(), 4),
                "YES": round(probs[1].item(), 4),
            },
        }
