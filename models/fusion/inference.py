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


def tokenize(text: str) -> list[str]:
    """Tokenize query string into lower-case word tokens."""
    return re.findall(r"\b\w+\b", str(text).lower())


def encode_question(
    text: str,
    word_to_id: dict[str, int],
    max_length: int = 40,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode question into token IDs tensor [1, max_length]."""
    tokens = tokenize(text)
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
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Invalid checkpoint format in {self.weights_path}")

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
            box = outputs["bbox"][0].detach().cpu().tolist()
            return {
                "task_sub_type": "bbox",
                "answer": "Detected target spatial region corresponding to the multimodal query.",
                "confidence": 0.85,
                "visual_evidence": {
                    "type": "bbox",
                    "coordinates": box,
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
            "visual_evidence": None,
            "probabilities": {
                "NO": round(probs[0].item(), 4),
                "YES": round(probs[1].item(), 4),
            },
        }
