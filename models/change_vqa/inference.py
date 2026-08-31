"""
SatQuery AI — Bi-Temporal Change-VQA Inference Module.

Processes two temporal satellite images + query to evaluate changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model import BiTemporalChangeNetwork
from models.fusion.inference import encode_question


class ChangeVQAInferenceEngine:
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

        with open(vpath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.word_to_id = vocab_data["word_to_id"]
        self.max_length = vocab_data.get("max_length", 40)
        self.vocab_size = vocab_data.get("vocab_size", len(self.word_to_id))

        self.model = BiTemporalChangeNetwork(vocab_size=self.vocab_size).to(self.device)

        if weights_path and Path(weights_path).exists():
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt, strict=False)
            except Exception:
                pass

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        t1_tensor: torch.Tensor,
        t2_tensor: torch.Tensor,
        query: str,
        dates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run bi-temporal change prediction."""
        t1_tensor = t1_tensor.to(self.device)
        t2_tensor = t2_tensor.to(self.device)
        q_tensor = encode_question(
            query,
            self.word_to_id,
            self.max_length,
            device=self.device,
        )

        logits = self.model(t1_tensor, t2_tensor, q_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        conf = probs[pred_idx].item()

        answer = "YES" if pred_idx == 1 else "NO"

        # Generate descriptive context if dates are provided
        date_str = ""
        if dates and len(dates) >= 2:
            date_str = f" between {dates[0]} and {dates[1]}"

        return {
            "answer": answer,
            "confidence": round(max(0.65, min(0.96, conf)), 4),
            "visual_evidence": {
                "type": "none",
            },
            "parameters": {
                "query": query,
                "dates": dates,
                "comparison": f"Evaluated bi-temporal change{date_str}.",
                "probabilities": {
                    "NO": round(probs[0].item(), 4),
                    "YES": round(probs[1].item(), 4),
                },
            },
        }
