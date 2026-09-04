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

# Safe linguistic normalization dictionary preserving target object semantics
FUSION_SYNONYMS: dict[str, str] = {
    "road": "roads",
    "building": "buildings",
    "tree": "trees",
    "vehicle": "vehicles",
    "car": "cars",
    "field": "fields",
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


def is_mcq_query(query: str, task_hint: str | None = None) -> bool:
    """Determine if query is a multiple-choice question (MCQ)."""
    if task_hint == "mcq":
        return True
    if task_hint is not None and task_hint != "auto":
        return False

    q_lower = query.lower()

    mcq_keywords = (
        "option",
        "which of the following",
        "which category",
        "which class",
        "which type",
        "which of these",
        "best describes",
        "choose",
        "select",
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "a)",
        "b)",
        "c)",
        "d)",
        "a.",
        "b.",
    )

    if any(k in q_lower for k in mcq_keywords):
        return True

    # Check for lists with ' or ' (e.g. "x, y, or z" or "x, y, z, or w")
    if " or " in q_lower and ("," in q_lower or ":" in q_lower or "?" in q_lower):
        parts = [p.strip() for p in re.split(r"[,;]|\bor\b", q_lower) if p.strip()]
        if len(parts) >= 3:
            return True

    return False


def parse_mcq_options(query: str) -> dict[str, str]:
    """Parse MCQ options (A/B/C/D) from a natural language query string."""
    options = {}
    q_text = query.strip()

    matches = list(re.finditer(r'(?:^|[\s,;:(])(?:\(?([A-Da-d])[\)\.]|\b([A-Da-d])\))\s*', q_text))
    if len(matches) >= 2:
        for i in range(len(matches)):
            m = matches[i]
            letter = (m.group(1) or m.group(2)).upper()
            start_idx = m.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(q_text)
            val = q_text[start_idx:end_idx].strip().rstrip(",;.? \t\n")
            if val and letter not in options:
                options[letter] = val
        if len(options) >= 2:
            return options

    if ":" in q_text:
        after_colon = q_text.split(":", 1)[1].rstrip("?")
        parts = [p.strip() for p in re.split(r"[,;]|\bor\b", after_colon) if p.strip()]
        if len(parts) >= 2:
            letters = ["A", "B", "C", "D"]
            for i, p in enumerate(parts[:4]):
                clean_p = re.sub(r'^\(?[A-Da-d][\)\.]\s*', '', p).strip()
                if clean_p:
                    options[letters[i]] = clean_p
            return options

    if " or " in q_text.lower():
        raw = re.sub(r'^(?:which|select|choose|describe|what is).*?:\s*', '', q_text, flags=re.IGNORECASE).rstrip("?")
        parts = [p.strip() for p in re.split(r"[,;]|\bor\b", raw) if p.strip()]
        if len(parts) >= 2:
            letters = ["A", "B", "C", "D"]
            for i, p in enumerate(parts[:4]):
                clean_p = re.sub(r'^\(?[A-Da-d][\)\.]\s*', '', p).strip()
                if clean_p:
                    options[letters[i]] = clean_p
            return options

    return options


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
        if task_hint == "bbox" or (task_hint is None and any(k in query_lower for k in ("where", "locate", "highlight", "find", "bbox", "region", "coordinates"))):
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
        if is_mcq_query(query, task_hint=task_hint):
            probs = torch.softmax(outputs["mcq"], dim=1)[0]
            pred_idx = probs.argmax().item()
            conf = probs[pred_idx].item()
            pred_option = chr(ord("A") + pred_idx)

            parsed_options = parse_mcq_options(query)
            selected_text = parsed_options.get(pred_option)
            if selected_text:
                answer = f"{pred_option} ({selected_text})"
            else:
                answer = pred_option

            prob_dict = {}
            for i, p in enumerate(probs):
                opt_letter = chr(ord("A") + i)
                opt_label = f"{opt_letter} ({parsed_options[opt_letter]})" if opt_letter in parsed_options else opt_letter
                prob_dict[opt_label] = round(p.item(), 4)

            return {
                "task_sub_type": "mcq",
                "answer": answer,
                "confidence": round(conf, 4),
                "normalized_query": norm_q,
                "visual_evidence": None,
                "options": parsed_options if parsed_options else None,
                "probabilities": prob_dict,
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
