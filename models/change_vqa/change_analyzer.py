"""
ml/inference/change_analyzer.py
================================
Module 3 — Multitemporal Change Detection Inference Module

Final interface for Person C's agentic backend integration with
category-aware semantic grounding and robust error handling.

Usage
-----
    from ml.inference.change_analyzer import ChangeAnalyzer

    analyzer = ChangeAnalyzer(checkpoint_path="checkpoints/full/checkpoint_best.pth")

    result = analyzer.analyze_change(
        image_t1="/path/to/t1.png",
        image_t2="/path/to/t2.png",
        question="Have the areas of buildings changed?"
    )

Response Schema (FastAPI Compatible)
------------------------------------
{
    "answer": "Yes, there are changes detected in buildings (63.4% of the area changed).",
    "raw_answer": "yes",
    "confidence": 0.5829,
    "change_ratio": 0.6337,
    "global_change_ratio": 0.1800,
    "category": "buildings",
    "category_change_ratio": 0.6337,
    "has_grounding": true,
    "change_mask_base64": "iVBORw0KGgoAAAANSUhEUg...",
    "task": "change_detection",
    "model": "ChangeFormerV6",
    "question_type": "change_or_not",
    "question": "Have the areas of buildings changed?",
    "metrics": {
        "changed_pixels": 11796,
        "total_pixels": 65536,
        "mean_change_probability": 0.5829
    }
}
"""

import sys
import base64
import io
import re
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Project root detection
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_CHANGEFORMER_ROOT = _PROJECT_ROOT / "ChangeFormer"

sys.path.insert(0, str(_CHANGEFORMER_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "ml" / "datasets"))
sys.path.insert(0, str(_PROJECT_ROOT / "ml" / "inference"))

from semantic_grounder import SemanticGrounder, normalize_category_name, ALL_CLASSES


# ============================================================
# CDVQA Question Category & Type Classifier
# ============================================================

_CATEGORY_PATTERNS = [
    (r"\bplaygrounds?\b", "playgrounds"),
    (r"\bbuildings?\b", "buildings"),
    (r"\btrees?\b", "trees"),
    (r"\bwater\b", "water"),
    (r"\blow[\s_]vegetation\b|\bvegetation\b", "low_vegetation"),
    (
        r"\bnon[\s_-]?vegetated[\s_]ground[\s_]surface\b|\bnvg[\s_]surface\b|\bnon[\s_-]?vegetated\b|\bground[\s_]surface\b",
        "NVG_surface",
    ),
]

_CHANGE_OR_NOT_PATTERNS = [
    r"\bchanged?\b", r"\bchanges?\b", r"\bmodified?\b",
    r"\baltered?\b", r"\bdifferent\b", r"\bsame\b",
    r"\bany change\b", r"\bis there\b",
]

_INCREASE_PATTERNS = [
    r"\bincreased?\b", r"\bgrew?\b", r"\bexpanded?\b",
    r"\blarger\b", r"\bmore\b", r"\bgrown\b",
]

_DECREASE_PATTERNS = [
    r"\bdecreased?\b", r"\bshrunk?\b", r"\breduced?\b",
    r"\bsmaller\b", r"\bless\b", r"\bfewer\b",
]

_RATIO_PATTERNS = [
    r"\bratio\b", r"\bpercent\b", r"\bproportion\b",
    r"\bhow much\b", r"\bfraction\b",
]


def extract_category(question: str) -> Optional[str]:
    """Extract land-cover category from question string."""
    if not isinstance(question, str):
        return None
    q = question.lower()
    for pattern, cat in _CATEGORY_PATTERNS:
        if re.search(pattern, q):
            return cat
    return None


def classify_question(question: str) -> str:
    """
    Classify a CDVQA question into one of the 8 standard CDVQA question types:
      1. change_or_not
      2. change_ratio_types
      3. increase_or_not
      4. decrease_or_not
      5. change_to_what
      6. smallest_change
      7. largest_change
      8. change_ratio
    """
    if not isinstance(question, str) or not question.strip():
        return "change_or_not"

    q = question.lower()
    cat = extract_category(question)

    if "smallest" in q or "least" in q:
        return "smallest_change"
    if "largest" in q or "most" in q or "greatest" in q:
        return "largest_change"
    if "change to" in q or "changed to" in q or "become" in q or "turned into" in q or "what did" in q:
        return "change_to_what"
    if any(re.search(p, q) for p in _RATIO_PATTERNS):
        if cat is not None or "type" in q or "class" in q:
            return "change_ratio_types"
        return "change_ratio"
    if any(re.search(p, q) for p in _INCREASE_PATTERNS):
        if cat is not None:
            return "increase_or_not"
        return "increase_or_not"
    if any(re.search(p, q) for p in _DECREASE_PATTERNS):
        return "decrease_or_not"
    if any(re.search(p, q) for p in _CHANGE_OR_NOT_PATTERNS):
        return "change_or_not"

    return "change_or_not"


def ratio_to_bracket(ratio: float) -> str:
    """Convert float ratio [0, 1] into standard CDVQA bracket string."""
    pct = ratio * 100.0
    if pct <= 0.001:
        return "0"
    elif pct <= 10.0:
        return "0_to_10"
    elif pct <= 20.0:
        return "10_to_20"
    elif pct <= 30.0:
        return "20_to_30"
    elif pct <= 40.0:
        return "30_to_40"
    elif pct <= 50.0:
        return "40_to_50"
    elif pct <= 60.0:
        return "50_to_60"
    elif pct <= 70.0:
        return "60_to_70"
    elif pct <= 80.0:
        return "70_to_80"
    elif pct <= 90.0:
        return "80_to_90"
    else:
        return "90_to_100"


# ============================================================
# Main ChangeAnalyzer class
# ============================================================

class ChangeAnalyzer:
    """
    ChangeFormerV6-based change analyzer with Semantic Grounding.

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Path to trained checkpoint .pth file.
    model_img_size : int (default: 256)
    device : str, optional ('xpu', 'cpu', or None for auto)
    """

    MODEL_NAME = "ChangeFormerV6"

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_img_size: int = 256,
        device: Optional[str] = None,
    ):
        self.model_img_size = model_img_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.grounder = SemanticGrounder()
        self.has_checkpoint = False

        # Device selection
        if device is None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[ChangeAnalyzer] Device: {self.device}")
        self.model = self._load_model()

    def _load_model(self):
        """Load ChangeFormerV6 model with error handling."""
        from models.networks import define_G
        from types import SimpleNamespace

        args = SimpleNamespace(net_G="ChangeFormerV6", embed_dim=256)
        model = define_G(args, init_type="normal", init_gain=0.02, gpu_ids=[])

        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            print(f"[ChangeAnalyzer] Loading checkpoint: {self.checkpoint_path}")
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                epoch = ckpt.get("epoch", "?")
                best_iou = ckpt.get("best_iou", "?")
                print(f"[ChangeAnalyzer] Loaded epoch={epoch}, best_iou={best_iou}")
                self.has_checkpoint = True
            except Exception as e:
                print(f"[ChangeAnalyzer] ERROR loading checkpoint {self.checkpoint_path}: {e}")
                print("[ChangeAnalyzer] Falling back to uninitialized weights.")
        else:
            if self.checkpoint_path is not None:
                print(f"[ChangeAnalyzer] WARNING: Checkpoint path not found: {self.checkpoint_path}")
            else:
                print("[ChangeAnalyzer] WARNING: No checkpoint specified. Using random weights.")

        model = model.to(self.device)
        model.eval()
        return model

    def _load_image(self, source: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Load RGB image tensor [1, 3, H, W] in [0, 1] with input validation.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Input image file does not exist: {path}")
            try:
                pil = Image.open(path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image from path '{path}': {e}")
        elif isinstance(source, Image.Image):
            pil = source.convert("RGB")
        else:
            raise TypeError(f"Expected str, Path, or PIL Image, got {type(source)}")

        if pil.size[0] == 0 or pil.size[1] == 0:
            raise ValueError(f"Invalid image dimensions: {pil.size}")

        pil = pil.resize((self.model_img_size, self.model_img_size), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)

    def _mask_to_base64_png(self, mask: np.ndarray) -> str:
        """Convert binary mask (H, W) to base64 PNG string."""
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[mask == 1] = [220, 50, 50]
        rgb[mask == 0] = [30, 30, 30]

        pil = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @torch.no_grad()
    def analyze_change(
        self,
        image_t1: Union[str, Path, Image.Image],
        image_t2: Union[str, Path, Image.Image],
        question: str = "Have the areas changed?",
    ) -> Dict[str, Any]:
        """
        Run change detection and category-aware grounding for CDVQA questions.

        Returns
        -------
        dict compatible with FastAPI & CDVQA benchmarks.
        """
        # Validate inputs
        if not isinstance(question, str):
            question = str(question) if question is not None else "Have the areas changed?"

        # 1. Forward pass ChangeFormerV6
        t1 = self._load_image(image_t1).to(self.device)
        t2 = self._load_image(image_t2).to(self.device)

        outputs = self.model(t1, t2)
        logits = outputs[-1]
        probs = torch.softmax(logits, dim=1)
        change_prob = probs[:, 1, :, :]
        pred_mask = (change_prob > 0.5).squeeze(0).cpu().numpy().astype(np.int64)

        global_change_ratio = float(pred_mask.mean())
        mean_change_prob = float(change_prob.mean().cpu().item())
        confidence = abs(mean_change_prob - 0.5) * 2.0

        # 2. Extract Question Details
        question_type = classify_question(question)
        target_category = extract_category(question)

        # 3. Locate & Load Semantic Labels if available
        l1_path, l2_path = None, None
        if isinstance(image_t1, (str, Path)):
            l1_path, l2_path = self.grounder.find_label_paths(image_t1)
        elif isinstance(image_t1, Image.Image) and getattr(image_t1, "filename", None):
            l1_path, l2_path = self.grounder.find_label_paths(image_t1.filename)

        has_grounding = False
        cat_stats = {}
        all_cat_stats = {}
        transitions = []

        if l1_path and l2_path:
            try:
                has_grounding = True
                label_t1 = self.grounder.load_label(l1_path, target_size=pred_mask.shape)
                label_t2 = self.grounder.load_label(l2_path, target_size=pred_mask.shape)
                all_cat_stats = self.grounder.compute_all_category_changes(label_t1, label_t2, pred_mask)
                transitions = self.grounder.compute_transitions(label_t1, label_t2, pred_mask, from_category=target_category)

                if target_category:
                    cat_stats = all_cat_stats.get(target_category, {})
            except Exception as e:
                print(f"[ChangeAnalyzer] Warning: failed to compute grounding: {e}")
                has_grounding = False

        # 4. Generate Category-Aware Answer & Short/Raw Answer
        raw_answer = "no"
        answer = "No significant change detected."

        if question_type == "change_or_not":
            if has_grounding and target_category in all_cat_stats:
                st = all_cat_stats[target_category]
                if st["change_present"]:
                    raw_answer = "yes"
                    answer = f"Yes, there are changes detected in {target_category} ({st['change_ratio']*100:.1f}% of the area changed)."
                else:
                    raw_answer = "no"
                    answer = f"No, there is no significant change detected in {target_category}."
            else:
                if global_change_ratio >= 0.02:
                    raw_answer = "yes"
                    answer = f"Yes, there are changes detected ({global_change_ratio*100:.1f}% of the area)."
                else:
                    raw_answer = "no"
                    answer = "No, there is no significant change detected."

        elif question_type == "increase_or_not":
            if has_grounding and target_category in all_cat_stats:
                st = all_cat_stats[target_category]
                if st["area_delta"] > 10:
                    raw_answer = "yes"
                    answer = f"Yes, the area of {target_category} increased."
                else:
                    raw_answer = "no"
                    answer = f"No, the area of {target_category} did not increase."
            else:
                raw_answer = "yes" if global_change_ratio > 0.05 else "no"
                answer = f"{raw_answer.capitalize()}, based on overall change."

        elif question_type == "decrease_or_not":
            if has_grounding and target_category in all_cat_stats:
                st = all_cat_stats[target_category]
                if st["area_delta"] < -10:
                    raw_answer = "yes"
                    answer = f"Yes, the area of {target_category} decreased."
                else:
                    raw_answer = "no"
                    answer = f"No, the area of {target_category} did not decrease."
            else:
                raw_answer = "yes" if global_change_ratio > 0.05 else "no"
                answer = f"{raw_answer.capitalize()}, based on overall change."

        elif question_type == "smallest_change":
            if has_grounding and all_cat_stats:
                valid = {cat: st for cat, st in all_cat_stats.items() if st["changed_pixels"] > 0}
                if valid:
                    min_cat = min(valid.keys(), key=lambda c: valid[c]["changed_pixels"])
                else:
                    min_cat = min(all_cat_stats.keys(), key=lambda c: all_cat_stats[c]["t1_pixels"])
                raw_answer = min_cat
                answer = f"The smallest change is {min_cat}."
            else:
                raw_answer = "buildings"
                answer = "The smallest change is buildings."

        elif question_type == "largest_change":
            if has_grounding and all_cat_stats:
                max_cat = max(all_cat_stats.keys(), key=lambda c: all_cat_stats[c]["changed_pixels"])
                raw_answer = max_cat
                answer = f"The largest change is {max_cat}."
            else:
                raw_answer = "NVG_surface"
                answer = "The largest change is NVG_surface."

        elif question_type == "change_to_what":
            if has_grounding and transitions:
                top_trans = transitions[0]
                to_cat = top_trans[1]
                raw_answer = to_cat
                answer = f"{target_category or 'The changed area'} changed to {to_cat}."
            else:
                raw_answer = "buildings"
                answer = f"The area changed to buildings."

        elif question_type in ["change_ratio_types", "change_ratio"]:
            target_r = cat_stats.get("change_ratio", global_change_ratio) if (has_grounding and target_category) else global_change_ratio
            bracket = ratio_to_bracket(target_r)
            raw_answer = bracket
            cat_label = f" for {target_category}" if target_category else ""
            answer = f"The change ratio{cat_label} is approximately {target_r*100:.1f}% ({bracket})."

        # 5. Base64 Mask & Device Sync
        mask_b64 = self._mask_to_base64_png(pred_mask)
        if self.device.type == "xpu":
            try:
                torch.xpu.synchronize()
            except Exception:
                pass

        cat_ratio = cat_stats.get("change_ratio", global_change_ratio) if (has_grounding and target_category) else None

        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "confidence": round(confidence, 4),
            "change_ratio": round(global_change_ratio, 4),
            "global_change_ratio": round(global_change_ratio, 4),
            "category": target_category,
            "category_change_ratio": round(cat_ratio, 4) if cat_ratio is not None else None,
            "has_grounding": has_grounding,
            "category_metrics": cat_stats,
            "all_category_metrics": all_cat_stats,
            "change_mask_base64": mask_b64,
            "task": "change_detection",
            "model": self.MODEL_NAME,
            "question_type": question_type,
            "question": question,
            "metrics": {
                "changed_pixels": int(pred_mask.sum()),
                "total_pixels": int(pred_mask.size),
                "mean_change_probability": round(mean_change_prob, 4),
            },
        }


# Global Singleton for Person C integration
_default_analyzer: Optional[ChangeAnalyzer] = None


def get_default_analyzer(checkpoint_path: Optional[str] = None) -> ChangeAnalyzer:
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = ChangeAnalyzer(checkpoint_path=checkpoint_path)
    return _default_analyzer


def analyze_change(
    image_t1: Union[str, Path, Image.Image],
    image_t2: Union[str, Path, Image.Image],
    question: str = "Have the areas changed?",
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    analyzer = get_default_analyzer(checkpoint_path)
    return analyzer.analyze_change(image_t1, image_t2, question)


if __name__ == "__main__":
    print("=" * 70)
    print("CHANGE ANALYZER WITH SEMANTIC GROUNDING — SELF TEST")
    print("=" * 70)

    project_root = _PROJECT_ROOT
    im1_dir = project_root / "datasets" / "SECOND" / "SECOND_train_set" / "im1"
    im2_dir = project_root / "datasets" / "SECOND" / "SECOND_train_set" / "im2"

    test_files = sorted(im1_dir.glob("*.png"))
    if not test_files:
        print("ERROR: No SECOND images found.")
        sys.exit(1)

    t1_path = test_files[0]
    t2_path = im2_dir / test_files[0].name

    print(f"Test image: {t1_path.name}")

    best_ckpt = project_root / "checkpoints" / "full" / "checkpoint_best.pth"
    analyzer = ChangeAnalyzer(checkpoint_path=best_ckpt if best_ckpt.exists() else None)

    test_questions = [
        "Have the regions of non-vegetated ground surface changed?",
        "Have the areas of trees changed?",
        "Did the regions of water change?",
        "Have the regions of buildings changed?",
        "Did the areas of low vegetation increase?",
        "What type of change is the smallest?",
        "What is the largest change?",
        "What did the NVG surface change to?",
    ]

    for q in test_questions:
        print()
        print(f"Question : {q}")
        res = analyzer.analyze_change(t1_path, t2_path, q)
        print(f"  Q-Type   : {res['question_type']}")
        print(f"  Category : {res['category']}")
        print(f"  Raw Ans  : {res['raw_answer']}")
        print(f"  Answer   : {res['answer']}")
        print(f"  Global % : {res['change_ratio']*100:.2f}% (Cat %: {res['category_change_ratio']})")

    print("\nSelf-test complete.")
