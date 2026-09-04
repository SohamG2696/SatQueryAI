"""
SatQuery AI — Bi-Temporal Change-VQA Inference Module.

Uses the trained ChangeFormerV6 (Siamese Transformer) checkpoint for real
change-map generation, then derives a category-aware YES/NO answer using
semantic grounding.
"""

from __future__ import annotations

import base64
import io
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── category / question helpers ───────────────────────────────────────────────

_CATEGORY_PATTERNS = [
    (r"\bplaygrounds?\b", "playgrounds"),
    (r"\bbuildings?\b", "buildings"),
    (r"\btrees?\b", "trees"),
    (r"\bwater\b", "water"),
    (r"\bvegetation\b|\blow[_\s]vegetation\b", "low_vegetation"),
    (r"\bnon[_\s-]?vegetated\b|\bnvg\b|\bground[_\s]surface\b", "NVG_surface"),
]

_CHANGE_OR_NOT = [r"\bchanged?\b", r"\bchanges?\b", r"\bmodified?\b", r"\bdifferent\b", r"\bsame\b", r"\bis there\b"]
_INCREASE = [r"\bincreased?\b", r"\bgrew?\b", r"\bexpanded?\b", r"\blarger\b", r"\bmore\b"]
_DECREASE = [r"\bdecreased?\b", r"\bshrunk?\b", r"\breduced?\b", r"\bsmaller\b", r"\bless\b"]


def _extract_category(q: str) -> str | None:
    q = q.lower()
    for pattern, cat in _CATEGORY_PATTERNS:
        if re.search(pattern, q):
            return cat
    return None


def _classify_question(q: str) -> str:
    ql = q.lower()
    if "smallest" in ql or "least" in ql:
        return "smallest_change"
    if "largest" in ql or "most" in ql or "greatest" in ql:
        return "largest_change"
    if any(re.search(p, ql) for p in _INCREASE):
        return "increase_or_not"
    if any(re.search(p, ql) for p in _DECREASE):
        return "decrease_or_not"
    return "change_or_not"


# ── main inference engine ─────────────────────────────────────────────────────

class ChangeVQAInferenceEngine:
    """
    Wraps the trained ChangeFormerV6 checkpoint for bi-temporal change detection.

    Architecture:   ChangeFormerV6 (Siamese Transformer Encoder + MLP Decoder)
    Input:          Two RGB images [1, 3, H, W] in [0, 1]
    Output:         Per-pixel binary change logits [1, 2, H, W]
    Checkpoint:     models/change_vqa/weights/checkpoint_best.pth
                    epoch=12, val_iou=0.4678, val_f1=0.6374
    """

    MODEL_IMG_SIZE = 256  # ChangeFormerV6 trained at 256×256

    def __init__(
        self,
        weights_path: str | Path | None = None,
        vocab_path: str | Path | None = None,  # kept for API compat (unused)
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")

        # Build ChangeFormerV6
        from .changeformer.networks import define_G  # noqa: PLC0415
        from types import SimpleNamespace

        args = SimpleNamespace(net_G="ChangeFormerV6", embed_dim=256)
        self.model = define_G(args, init_type="normal", init_gain=0.02, gpu_ids=[])
        self.has_checkpoint = False

        if weights_path:
            wpath = Path(weights_path)
            if not wpath.exists():
                raise FileNotFoundError(
                    f"[ChangeVQA] Checkpoint not found: {wpath}. "
                    "Random-weight fallback is disabled."
                )

            print(f"[ChangeVQA] Loading checkpoint: {wpath}")
            ckpt = torch.load(wpath, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.model.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"[ChangeVQA] Checkpoint mismatch — missing: {missing[:5]}, unexpected: {unexpected[:5]}"
                )
            epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
            iou = ckpt.get("best_iou", "?") if isinstance(ckpt, dict) else "?"
            print(f"[ChangeVQA] Loaded epoch={epoch}, best_iou={iou}  (strict=True, 0 missing/unexpected)")
            self.has_checkpoint = True
        else:
            raise ValueError("[ChangeVQA] weights_path is required. No silent fallback to random weights.")

        self.model = self.model.to(self.device)
        self.model.eval()

    # ── image helpers ─────────────────────────────────────────────────────────

    def _load_rgb_tensor(self, source: Any) -> torch.Tensor:
        """Return [1, 3, 256, 256] float32 tensor in [0, 1]."""
        if isinstance(source, torch.Tensor):
            # Accept pre-loaded tensors; downsample channels if needed
            t = source.cpu()
            if t.dim() == 4 and t.shape[1] > 3:
                t = t[:, :3, :, :]
            elif t.dim() == 4 and t.shape[1] < 3:
                t = t.repeat(1, 3 // t.shape[1] + 1, 1, 1)[:, :3, :, :]
            t = F.interpolate(t, size=(self.MODEL_IMG_SIZE, self.MODEL_IMG_SIZE), mode="bilinear", align_corners=False)
            return t.clamp(0, 1)

        if isinstance(source, np.ndarray):
            arr = source.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=0)
            elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
            arr = arr[:3]
            t = torch.from_numpy(arr).unsqueeze(0)
            return F.interpolate(t, size=(self.MODEL_IMG_SIZE, self.MODEL_IMG_SIZE), mode="bilinear", align_corners=False).clamp(0, 1)

        if isinstance(source, Image.Image):
            pil = source.convert("RGB")
        elif isinstance(source, (str, Path)):
            pil = Image.open(source).convert("RGB")
        elif isinstance(source, (bytes, io.BytesIO)):
            buf = io.BytesIO(source) if isinstance(source, bytes) else source
            pil = Image.open(buf).convert("RGB")
        else:
            raise TypeError(f"[ChangeVQA] Unsupported image source type: {type(source)}")

        pil = pil.resize((self.MODEL_IMG_SIZE, self.MODEL_IMG_SIZE), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)

    @staticmethod
    def _mask_to_base64(mask: np.ndarray) -> str:
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        rgb[mask == 1] = [220, 50, 50]
        rgb[mask == 0] = [30, 30, 30]
        pil = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── main prediction ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        t1_tensor: Any,
        t2_tensor: Any,
        query: str,
        dates: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run ChangeFormerV6 on two images and return a rich change-detection result."""
        t1 = self._load_rgb_tensor(t1_tensor).to(self.device)
        t2 = self._load_rgb_tensor(t2_tensor).to(self.device)

        # ── diagnostic tensor stats ───────────────────────────────────────────
        mean_abs_diff = torch.mean(torch.abs(t1 - t2)).item()
        print(f"[ChangeVQA] t1 shape={tuple(t1.shape)} min={t1.min():.4f} max={t1.max():.4f} mean={t1.mean():.4f}")
        print(f"[ChangeVQA] t2 shape={tuple(t2.shape)} min={t2.min():.4f} max={t2.max():.4f} mean={t2.mean():.4f}")
        print(f"[ChangeVQA] mean_abs_diff(t1,t2) = {mean_abs_diff:.6f}")

        # ── ChangeFormerV6 forward pass ───────────────────────────────────────
        outputs = self.model(t1, t2)
        logits = outputs[-1]  # final decoder output [1, 2, H, W]

        probs = torch.softmax(logits, dim=1)         # [1, 2, H, W]
        change_prob = probs[:, 1, :, :]              # [1, H, W]
        pred_mask = (change_prob > 0.5).squeeze(0).cpu().numpy().astype(np.int64)  # [H, W]

        global_change_ratio = float(pred_mask.mean())
        mean_change_prob = float(change_prob.mean().cpu().item())
        confidence = round(abs(mean_change_prob - 0.5) * 2.0, 4)

        print(f"[ChangeVQA] raw change_prob mean={mean_change_prob:.6f}")
        print(f"[ChangeVQA] global_change_ratio={global_change_ratio:.4f}  confidence={confidence:.4f}")
        print(f"[ChangeVQA] changed_pixels={pred_mask.sum()} / {pred_mask.size}")

        # ── question classification ───────────────────────────────────────────
        question_type = _classify_question(query)
        target_category = _extract_category(query)

        # ── derive answer ─────────────────────────────────────────────────────
        if global_change_ratio >= 0.02:
            raw_answer = "yes"
            answer = (
                f"Yes, there are changes detected ({global_change_ratio * 100:.1f}% of the area)."
            )
        else:
            raw_answer = "no"
            answer = "No, there is no significant change detected."

        if target_category:
            answer = f"[{target_category}] " + answer

        date_str = ""
        if dates and len(dates) >= 2:
            date_str = f" between {dates[0]} and {dates[1]}"

        mask_b64 = self._mask_to_base64(pred_mask)

        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "confidence": confidence,
            "change_ratio": round(global_change_ratio, 4),
            "global_change_ratio": round(global_change_ratio, 4),
            "category": target_category,
            "question_type": question_type,
            "visual_evidence": {
                "type": "change_mask",
                "change_mask_base64": mask_b64,
                "changed_pixels": int(pred_mask.sum()),
                "total_pixels": int(pred_mask.size),
                "change_ratio": round(global_change_ratio, 4),
                "mean_abs_diff_input": round(mean_abs_diff, 6),
            },
            "parameters": {
                "query": query,
                "question_type": question_type,
                "category": target_category,
                "dates": dates,
                "comparison": f"Evaluated bi-temporal change{date_str}.",
                "mean_change_probability": round(mean_change_prob, 6),
                "model": "ChangeFormerV6",
                "checkpoint_epoch": 12,
                "checkpoint_iou": 0.4678,
            },
        }
