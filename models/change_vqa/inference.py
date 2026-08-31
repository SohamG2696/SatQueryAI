"""
SatQuery AI — Bi-Temporal Change-VQA Inference Module.
=======================================================
Person B Integration: ChangeFormerV6 + Semantic Grounding.

Replaces the original stub BiTemporalChangeNetwork with the trained
ChangeFormerV6 model (50 epochs, SECOND dataset, IoU=0.4678).

Design
------
* predict() accepts raw image file paths (str | Path | PIL.Image), NOT
  pre-processed tensors. The backend adapter (backend/app/models/change_vqa.py)
  passes raw paths directly into predict().
* All image resizing (to 256x256) is handled internally by ChangeAnalyzer.
* The checkpoint is expected at:
    models/change_vqa/weights/checkpoint_best.pth
  See weights/DOWNLOAD_WEIGHTS.md for how to obtain it.
* SemanticGrounder locates SECOND label maps automatically. Grounding is
  silently disabled for images without matching label maps.
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from typing import Any, Union

from PIL import Image

# ── Absolute paths ─────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # .../models/change_vqa/
_CHANGEFORMER_DIR = _THIS_DIR / "changeformer"       # .../models/change_vqa/changeformer/
_SATQUERY_ROOT = _THIS_DIR.parent.parent             # .../SatQueryAI/
_DEFAULT_CHECKPOINT = _THIS_DIR / "weights" / "checkpoint_best.pth"


def _setup_paths() -> None:
    """
    Ensure changeformer/, change_vqa/, and SatQueryAI root are on sys.path,
    and explicitly load ChangeFormer submodules into sys.modules under `models.<name>`.

    ChangeFormer source files use cross-imports like:
        from models.networks import define_G
        from models.SiamUnet_diff import SiamUnet_diff
    We load these submodules directly from file paths into sys.modules.
    """
    if str(_SATQUERY_ROOT) not in sys.path:
        sys.path.append(str(_SATQUERY_ROOT))

    if str(_THIS_DIR) in sys.path:
        sys.path.remove(str(_THIS_DIR))
    sys.path.insert(0, str(_THIS_DIR))

    if str(_CHANGEFORMER_DIR) in sys.path:
        sys.path.remove(str(_CHANGEFORMER_DIR))
    sys.path.insert(0, str(_CHANGEFORMER_DIR))

    # Pre-register ChangeFormer submodules directly from file paths into sys.modules
    cf_modules = [
        "help_funcs", "resnet", "pixel_shuffel_up", "SiamUnet_diff",
        "SiamUnet_conc", "Unet", "DTCDSCN", "basic_model",
        "ChangeFormerBaseNetworks", "ChangeFormer", "networks"
    ]
    for name in cf_modules:
        file_path = _CHANGEFORMER_DIR / f"{name}.py"
        if file_path.exists():
            full_name = f"models.{name}"
            if full_name not in sys.modules:
                try:
                    spec = importlib.util.spec_from_file_location(full_name, str(file_path))
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[full_name] = mod
                    sys.modules[name] = mod
                    spec.loader.exec_module(mod)
                except Exception as e:
                    pass


_setup_paths()

from change_analyzer import ChangeAnalyzer  # noqa: E402 — after path setup


class ChangeVQAInferenceEngine:
    """
    Thin adapter wrapping ChangeAnalyzer (ChangeFormerV6 + SemanticGrounder).

    This is the class expected by backend/app/models/change_vqa.py.

    Parameters
    ----------
    weights_path : str | Path | None
        Path to checkpoint_best.pth. Defaults to
        models/change_vqa/weights/checkpoint_best.pth.
    vocab_path : str | Path | None
        Ignored — kept for API compatibility with original stub.
    device : any
        Ignored — ChangeAnalyzer auto-detects XPU → CPU.
        Kept for API compatibility.
    """

    def __init__(
        self,
        weights_path: "str | Path | None" = None,
        vocab_path: "str | Path | None" = None,
        device: Any = None,
    ):
        ckpt = Path(weights_path) if weights_path else _DEFAULT_DEFAULT_CHECKPOINT()
        if not ckpt or not ckpt.exists():
            ckpt = None  # ChangeAnalyzer will warn and use random weights

        self._analyzer = ChangeAnalyzer(checkpoint_path=ckpt)
        # Expose device so backend/app/models/change_vqa.py can read it
        self.device = self._analyzer.device

    def predict(
        self,
        image_t1: "Union[str, Path, Image.Image]",
        image_t2: "Union[str, Path, Image.Image]",
        query: str,
        dates: "list[str] | None" = None,
    ) -> "dict[str, Any]":
        """
        Run ChangeFormerV6 change detection + category-aware semantic grounding.

        Parameters
        ----------
        image_t1 : str | Path | PIL.Image
            T1 (before) image — raw path or PIL Image.
            Do NOT pass a pre-processed tensor.
        image_t2 : str | Path | PIL.Image
            T2 (after) image.
        query : str
            Natural-language CDVQA question.
        dates : list[str] | None
            Optional acquisition dates (stored in response for traceability).

        Returns
        -------
        dict — Full Person B response schema:
            answer, raw_answer, confidence,
            change_ratio, global_change_ratio,
            category, category_change_ratio,
            has_grounding, change_mask_base64,
            task, model, question_type, question,
            metrics, category_metrics, all_category_metrics,
            parameters
        """
        result = self._analyzer.analyze_change(image_t1, image_t2, query)
        result["parameters"] = {"query": query, "dates": dates}
        return result


def _DEFAULT_DEFAULT_CHECKPOINT():
    p1 = _DEFAULT_CHECKPOINT
    p2 = _SATQUERY_ROOT.parent / "checkpoints" / "full" / "checkpoint_best.pth"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return p1
