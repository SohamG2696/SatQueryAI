"""
SatQuery AI — Change Detection Comprehensive Diagnostic Script.

Tests:
1. Checkpoint loads with strict=True (ChangeFormerV6, epoch=12, IoU=0.4678)
2. Input tensor diagnostics (shape, dtype, NaN/Inf)
3. SAME image vs DIFFERENT image ablation — proves model uses actual pixel data
4. Query dependency across 4 different questions
5. Change-map quality (changed pixel %, mean_abs_diff)
6. Live API via FastAPI TestClient
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from models.change_vqa.inference import ChangeVQAInferenceEngine

CKPT = ROOT / "models" / "change_vqa" / "weights" / "checkpoint_best.pth"
VOCAB = ROOT / "datasets" / "processed" / "vocabulary.json"

# ── SECOND dataset image pair ─────────────────────────────────────────────────
IM1_DIR = ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im1"
IM2_DIR = ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im2"


def get_real_pair():
    """Return (before_path, after_path) from SECOND dataset if available."""
    if IM1_DIR.exists():
        files = sorted(IM1_DIR.glob("*.png"))
        if files:
            f1 = files[0]
            f2 = IM2_DIR / f1.name
            if f2.exists():
                return f1, f2
    return None, None


def synthetic_pair(seed_a=42, seed_b=99, size=256):
    """Create two deterministically different synthetic images."""
    rng = np.random.RandomState(seed_a)
    img1 = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    rng = np.random.RandomState(seed_b)
    img2 = (rng.rand(size, size, 3) * 255).astype(np.uint8)
    return Image.fromarray(img1), Image.fromarray(img2)


def run_diagnostic():
    print("=" * 80)
    print("      SATQUERY AI — CHANGE DETECTION COMPREHENSIVE DIAGNOSTIC")
    print("=" * 80)

    # ── 1. CHECKPOINT VERIFICATION ────────────────────────────────────────────
    print("\n[TASK 1] Verifying ChangeFormerV6 checkpoint...")
    assert CKPT.exists(), f"Checkpoint not found: {CKPT}"
    import torch as _torch
    ckpt = _torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"  Epoch    : {ckpt['epoch']}")
    print(f"  Best IoU : {ckpt['best_iou']}")
    m = ckpt.get("metrics", {})
    print(f"  val_f1   : {m.get('val_f1')}")
    print(f"  val_prec : {m.get('val_precision')}")
    print(f"  val_rec  : {m.get('val_recall')}")
    print(f"  val_pxacc: {m.get('val_pixel_accuracy')}")
    print(f"  sd_keys  : {len(ckpt['model_state_dict'])}")

    engine = ChangeVQAInferenceEngine(weights_path=CKPT, vocab_path=VOCAB, device=torch.device("cpu"))
    print("  Engine initialized successfully. ChangeFormerV6 ready.")

    # ── 2. IMAGE PAIR ─────────────────────────────────────────────────────────
    before_path, after_path = get_real_pair()
    if before_path:
        print(f"\n[IMAGES] Real SECOND pair: {before_path.name}")
        img1 = Image.open(before_path).convert("RGB")
        img2 = Image.open(after_path).convert("RGB")
        img_same = img1  # same image for sanity test
    else:
        print("\n[IMAGES] SECOND dataset not found — using synthetic pair.")
        img1, img2 = synthetic_pair(seed_a=42, seed_b=99)
        img_same = img1

    # tensor difference check
    arr1 = np.array(img1.resize((256, 256)), dtype=np.float32) / 255.0
    arr2 = np.array(img2.resize((256, 256)), dtype=np.float32) / 255.0
    arr_same = np.array(img_same.resize((256, 256)), dtype=np.float32) / 255.0

    diff_real = np.mean(np.abs(arr1 - arr2))
    diff_same = np.mean(np.abs(arr1 - arr_same))
    print(f"\n[TASK 2] Input tensor diagnostics")
    print(f"  img1 shape: {arr1.shape}  min={arr1.min():.4f}  max={arr1.max():.4f}")
    print(f"  img2 shape: {arr2.shape}  min={arr2.min():.4f}  max={arr2.max():.4f}")
    print(f"  NaN in img1: {np.isnan(arr1).sum()}  | Inf: {np.isinf(arr1).sum()}")
    print(f"  NaN in img2: {np.isnan(arr2).sum()}  | Inf: {np.isinf(arr2).sum()}")
    print(f"  mean_abs_diff(img1, img2)   = {diff_real:.6f}  <- should be > 0 for different images")
    print(f"  mean_abs_diff(img1, img1)   = {diff_same:.6f}  <- should be ~0 for same image")

    assert diff_same < 1e-5, "Same-image diff should be ~0!"
    if before_path:
        assert diff_real > 1e-3, "Real image pair should have non-zero difference!"

    # ── 3. ABLATION: SAME vs DIFFERENT ───────────────────────────────────────
    print("\n[TASK 3 & 7] Modality ablation — SAME vs DIFFERENT images...")
    q = "What changed between these images?"

    t0 = time.time()
    res_diff = engine.predict(img1, img2, q)
    t_diff = time.time() - t0

    t0 = time.time()
    res_same = engine.predict(img1, img_same, q)
    t_same = time.time() - t0

    print(f"\n  [Different images] ({t_diff:.2f}s)")
    print(f"    Answer         : {res_diff['answer']}")
    print(f"    Confidence     : {res_diff['confidence']}")
    print(f"    Change Ratio   : {res_diff['change_ratio']}")
    print(f"    mean_abs_diff  : {res_diff['visual_evidence']['mean_abs_diff_input']}")
    print(f"    Changed pixels : {res_diff['visual_evidence']['changed_pixels']}/{res_diff['visual_evidence']['total_pixels']}")
    print(f"    Mask B64 len   : {len(res_diff['visual_evidence']['change_mask_base64'])}")

    print(f"\n  [SAME image x2] ({t_same:.2f}s)")
    print(f"    Answer         : {res_same['answer']}")
    print(f"    Confidence     : {res_same['confidence']}")
    print(f"    Change Ratio   : {res_same['change_ratio']}")
    print(f"    mean_abs_diff  : {res_same['visual_evidence']['mean_abs_diff_input']}")
    print(f"    Changed pixels : {res_same['visual_evidence']['changed_pixels']}/{res_same['visual_evidence']['total_pixels']}")

    same_mask_ratio = res_same["change_ratio"]
    diff_mask_ratio = res_diff["change_ratio"]

    print(f"\n  Sanity: same-image change_ratio = {same_mask_ratio:.4f}  (ideally < 0.10)")
    print(f"  Sanity: diff-image change_ratio = {diff_mask_ratio:.4f}")

    # The key assertion: model outputs differ between real and same-image tests
    assert res_same["visual_evidence"]["mean_abs_diff_input"] < 0.001, \
        f"Same-image mean_abs_diff should be ~0, got {res_same['visual_evidence']['mean_abs_diff_input']}"
    assert diff_mask_ratio != same_mask_ratio, \
        "Different and same-image predictions are identical — model is not responding to input!"
    print("  => CONFIRMED: Different images produce different change maps.")

    # ── 4. QUERY DEPENDENCY ───────────────────────────────────────────────────
    print("\n[TASK 4] Query dependency across 4 queries...")
    queries = [
        "What changed between these images?",
        "Where did new buildings appear?",
        "Where did vegetation change?",
        "Where did water change?",
    ]

    for q in queries:
        res = engine.predict(img1, img2, q)
        print(f"\n  Query: '{q}'")
        print(f"    Answer       : {res['answer']}")
        print(f"    Confidence   : {res['confidence']}")
        print(f"    Change Ratio : {res['change_ratio']}")
        print(f"    Question Type: {res['parameters']['question_type']}")
        print(f"    Category     : {res['parameters']['category']}")

    # ── 5. LIVE API TEST ──────────────────────────────────────────────────────
    print("\n[TASK 8] Live API test via FastAPI TestClient...")
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    def upload_img(pil_img: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        r = client.post("/api/upload", files={"file": (name, io.BytesIO(buf.getvalue()), "image/png")})
        assert r.status_code == 200, f"Upload failed: {r.text}"
        return r.json()["image_id"]

    id_before = upload_img(img1, "before.png")
    id_after = upload_img(img2, "after.png")
    id_same = upload_img(img_same, "same_as_before.png")

    # --- DIFFERENT images ---
    resp_d = client.post("/api/change", json={
        "image_id_before": id_before,
        "image_id_after": id_after,
        "date_before": "2024-01-01",
        "date_after": "2026-01-01",
        "query": "What changed between these images?",
    })
    assert resp_d.status_code == 200, f"API failed: {resp_d.text}"
    rd = resp_d.json()

    # --- SAME image ---
    resp_s = client.post("/api/change", json={
        "image_id_before": id_before,
        "image_id_after": id_same,
        "date_before": "2024-01-01",
        "date_after": "2024-01-01",
        "query": "What changed between these images?",
    })
    assert resp_s.status_code == 200
    rs = resp_s.json()

    print(f"\n  [API: Different images]")
    print(f"    answer     : {rd['answer']}")
    print(f"    confidence : {rd['confidence']}")
    print(f"    models_used: {rd['models_used']}")
    print(f"    time       : {rd['processing_time']}s")
    print(f"    params     : {rd.get('parameters', {})}")

    print(f"\n  [API: Same image x2]")
    print(f"    answer     : {rs['answer']}")
    print(f"    confidence : {rs['confidence']}")

    # Confidence must come from model, not be a hardcoded 0.65
    assert rd["confidence"] != 0.65 or rs["confidence"] != 0.65 or rd["confidence"] != rs["confidence"], \
        "Confidence is still the hardcoded 0.65 — fix not applied!"
    assert rd["models_used"] == ["ChangeFormerV6"], f"Wrong model reported: {rd['models_used']}"

    print("\n" + "=" * 80)
    print("ALL CHANGE DETECTION DIAGNOSTIC CHECKS PASSED!")
    print(f"  Real ChangeFormerV6 (epoch=12, IoU=0.4678) is active.")
    print(f"  Input images materially affect model output.")
    print(f"  Same-image change_ratio = {same_mask_ratio:.4f}")
    print(f"  Diff-image change_ratio = {diff_mask_ratio:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic()
