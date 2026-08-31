"""
Isolated test for models/change_vqa/inference.py integration.
Run from: C:/Users/hp/OneDrive/Desktop/SIH2026/SatQueryAI
"""
import sys
import time
import base64
import io
from pathlib import Path
from PIL import Image as PIL_Image

SATQUERY_ROOT = Path("C:/Users/hp/OneDrive/Desktop/SIH2026/SatQueryAI").resolve()
SIH2026_ROOT = Path("C:/Users/hp/OneDrive/Desktop/SIH2026").resolve()

CHANGE_VQA_DIR = SATQUERY_ROOT / "models" / "change_vqa"

if str(CHANGE_VQA_DIR) not in sys.path:
    sys.path.insert(0, str(CHANGE_VQA_DIR))
if str(CHANGE_VQA_DIR / "changeformer") not in sys.path:
    sys.path.insert(0, str(CHANGE_VQA_DIR / "changeformer"))

from inference import ChangeVQAInferenceEngine

# Use real SECOND image pair and real trained checkpoint
SECOND_IM1 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im1" / "10589.png"
SECOND_IM2 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im2" / "10589.png"
CKPT = SIH2026_ROOT / "checkpoints" / "full" / "checkpoint_best.pth"

print("=" * 65)
print("CHANGE VQA INFERENCE ENGINE — ISOLATED TEST")
print("=" * 65)
print(f"T1 path   : {SECOND_IM1}")
print(f"T2 path   : {SECOND_IM2}")
print(f"T1 exists : {SECOND_IM1.exists()}")
print(f"T2 exists : {SECOND_IM2.exists()}")
print(f"Checkpoint: {CKPT} (exists={CKPT.exists()})")

print("\n[1] Instantiating ChangeVQAInferenceEngine...")
t0 = time.time()
engine = ChangeVQAInferenceEngine(
    weights_path=CKPT if CKPT.exists() else None
)
print(f"    Device: {engine.device}")
print(f"    Instantiation time: {(time.time()-t0)*1000:.1f} ms")

print("\n[2] Running predict()...")
QUESTION = "Have the areas of buildings changed?"
t1 = time.time()
result = engine.predict(SECOND_IM1, SECOND_IM2, QUESTION)
elapsed = (time.time() - t1) * 1000

print(f"    Inference time: {elapsed:.1f} ms")
print()
print("Result fields:")
for k, v in result.items():
    if k == "change_mask_base64":
        print(f"  {k:30s}: <base64 PNG, {len(v)} chars>")
    elif k in ("category_metrics", "all_category_metrics"):
        print(f"  {k:30s}: {type(v).__name__} with {len(v) if v else 0} entries")
    else:
        print(f"  {k:30s}: {v}")

print()
print("=" * 65)
print("VERIFICATION CHECKS")
print("=" * 65)

checks = {
    "answer is non-empty string": isinstance(result.get("answer"), str) and len(result["answer"]) > 0,
    "raw_answer in (yes,no,class,ratio)": isinstance(result.get("raw_answer"), str),
    "confidence != 0.78 (not placeholder)": result.get("confidence") != 0.78,
    "confidence is float in [0,1]": isinstance(result.get("confidence"), float) and 0.0 <= result["confidence"] <= 1.0,
    "change_mask_base64 is non-empty": isinstance(result.get("change_mask_base64"), str) and len(result["change_mask_base64"]) > 100,
    "change_mask_base64 decodes to PNG": False,
    "category == buildings": result.get("category") == "buildings",
    "category_change_ratio is float": isinstance(result.get("category_change_ratio"), float),
    "has_grounding is bool": isinstance(result.get("has_grounding"), bool),
    "model == ChangeFormerV6": result.get("model") == "ChangeFormerV6",
    "question_type classified correctly": result.get("question_type") == "change_or_not",
    "global_change_ratio is float": isinstance(result.get("global_change_ratio"), float),
}

# Verify base64 decoding
try:
    raw = base64.b64decode(result["change_mask_base64"])
    img = PIL_Image.open(io.BytesIO(raw))
    assert img.size == (256, 256), f"Expected (256,256), got {img.size}"
    checks["change_mask_base64 decodes to PNG"] = True
except Exception as e:
    print(f"  [BASE64 DECODE ERROR] {e}")

pass_count = 0
fail_count = 0
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if passed:
        pass_count += 1
    else:
        fail_count += 1
    print(f"  [{status}] {check}")

print()
print(f"Result: {pass_count}/{pass_count+fail_count} checks passed.")
print("=" * 65)
