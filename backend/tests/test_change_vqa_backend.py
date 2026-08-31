"""
Integration test for backend/app/models/change_vqa.py.
Run from: C:/Users/hp/OneDrive/Desktop/SIH2026/SatQueryAI
"""
import sys
import time
import base64
import io
import importlib.util
from pathlib import Path
from PIL import Image as PIL_Image

SATQUERY_ROOT = Path("C:/Users/hp/OneDrive/Desktop/SIH2026/SatQueryAI").resolve()
BACKEND_ROOT = SATQUERY_ROOT / "backend"
SIH2026_ROOT = SATQUERY_ROOT.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SATQUERY_ROOT) not in sys.path:
    sys.path.insert(0, str(SATQUERY_ROOT))

# Load backend/app/models/change_vqa.py directly without triggering app.models.__init__
vqa_adapter_path = BACKEND_ROOT / "app" / "models" / "change_vqa.py"
spec = importlib.util.spec_from_file_location("backend_change_vqa_adapter", str(vqa_adapter_path))
change_vqa_mod = importlib.util.module_from_spec(spec)
sys.modules["backend_change_vqa_adapter"] = change_vqa_mod
spec.loader.exec_module(change_vqa_mod)

run_module = change_vqa_mod.run_module

# Target test image pair
SECOND_IM1 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im1" / "10589.png"
SECOND_IM2 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im2" / "10589.png"

print("=" * 65)
print("BACKEND CHANGE VQA ADAPTER INTEGRATION TEST")
print("=" * 65)
print(f"T1 Path: {SECOND_IM1} (exists={SECOND_IM1.exists()})")
print(f"T2 Path: {SECOND_IM2} (exists={SECOND_IM2.exists()})")

# 1. Test direct run_module call
print("\n[1] Testing backend/app/models/change_vqa.py -> run_module()...")
QUESTION = "Have the areas of buildings changed?"
t0 = time.time()
res1 = run_module([SECOND_IM1, SECOND_IM2], QUESTION, metadata={"dates": ["2022-01-01", "2025-01-01"]})
elapsed1 = (time.time() - t0) * 1000

print(f"    Elapsed               : {elapsed1:.1f} ms")
print(f"    Answer                : {res1.get('answer')}")
print(f"    Raw Answer            : {res1.get('raw_answer')}")
print(f"    Confidence            : {res1.get('confidence')}")
print(f"    Category              : {res1.get('category')}")
print(f"    Category Change Ratio : {res1.get('category_change_ratio')}")
print(f"    Global Change Ratio   : {res1.get('global_change_ratio')}")
print(f"    Has Grounding         : {res1.get('has_grounding')}")
print(f"    Model Name            : {res1.get('model_name')}")
print(f"    Question Type         : {res1.get('question_type')}")
print(f"    Change Mask Base64    : <{len(res1.get('change_mask_base64', ''))} chars>")

print("\n" + "=" * 65)
print("VERIFICATION CHECKS")
print("=" * 65)

checks = {
    "category == buildings": res1.get("category") == "buildings",
    "category_change_ratio ~ 0.6337": res1.get("category_change_ratio") is not None and abs(res1["category_change_ratio"] - 0.6337) < 0.05,
    "global_change_ratio ~ 0.18": res1.get("global_change_ratio") is not None and abs(res1["global_change_ratio"] - 0.18) < 0.05,
    "raw_answer == yes": res1.get("raw_answer") == "yes",
    "has_grounding == True": res1.get("has_grounding") is True,
    "valid change_mask_base64": isinstance(res1.get("change_mask_base64"), str) and len(res1["change_mask_base64"]) > 100,
    "model_name == ChangeFormerV6": res1.get("model_name") == "ChangeFormerV6",
}

pass_count = sum(1 for v in checks.values() if v)
total_count = len(checks)

for chk, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")

print(f"\nResult: {pass_count}/{total_count} checks passed.")
print("=" * 65)
