"""
API-Level Query Flow Integration Test for Change VQA.

Tests end-to-end processing from Controller -> Route -> Model Registry ->
Inference Service -> Change VQA Adapter -> Response Assembler -> QueryResponse schema.
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

# Target SECOND image pair
SECOND_IM1 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im1" / "10589.png"
SECOND_IM2 = SIH2026_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im2" / "10589.png"

print("=" * 65)
print("API-LEVEL QUERY FLOW INTEGRATION TEST")
print("=" * 65)

# Load PIL images as direct uploads simulate
img1 = PIL_Image.open(SECOND_IM1)
img2 = PIL_Image.open(SECOND_IM2)
question = "Have the areas of buildings changed?"

print(f"Testing direct PIL.Image inputs for query: '{question}'...")

# Load backend/app/models/change_vqa.py directly
vqa_adapter_path = BACKEND_ROOT / "app" / "models" / "change_vqa.py"
spec = importlib.util.spec_from_file_location("backend_change_vqa_adapter", str(vqa_adapter_path))
change_vqa_mod = importlib.util.module_from_spec(spec)
sys.modules["backend_change_vqa_adapter"] = change_vqa_mod
spec.loader.exec_module(change_vqa_mod)

t0 = time.time()
model_output = change_vqa_mod.run_module(
    images=[img1, img2],
    query=question,
    metadata={"dates": ["2022-01-01", "2025-01-01"]}
)
elapsed = (time.time() - t0) * 1000

print(f"\nCompleted in {elapsed:.1f} ms")
print("-" * 65)

# Assemble using response_service if pydantic is installed, else test model_output directly
response = None
try:
    from app.services.response_service import assemble_query_response
    response = assemble_query_response(
        task="change_vqa",
        task_route="bi_temporal_change_analysis",
        model_output=model_output
    )
    print(f"QueryResponse assembled via response_service successfully.")
except ModuleNotFoundError as e:
    print(f"Note: response_service assembly skipped ({e}). Testing model_output dict fields directly.")

if response is not None:
    answer = response.answer
    confidence = response.confidence
    ve = response.visual_evidence
    ve_type = ve.type if ve else None
    mask_b64 = ve.data.get("mask_base64") if (ve and isinstance(ve.data, dict)) else None
    params = response.execution_summary.parameters
else:
    answer = model_output.get("answer")
    confidence = model_output.get("confidence")
    ve_dict = model_output.get("visual_evidence", {})
    ve_type = ve_dict.get("type")
    mask_b64 = ve_dict.get("data", {}).get("mask_base64")
    params = model_output.get("parameters", {})

print(f"answer                             : {answer}")
print(f"confidence                         : {confidence}")
print(f"visual_evidence.type               : {ve_type}")
print(f"visual_evidence.data.mask_base64   : <{len(mask_b64) if mask_b64 else 0} chars>")
print(f"params.category                    : {params.get('category')}")
print(f"params.category_change_ratio       : {params.get('category_change_ratio')}")
print(f"params.global_change_ratio         : {params.get('global_change_ratio')}")
print(f"params.question_type               : {params.get('question_type')}")
print(f"params.has_grounding               : {params.get('has_grounding')}")
print(f"params.raw_answer                  : {params.get('raw_answer')}")

print("\n" + "=" * 65)
print("VERIFICATION CHECKS FOR QUERYRESPONSE SCHEMA")
print("=" * 65)

checks = {
    "answer is non-empty string": isinstance(answer, str) and len(answer) > 0,
    "confidence is valid float": isinstance(confidence, float) and 0.0 <= confidence <= 1.0,
    "visual_evidence.data.mask_base64 is present": isinstance(mask_b64, str) and len(mask_b64) > 100,
    "execution_summary.parameters.category == buildings": params.get("category") == "buildings",
    "execution_summary.parameters.category_change_ratio ~ 0.6337": params.get("category_change_ratio") is not None and abs(params["category_change_ratio"] - 0.6337) < 0.05,
    "execution_summary.parameters.global_change_ratio ~ 0.18": params.get("global_change_ratio") is not None and abs(params["global_change_ratio"] - 0.18) < 0.05,
    "execution_summary.parameters.question_type == change_or_not": params.get("question_type") == "change_or_not",
    "execution_summary.parameters.has_grounding == True": params.get("has_grounding") is True,
    "execution_summary.parameters.raw_answer == yes": params.get("raw_answer") == "yes",
}

pass_count = sum(1 for v in checks.values() if v)
total_count = len(checks)

for chk, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")

print(f"\nResult: {pass_count}/{total_count} checks passed.")
print("=" * 65)
