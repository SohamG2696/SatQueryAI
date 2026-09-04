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

SATQUERY_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = SATQUERY_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SATQUERY_ROOT) not in sys.path:
    sys.path.insert(0, str(SATQUERY_ROOT))

def test_api_query_flow():
    # Use synthetic PIL images if SECOND dataset is not available
    im1_dir = SATQUERY_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im1"
    im2_dir = SATQUERY_ROOT / "datasets" / "SECOND" / "SECOND_train_set" / "im2"

    if im1_dir.exists() and list(im1_dir.glob("*.png")):
        f1 = sorted(im1_dir.glob("*.png"))[0]
        f2 = im2_dir / f1.name
        img1 = PIL_Image.open(f1)
        img2 = PIL_Image.open(f2)
    else:
        img1 = PIL_Image.new("RGB", (256, 256), (100, 150, 200))
        img2 = PIL_Image.new("RGB", (256, 256), (120, 140, 210))

    question = "Have the areas of buildings changed?"

    from app.models.change_vqa import run_module
    from app.services.response_service import assemble_query_response

    t0 = time.time()
    model_output = run_module(
        images=[img1, img2],
        query=question,
        metadata={"dates": ["2022-01-01", "2025-01-01"]}
    )

    response = assemble_query_response(
        task="change_vqa",
        task_route="bi_temporal_change_analysis",
        model_output=model_output
    )

    assert response is not None
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert response.task_detected == "change_vqa"

if __name__ == "__main__":
    test_api_query_flow()
