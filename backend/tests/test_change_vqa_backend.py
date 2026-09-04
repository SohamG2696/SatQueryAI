"""
Integration test for backend/app/models/change_vqa.py.
"""
import sys
import time
import base64
import io
from pathlib import Path
from PIL import Image as PIL_Image

SATQUERY_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = SATQUERY_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SATQUERY_ROOT) not in sys.path:
    sys.path.insert(0, str(SATQUERY_ROOT))

def test_change_vqa_backend_adapter():
    from app.models.change_vqa import run_module

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
    res = run_module([img1, img2], question, metadata={"dates": ["2022-01-01", "2025-01-01"]})

    assert res is not None
    assert isinstance(res.get("answer"), str)
    assert len(res.get("answer")) > 0
    assert res.get("model_name") == "satquery-change-vqa-v1" or res.get("model_name") == "ChangeFormerV6"

if __name__ == "__main__":
    test_change_vqa_backend_adapter()
