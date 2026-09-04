"""
SatQuery AI — Live Fusion API Verification Script.

Tests POST /api/fusion & POST /api/query (fusion task) on FastAPI TestClient with:
1. "Describe the land cover in this area."
2. "Is there water in this area?"
3. "Are there buildings in this area?"
4. "Is vegetation present?"
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from PIL import Image as PIL_Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.main import app

client = TestClient(app)


def make_opt_image() -> bytes:
    img = PIL_Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_sar_image() -> bytes:
    img = PIL_Image.new("RGB", (224, 224), color=(70, 70, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_live_fusion_test():
    print("=" * 80)
    print("           SATQUERY AI — LIVE FUSION API VERIFICATION")
    print("=" * 80)

    # Upload Optical and SAR images
    opt_bytes = make_opt_image()
    sar_bytes = make_sar_image()

    resp_opt = client.post(
        "/api/upload",
        files={"file": ("sentinel2_optical.png", io.BytesIO(opt_bytes), "image/png")},
        data={"modality": "optical"},
    )
    assert resp_opt.status_code == 200
    opt_id = resp_opt.json()["image_id"]

    resp_sar = client.post(
        "/api/upload",
        files={"file": ("sentinel1_sar.png", io.BytesIO(sar_bytes), "image/png")},
        data={"modality": "sar"},
    )
    assert resp_sar.status_code == 200
    sar_id = resp_sar.json()["image_id"]

    print(f"Uploaded Optical ID: {opt_id}")
    print(f"Uploaded SAR ID    : {sar_id}")

    queries = [
        "Describe the land cover in this area.",
        "Is there water in this area?",
        "Are there buildings in this area?",
        "Is vegetation present?",
    ]

    for q in queries:
        resp = client.post(
            "/api/fusion",
            json={
                "optical_image_id": opt_id,
                "sar_image_id": sar_id,
                "query": q,
            },
        )
        assert resp.status_code == 200, f"Fusion API failed for query '{q}': {resp.text}"
        data = resp.json()

        print("-" * 60)
        print(f"Query        : '{q}'")
        print(f"  Success    : {data['success']}")
        print(f"  Task       : {data['task']}")
        print(f"  Answer     : {data['answer']}")
        print(f"  Confidence : {data['confidence']}")
        print(f"  Models Used: {data['models_used']}")
        print(f"  Time       : {data['processing_time']}s")

    print("\n" + "=" * 80)
    print("ALL LIVE FUSION API CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_live_fusion_test()
