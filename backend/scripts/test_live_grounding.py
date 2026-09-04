"""
SatQuery AI — Live Grounding API Verification Script.

Tests POST /api/grounding on FastAPI TestClient with:
1. "Locate the roads in the image."
2. "Locate the buildings in the image."
3. "Locate the water tank in the image."
4. "Locate vegetation in the image."
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


def make_test_image() -> bytes:
    img = PIL_Image.new("RGB", (224, 224), color=(80, 150, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_live_grounding_test():
    print("=" * 80)
    print("           SATQUERY AI — LIVE GROUNDING API VERIFICATION")
    print("=" * 80)

    # 1. Upload Test Image
    img_bytes = make_test_image()
    resp_up = client.post(
        "/api/upload",
        files={"file": ("sentinel_test.png", io.BytesIO(img_bytes), "image/png")},
        data={"modality": "optical"},
    )
    assert resp_up.status_code == 200, f"Upload failed: {resp_up.text}"
    img_id = resp_up.json()["image_id"]
    print(f"Uploaded Test Image ID: {img_id}")

    queries = [
        "Locate the roads in the image.",
        "Locate the buildings in the image.",
        "Locate the water tank in the image.",
        "Locate vegetation in the image.",
    ]

    results = {}
    boxes = {}

    for q in queries:
        resp = client.post(
            "/api/grounding",
            json={"image_id": img_id, "query": q},
        )
        assert resp.status_code == 200, f"Grounding API failed for query '{q}': {resp.text}"
        data = resp.json()

        bbox = data["evidence"][0] if data["evidence"] else None
        conf = data["confidence"]
        norm_q = data["parameters"].get("normalized_query")

        results[q] = data
        boxes[q] = bbox

        print("-" * 60)
        print(f"Query           : '{q}'")
        print(f"  Normalized Q  : '{norm_q}'")
        print(f"  BBox Evidence : {bbox}")
        print(f"  Confidence    : {conf}")

    print("\n" + "=" * 80)
    print("VERIFYING BBOX DIFFERENTIATION ACROSS API QUERIES")
    print("=" * 80)

    unique_boxes = set(boxes.values())
    print(f"Total Unique BBoxes returned: {len(unique_boxes)} out of {len(queries)}")

    for q1, b1 in boxes.items():
        print(f"  '{q1}' -> {b1}")

    assert len(unique_boxes) > 1, f"API returned identical bounding boxes for all queries: {unique_boxes}"

    print("\n" + "=" * 80)
    print("ALL LIVE GROUNDING API CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_live_grounding_test()
