"""
SatQuery AI — Query Routing & Controller Verification Script.

Tests POST /api/query for the 4 target query types:
1. Grounding: "Find the buildings in this image."
2. Change: "Did anything change between these two images?"
3. Fusion: "Compare the optical and SAR images and tell me whether the region contains buildings."
4. Multi-Model: "Find buildings in the image and tell me whether they changed between the two dates."
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_test_img(color=(100, 150, 200), size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_query_routing_test():
    print("=" * 80)
    print("       SATQUERY AI — /api/query & AGENTIC CONTROLLER VERIFICATION")
    print("=" * 80)

    img1 = make_test_img((120, 180, 100))
    img2 = make_test_img((150, 90, 60))

    # ── QUERY 1: GROUNDING ───────────────────────────────────────────────────
    q1 = "Find the buildings in this image."
    print(f"\n[TEST 1] Query: '{q1}'")
    resp1 = client.post(
        "/api/query",
        files=[("images", ("scene.png", io.BytesIO(img1), "image/png"))],
        data={"query": q1},
    )
    assert resp1.status_code == 200, f"Q1 failed: {resp1.text}"
    d1 = resp1.json()
    print(f"  task_detected    : {d1['task_detected']}")
    print(f"  answer           : {d1['answer']}")
    print(f"  confidence       : {d1['confidence']}")
    print(f"  models_used      : {d1['execution_summary']['models_used']}")
    print(f"  visual_evidence  : {d1['visual_evidence']['type']} (bbox={d1['visual_evidence'].get('coordinates')})")
    assert d1["task_detected"] == "grounding"
    assert "satquery-region-grounding-v1" in d1["execution_summary"]["models_used"]
    print("  => SUCCESS: Grounding model correctly selected and executed.")

    # ── QUERY 2: CHANGE DETECTION ────────────────────────────────────────────
    q2 = "Did anything change between these two images?"
    print(f"\n[TEST 2] Query: '{q2}'")
    resp2 = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2), "image/png")),
        ],
        data={
            "query": q2,
            "metadata": json.dumps({"dates": ["2024-01-01", "2026-01-01"]}),
        },
    )
    assert resp2.status_code == 200, f"Q2 failed: {resp2.text}"
    d2 = resp2.json()
    print(f"  task_detected    : {d2['task_detected']}")
    print(f"  answer           : {d2['answer']}")
    print(f"  confidence       : {d2['confidence']}")
    print(f"  models_used      : {d2['execution_summary']['models_used']}")
    print(f"  trace            : {d2['execution_summary']['parameters'].get('execution_trace')}")
    assert d2["task_detected"] == "change_vqa"
    assert "ChangeFormerV6" in d2["execution_summary"]["models_used"]
    print("  => SUCCESS: ChangeFormerV6 correctly selected and executed.")

    # ── QUERY 3: OPTICAL-SAR FUSION ──────────────────────────────────────────
    q3 = "Compare the optical and SAR images and tell me whether the region contains buildings."
    print(f"\n[TEST 3] Query: '{q3}'")
    resp3 = client.post(
        "/api/query",
        files=[
            ("images", ("optical.png", io.BytesIO(img1), "image/png")),
            ("images", ("sar.png", io.BytesIO(img2), "image/png")),
        ],
        data={
            "query": q3,
            "metadata": json.dumps({"modalities": ["optical", "sar"]}),
        },
    )
    assert resp3.status_code == 200, f"Q3 failed: {resp3.text}"
    d3 = resp3.json()
    print(f"  task_detected    : {d3['task_detected']}")
    print(f"  answer           : {d3['answer']}")
    print(f"  confidence       : {d3['confidence']}")
    print(f"  models_used      : {d3['execution_summary']['models_used']}")
    assert d3["task_detected"] == "fusion"
    assert "satquery-optical-sar-fusion-v1" in d3["execution_summary"]["models_used"]
    print("  => SUCCESS: Fusion model correctly selected and executed.")

    # ── QUERY 4: MULTI-MODEL (GROUNDING + CHANGE) ─────────────────────────────
    q4 = "Find buildings in the image and tell me whether they changed between the two dates."
    print(f"\n[TEST 4] Query: '{q4}'")
    resp4 = client.post(
        "/api/query",
        files=[
            ("images", ("t1.png", io.BytesIO(img1), "image/png")),
            ("images", ("t2.png", io.BytesIO(img2), "image/png")),
        ],
        data={
            "query": q4,
            "metadata": json.dumps({"dates": ["2024-01-01", "2026-01-01"]}),
        },
    )
    assert resp4.status_code == 200, f"Q4 failed: {resp4.text}"
    d4 = resp4.json()
    print(f"  task_detected    : {d4['task_detected']}")
    print(f"  answer           : {d4['answer']}")
    print(f"  confidence       : {d4['confidence']}")
    print(f"  models_used      : {d4['execution_summary']['models_used']}")
    print(f"  visual_evidence  : {d4['visual_evidence']['type']}")
    print(f"  trace            : {d4['execution_summary']['parameters'].get('execution_trace')}")
    assert d4["task_detected"] == "multi_model"
    assert len(d4["execution_summary"]["models_used"]) >= 2
    assert "satquery-region-grounding-v1" in d4["execution_summary"]["models_used"]
    assert "ChangeFormerV6" in d4["execution_summary"]["models_used"]
    print("  => SUCCESS: Multi-model workflow executed Grounding + ChangeFormer in sequence and combined outputs.")

    # ── TEST ERROR HANDLING (MISSING REQUIRED IMAGE) ─────────────────────────
    print("\n[TEST 5] Error Handling: Change query with only 1 image...")
    err_resp = client.post(
        "/api/query",
        files=[("images", ("t1.png", io.BytesIO(img1), "image/png"))],
        data={"query": "Did anything change between these two images?"},
    )
    print(f"  status_code      : {err_resp.status_code}  (expected 400)")
    print(f"  detail           : {err_resp.json().get('detail')}")
    assert err_resp.status_code == 400
    assert "at least 2 images" in err_resp.json()["detail"].lower()
    print("  => SUCCESS: Explicit 400 error returned for missing required images.")

    print("\n" + "=" * 80)
    print(" ALL 4 TARGET QUERIES AND ERROR HANDLING VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_query_routing_test()
