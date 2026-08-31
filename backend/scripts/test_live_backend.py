"""
SatQuery AI — Live Backend & Model Inference Verification Script.

Tests the FastAPI application end-to-end against all 5 multi-modal scenarios:
1. Optical-SAR Cross-Modal Fusion (Binary, MCQ, BBox)
2. Single-Image Region Grounding
3. Bi-Temporal Change-VQA
4. Single-Image VQA (Placeholder verification)
5. Single-Image Captioning (Placeholder verification)
6. Upload & Metadata extraction
7. Session History tracking
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Ensure sys.path includes SatQuery-AI and backend
_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.main import app
from app.utils.device import get_device

client = TestClient(app)


def make_test_image(color=(120, 180, 80), size=(224, 224)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_live_tests():
    print("=" * 80)
    print("           SATQUERY AI (SIH26167) — LIVE BACKEND VERIFICATION")
    print("=" * 80)

    device = get_device()
    print(f"[Device] Active PyTorch compute device: {device}")

    # ── 1. Test /health ───────────────────────────────────────────
    print("\n[1] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print("    Status:", health_data["status"])
    print("    Service:", health_data["service"], f"v{health_data['version']}")
    print("    Models status:", json.dumps(health_data["models"], indent=2))
    assert health_data["models"]["fusion"] == "ready"
    assert health_data["models"]["grounding"] == "ready"
    assert health_data["models"]["change_vqa"] == "ready"

    # ── 2. Test Image Upload ─────────────────────────────────────
    print("\n[2] Testing POST /api/upload (Optical & SAR)...")
    opt_bytes = make_test_image((34, 139, 34))
    sar_bytes = make_test_image((70, 70, 70))

    resp_opt = client.post(
        "/api/upload",
        files={"file": ("sentinel2_optical.png", io.BytesIO(opt_bytes), "image/png")},
        data={"modality": "optical"},
    )
    assert resp_opt.status_code == 200, f"Upload optical failed: {resp_opt.text}"
    opt_data = resp_opt.json()
    opt_id = opt_data["image_id"]
    print(f"    Uploaded Optical Image ID: {opt_id}")
    print(f"    Detected Metadata: {opt_data['metadata_detected']}")

    resp_sar = client.post(
        "/api/upload",
        files={"file": ("sentinel1_sar.png", io.BytesIO(sar_bytes), "image/png")},
        data={"modality": "sar"},
    )
    assert resp_sar.status_code == 200, f"Upload SAR failed: {resp_sar.text}"
    sar_data = resp_sar.json()
    sar_id = sar_data["image_id"]
    print(f"    Uploaded SAR Image ID: {sar_id}")
    print(f"    Detected Metadata: {sar_data['metadata_detected']}")

    session_id = "sih26167_live_session"

    # ── 3. Test Optical-SAR Fusion ───────────────────────────────
    print("\n[3] Testing POST /api/query -> Optical-SAR Fusion...")
    fusion_meta = json.dumps({"modalities": ["optical", "sar"], "session_id": session_id})
    resp_fusion = client.post(
        "/api/query",
        data={
            "image_ids": f"{opt_id},{sar_id}",
            "query": "Does the SAR imagery confirm the land cover pattern in the optical image?",
            "metadata": fusion_meta,
        },
    )
    assert resp_fusion.status_code == 200, f"Fusion query failed: {resp_fusion.text}"
    f_res = resp_fusion.json()
    print("    Task Detected:", f_res["task_detected"])
    print("    Answer:", f_res["answer"])
    print("    Confidence:", f_res["confidence"])
    print("    Route:", f_res["execution_summary"]["task_route"])
    print("    Processing Time:", f"{f_res['execution_summary']['processing_time_ms']} ms")

    # ── 4. Test Region Grounding ─────────────────────────────────
    print("\n[4] Testing POST /api/query -> Region Grounding...")
    resp_ground = client.post(
        "/api/query",
        data={
            "image_ids": opt_id,
            "query": "Locate the smallest contiguous forest region in this image.",
            "metadata": json.dumps({"session_id": session_id}),
        },
    )
    assert resp_ground.status_code == 200, f"Grounding query failed: {resp_ground.text}"
    g_res = resp_ground.json()
    print("    Task Detected:", g_res["task_detected"])
    print("    Answer:", g_res["answer"])
    print("    Visual Evidence:", g_res["visual_evidence"])
    print("    Route:", g_res["execution_summary"]["task_route"])
    print("    Processing Time:", f"{g_res['execution_summary']['processing_time_ms']} ms")

    # ── 5. Test Change-VQA ───────────────────────────────────────
    print("\n[5] Testing POST /api/query -> Bi-Temporal Change Analysis...")
    change_meta = json.dumps({
        "dates": ["2022-06-01", "2025-06-01"],
        "session_id": session_id,
    })
    resp_change = client.post(
        "/api/query",
        data={
            "image_ids": f"{opt_id},{opt_id}",
            "query": "Did vegetation cover increase between 2022 and 2025?",
            "metadata": change_meta,
        },
    )
    assert resp_change.status_code == 200, f"Change query failed: {resp_change.text}"
    c_res = resp_change.json()
    print("    Task Detected:", c_res["task_detected"])
    print("    Answer:", c_res["answer"])
    print("    Confidence:", c_res["confidence"])
    print("    Route:", c_res["execution_summary"]["task_route"])
    print("    Processing Time:", f"{c_res['execution_summary']['processing_time_ms']} ms")

    # ── 6. Test VQA Placeholder ──────────────────────────────────
    print("\n[6] Testing POST /api/query -> VQA (VLM placeholder)...")
    resp_vqa = client.post(
        "/api/query",
        data={
            "image_ids": opt_id,
            "query": "What types of crop fields are present?",
        },
    )
    assert resp_vqa.status_code == 200, f"VQA query failed: {resp_vqa.text}"
    v_res = resp_vqa.json()
    print("    Task Detected:", v_res["task_detected"])
    print("    Answer:", v_res["answer"])
    print("    Route:", v_res["execution_summary"]["task_route"])

    # ── 7. Test Captioning Placeholder ───────────────────────────
    print("\n[7] Testing POST /api/query -> Captioning (VLM placeholder)...")
    resp_cap = client.post(
        "/api/query",
        data={
            "image_ids": opt_id,
            "query": "Describe this satellite scene.",
        },
    )
    assert resp_cap.status_code == 200, f"Caption query failed: {resp_cap.text}"
    cap_res = resp_cap.json()
    print("    Task Detected:", cap_res["task_detected"])
    print("    Answer:", cap_res["answer"])

    # ── 8. Test Session History ──────────────────────────────────
    print("\n[8] Testing GET /api/history/{session_id}...")
    resp_hist = client.get(f"/api/history/{session_id}")
    assert resp_hist.status_code == 200, f"History query failed: {resp_hist.text}"
    hist_records = resp_hist.json()
    print(f"    Total recorded queries for session '{session_id}': {len(hist_records)}")
    for idx, r in enumerate(hist_records, start=1):
        print(f"      [{idx}] {r['timestamp']} | Task: {r['task_detected']} | Query: '{r['query']}' -> Answer: {r['answer']}")

    print("\n" + "=" * 80)
    print("           ALL 8 LIVE BACKEND VERIFICATION CHECKS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_live_tests()
