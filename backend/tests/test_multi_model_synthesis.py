"""
Tests for Multi-Model Result Synthesis and Query Decomposition in SatQuery AI.
"""

import io
import json
import pytest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.services.result_interpreter import decompose_query, synthesize_multi_model_results

client = TestClient(app)


def _create_test_image_bytes(color=(100, 150, 200), size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_1_decompose_query():
    """TEST 1: decompose_query() creates task-specific queries for change_vqa + grounding + fusion."""
    query = (
        "Compare the before and after satellite images, describe the major land-cover changes, "
        "identify newly developed built-up areas, verify those areas using both optical and SAR imagery, "
        "highlight the changed regions, explain any uncertainty, and provide an auditable summary."
    )
    tasks = ["change_vqa", "grounding", "fusion"]

    decomposed = decompose_query(query, tasks)

    assert "change_vqa" in decomposed
    assert "grounding" in decomposed
    assert "fusion" in decomposed

    assert "land-cover changes" in decomposed["change_vqa"].lower() or "changes" in decomposed["change_vqa"].lower()
    assert "built-up" in decomposed["grounding"].lower() or "locate" in decomposed["grounding"].lower()
    assert "optical and sar" in decomposed["fusion"].lower() or "verify" in decomposed["fusion"].lower()

    # Verify task queries are distinct and tailored, not raw unedited prompt
    assert decomposed["change_vqa"] != query
    assert decomposed["grounding"] != query
    assert decomposed["fusion"] != query


def test_2_synthesize_multi_model_results():
    """TEST 2: synthesize_multi_model_results() combines compatible specialist outputs into one coherent answer."""
    sub_results = [
        {
            "task": "change_vqa",
            "model_name": "satquery-change-vqa-v1",
            "answer": "Significant land cover change detected across 3.50% of the area.",
            "confidence": 0.8200,
        },
        {
            "task": "grounding",
            "model_name": "satquery-region-grounding-v1",
            "answer": "Localized target spatial region.",
            "confidence": 0.7800,
            "visual_evidence": {"type": "bbox", "coordinates": [0.1, 0.2, 0.4, 0.5]},
        },
        {
            "task": "fusion",
            "model_name": "satquery-optical-sar-fusion-v1",
            "answer": "YES",
            "confidence": 0.8500,
        },
    ]

    synth = synthesize_multi_model_results(
        original_query="Compare before and after, locate built-up areas, verify using optical and SAR.",
        sub_task_results=sub_results,
    )

    ans = synth["synthesized_answer"]
    assert not ans.startswith("[change_vqa]")
    assert "3.50%" in ans or "3.5%" in ans
    assert "confidence" in ans.lower()
    assert synth["synthesis"]["evidence_quality"] in ("strong", "high", "moderate")


def test_3_top_level_confidence_null():
    """TEST 3: Top-level confidence is None for multi-model queries."""
    img1_bytes = _create_test_image_bytes((120, 200, 100))
    img2_bytes = _create_test_image_bytes((80, 80, 80))

    meta_json = json.dumps({"modalities": ["optical", "sar"]})
    complex_query = (
        "Compare the before and after satellite images, locate the newly developed built-up areas, "
        "and verify those areas using both optical and SAR imagery."
    )

    response = client.post(
        "/api/query",
        files=[
            ("images", ("opt_t1.png", io.BytesIO(img1_bytes), "image/png")),
            ("images", ("sar_t2.png", io.BytesIO(img2_bytes), "image/png")),
        ],
        data={"query": complex_query, "metadata": meta_json},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["task_detected"] == "multi_model"
    assert data["confidence"] is None


def test_4_confidence_by_task_preserves_every_confidence():
    """TEST 4: confidence_by_task preserves every specialist confidence."""
    sub_results = [
        {"task": "change_vqa", "model_name": "m1", "answer": "Change ratio 2.77%", "confidence": 0.7982},
        {"task": "grounding", "model_name": "m2", "answer": "Box", "confidence": 0.0058},
        {"task": "fusion", "model_name": "m3", "answer": "NO", "confidence": 0.0068},
    ]

    synth = synthesize_multi_model_results("query", sub_results)

    conf_map = synth["confidence_by_task"]
    assert conf_map["change_vqa"] == 0.7982
    assert conf_map["grounding"] == 0.0058
    assert conf_map["fusion"] == 0.0068


def test_5_low_confidence_prevents_claiming_reliable_development():
    """TEST 5: Low-confidence grounding/fusion prevents synthesizer from claiming reliable built-up development."""
    sub_results = [
        {"task": "change_vqa", "model_name": "m1", "answer": "Significant change detected across 2.77% of region.", "confidence": 0.7982},
        {"task": "grounding", "model_name": "m2", "answer": "Box", "confidence": 0.0058},
        {"task": "fusion", "model_name": "m3", "answer": "NO", "confidence": 0.0068},
    ]

    synth = synthesize_multi_model_results("query", sub_results)
    ans = synth["synthesized_answer"]

    assert "cannot reliably confirm" in ans.lower() or "insufficient" in ans.lower() or "very low" in ans.lower()
    assert "new buildings were detected" not in ans.lower()


def test_6_conflicting_specialist_outputs():
    """TEST 6: Conflicting specialist outputs are explicitly represented as uncertainty/conflict."""
    sub_results = [
        {"task": "change_vqa", "model_name": "m1", "answer": "Significant change detected across 5.0% of area.", "confidence": 0.8500},
        {"task": "fusion", "model_name": "m3", "answer": "NO", "confidence": 0.8800},
    ]

    synth = synthesize_multi_model_results("query", sub_results)
    ans = synth["synthesized_answer"]

    assert synth["synthesis"]["evidence_quality"] == "conflicting" or "conflict" in ans.lower()
    assert len(synth["synthesis"]["uncertainties"]) >= 1


def test_7_missing_evidence_no_fabricated_spatial_claims():
    """TEST 7: Missing evidence does not cause fabricated spatial claims."""
    sub_results = [
        {"task": "change_vqa", "model_name": "m1", "answer": "Change detected.", "confidence": 0.6000},
    ]

    synth = synthesize_multi_model_results("query", sub_results)

    assert synth["primary_visual_evidence"].type == "none"
    assert "coordinates" not in synth["synthesized_answer"].lower() or synth["primary_visual_evidence"].coordinates is None


def test_8_synthesis_failure_fallback():
    """TEST 8: Synthesis failure falls back safely to existing specialist answers."""
    img1_bytes = _create_test_image_bytes((120, 200, 100))
    img2_bytes = _create_test_image_bytes((80, 80, 80))
    meta_json = json.dumps({"modalities": ["optical", "sar"]})

    with patch("app.agent.controller.synthesize_multi_model_results", side_effect=RuntimeError("Synthesis error")):
        response = client.post(
            "/api/query",
            files=[
                ("images", ("opt_t1.png", io.BytesIO(img1_bytes), "image/png")),
                ("images", ("sar_t2.png", io.BytesIO(img2_bytes), "image/png")),
            ],
            data={
                "query": "Compare before and after satellite images, locate built-up areas, verify using optical and SAR.",
                "metadata": meta_json,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_detected"] == "multi_model"
        assert "fallback" in " ".join(data["execution_summary"]["parameters"]["execution_trace"]).lower()
        assert data["answer"] != ""


def test_9_single_model_regression():
    """TEST 9: Existing single-model behavior still passes."""
    opt_bytes = _create_test_image_bytes()

    # 1. Captioning
    resp_cap = client.post(
        "/api/query",
        files=[("images", ("opt.png", io.BytesIO(opt_bytes), "image/png"))],
        data={"query": "Describe this satellite image."},
    )
    assert resp_cap.status_code == 200
    data_cap = resp_cap.json()
    assert data_cap["task_detected"] == "captioning"
    assert data_cap["confidence_by_task"] is None

    resp_grd = client.post(
        "/api/query",
        files=[("images", ("opt.png", io.BytesIO(opt_bytes), "image/png"))],
        data={"query": "Locate the lake in this image."},
    )
    assert resp_grd.status_code == 200
    data_grd = resp_grd.json()
    assert data_grd["task_detected"] == "grounding"
    assert data_grd["visual_evidence"]["type"] in ("bbox", "mask")
    assert data_grd["confidence_by_task"] is None
