"""
SatQuery AI — Result Interpreter Unit Tests.

Tests interpretation output for VLM, Captioning, Grounding,
ChangeFormer, Fusion, and GEE results using realistic payloads.
"""

from __future__ import annotations

import pytest

from app.services.result_interpreter import interpret_result, Interpretation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VLM / VQA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_vqa():
    result = {
        "answer": "Yes, buildings are visible in the image.",
        "confidence": 0.87,
        "model_name": "satquery-vlm-person-a",
    }
    interp = interpret_result(
        query="Are buildings visible in this image?",
        task="vqa",
        result=result,
    )
    assert isinstance(interp, Interpretation)
    assert "buildings" in interp.answer.lower()
    assert "satquery-vlm-person-a" in str(interp.key_findings)
    assert interp.raw_result is result
    assert "high" in interp.confidence_explanation.lower()
    assert interp.technical_interpretation


def test_interpret_vqa_low_confidence():
    result = {"answer": "Maybe", "confidence": 0.15}
    interp = interpret_result(query="Is there water?", task="vqa", result=result)
    assert "very low" in interp.confidence_explanation.lower()
    assert "caution" in interp.confidence_explanation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Captioning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_captioning():
    result = {
        "answer": "An aerial view of a coastal urban area with a harbor and green vegetation.",
        "model_name": "satquery-vlm-person-a",
    }
    interp = interpret_result(
        query="Describe this image.",
        task="captioning",
        result=result,
    )
    assert "coastal" in interp.answer.lower()
    assert "caption" in interp.technical_interpretation.lower()
    assert any("caption" in f.lower() for f in interp.key_findings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Grounding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_grounding_with_bbox():
    result = {
        "answer": "Region detected",
        "confidence": 0.42,
        "normalized_query": "water bodies",
        "visual_evidence": {
            "type": "bbox",
            "coordinates": [0.1, 0.2, 0.5, 0.6],
        },
    }
    interp = interpret_result(
        query="Locate the water bodies in the image.",
        task="grounding",
        result=result,
    )
    assert "water bodies" in str(interp.key_findings).lower()
    assert "[0.1, 0.2, 0.5, 0.6]" in str(interp.key_findings)
    assert "bounding box" in interp.technical_interpretation.lower()
    assert interp.confidence_explanation


def test_interpret_grounding_very_low_confidence():
    result = {
        "answer": "Region detected",
        "confidence": 0.003,
        "normalized_query": "roads",
        "visual_evidence": {"type": "bbox", "coordinates": [0.0, 0.0, 1.0, 1.0]},
    }
    interp = interpret_result(
        query="Locate roads",
        task="grounding",
        result=result,
    )
    assert interp.limitations
    assert "unreliable" in interp.limitations.lower() or "confidence" in interp.limitations.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChangeFormer / Change VQA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_change_detection():
    result = {
        "answer": "YES — significant urban expansion detected between the two dates.",
        "confidence": 0.78,
        "model_name": "ChangeFormerV6",
        "change_ratio": 0.1823,
        "visual_evidence": {"type": "change_map"},
    }
    interp = interpret_result(
        query="Has urban area increased?",
        task="change_vqa",
        result=result,
        evidence=result.get("visual_evidence"),
    )
    assert "urban" in interp.answer.lower() or "YES" in interp.answer
    assert "ChangeFormerV6" in str(interp.key_findings)
    assert "0.1823" in str(interp.key_findings)
    assert "changeformer analyzed" in interp.summary.lower()
    assert "bi-temporal" in interp.technical_interpretation.lower()


def test_interpret_change_no_evidence():
    result = {
        "answer": "NO",
        "confidence": 0.65,
        "model_name": "ChangeFormerV6",
    }
    interp = interpret_result(query="Any change?", task="change_vqa", result=result)
    assert "NO" in interp.answer
    assert "visualization" not in interp.summary.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optical-SAR Fusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_fusion_mcq():
    """The exact example from the requirements."""
    result = {
        "answer": "D (urban)",
        "confidence": 0.2645,
        "task_sub_type": "mcq",
        "probabilities": {
            "agriculture": 0.256,
            "forest": 0.244,
            "water": 0.2355,
            "urban": 0.2645,
        },
    }
    interp = interpret_result(
        query="Which category best describes the image?",
        task="fusion",
        result=result,
    )
    assert "urban" in interp.summary.lower()
    assert "26" in interp.summary  # confidence percentage
    assert any("urban" in f.lower() for f in interp.key_findings)
    assert "Optical-SAR" in str(interp.key_findings)
    assert "fusion model" in interp.technical_interpretation.lower()
    assert interp.limitations  # low confidence triggers limitation
    assert "uncertain" in interp.limitations.lower()


def test_interpret_fusion_binary():
    result = {
        "answer": "YES",
        "confidence": 0.82,
        "task_sub_type": "binary",
    }
    interp = interpret_result(
        query="Is there agricultural land?",
        task="fusion",
        result=result,
    )
    assert "YES" in interp.answer
    assert "cross-modal" in interp.technical_interpretation.lower()
    assert interp.limitations == ""  # high confidence, no limitation


def test_interpret_fusion_bbox():
    result = {
        "answer": "Region detected",
        "confidence": 0.55,
        "task_sub_type": "bbox",
        "visual_evidence": {
            "type": "bbox",
            "coordinates": [0.2, 0.3, 0.7, 0.8],
        },
    }
    interp = interpret_result(
        query="Where is the urban area?",
        task="fusion",
        result=result,
        evidence=result.get("visual_evidence"),
    )
    assert "[0.2, 0.3, 0.7, 0.8]" in str(interp.key_findings)
    assert "fusion model" in interp.technical_interpretation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Elevation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_elevation_point():
    result = {"elevation": 1500.5, "buffer_m": 0.0}
    interp = interpret_result(
        query="What is the elevation?",
        task="elevation",
        result=result,
    )
    assert "1500.50" in interp.answer
    assert "SRTM" in str(interp.key_findings)
    assert "Shuttle Radar" in interp.technical_interpretation


def test_interpret_elevation_buffer():
    result = {
        "elevation_mean": 1200.0,
        "elevation_min": 1100.0,
        "elevation_max": 1300.0,
        "buffer_m": 1000.0,
    }
    interp = interpret_result(
        query="What is the elevation around this point?",
        task="elevation",
        result=result,
    )
    assert "1200.00" in interp.answer
    assert "1100.00" in interp.answer
    assert "1300.00" in interp.answer
    assert "1000" in interp.answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Spectral Indices
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_ndvi():
    result = {
        "index_value": 0.72,
        "date": "2024-07-15",
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
    }
    interp = interpret_result(query="What is the NDVI?", task="ndvi", result=result)
    assert "0.7200" in interp.answer
    assert "vegetation" in interp.answer.lower() or "strong" in interp.answer.lower()
    assert "2024-07-15" in interp.answer
    assert "NIR" in interp.technical_interpretation


def test_interpret_ndwi():
    result = {
        "index_value": -0.10,
        "date": "2024-07-10",
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
    }
    interp = interpret_result(query="What is the NDWI?", task="ndwi", result=result)
    assert "-0.1000" in interp.answer
    assert "low" in interp.answer.lower()
    assert "Green" in interp.technical_interpretation


def test_interpret_ndbi():
    result = {
        "index_value": 0.11,
        "date": "2024-07-20",
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
    }
    interp = interpret_result(query="What is the NDBI?", task="ndbi", result=result)
    assert "0.1100" in interp.answer
    assert "built-up" in interp.answer.lower()
    assert "SWIR" in interp.technical_interpretation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE — Imagery Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_sentinel2_imagery():
    result = {
        "image_id": "S2A_MSIL2A_20240615",
        "date": "2024-06-15",
        "cloud_percentage": 8.5,
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
        "thumbnail_url": "http://example.com/thumb.png",
    }
    interp = interpret_result(
        query="Get Sentinel-2 imagery", task="sentinel2", result=result
    )
    assert "Sentinel-2" in interp.summary
    assert "2024-06-15" in interp.summary
    assert "8.5%" in interp.summary


def test_interpret_sentinel1_imagery():
    result = {
        "image_id": "S1A_IW_20240210",
        "date": "2024-02-10",
        "cloud_percentage": None,
        "dataset": "COPERNICUS/S1_GRD",
        "polarization": "VH",
        "orbit_pass": "ASCENDING",
        "instrument_mode": "IW",
    }
    interp = interpret_result(
        query="Get SAR imagery", task="sentinel1", result=result
    )
    assert "Sentinel-1" in interp.summary
    assert any("VH" in f for f in interp.key_findings)
    assert any("ASCENDING" in f for f in interp.key_findings)
    assert "polarization" in interp.technical_interpretation.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEE generic router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_gee_generic_routes_elevation():
    result = {
        "intent": "elevation",
        "result": {"elevation": 800.0, "buffer_m": 0.0},
        "explanation": "...",
    }
    interp = interpret_result(query="elevation?", task="gee", result=result)
    assert "800.00" in interp.answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Edge Cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_interpret_empty_result():
    interp = interpret_result(query="test", task="unknown_task", result={})
    assert isinstance(interp, Interpretation)
    assert interp.raw_result == {}


def test_interpret_missing_fields_no_crash():
    result = {"some_random_key": 42}
    interp = interpret_result(query="test", task="vqa", result=result)
    assert isinstance(interp, Interpretation)
    assert interp.answer is not None


def test_raw_result_always_preserved():
    original = {"answer": "YES", "confidence": 0.95, "custom_field": [1, 2, 3]}
    interp = interpret_result(query="test", task="vqa", result=original)
    assert interp.raw_result is original
    assert interp.raw_result["custom_field"] == [1, 2, 3]
