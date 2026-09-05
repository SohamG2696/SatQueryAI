"""
SatQuery AI — Result Interpreter & Synthesis Quality Tests.

Comprehensive tests verifying:
A. ChangeFormer interpretation (pixels, ratio, confidence, explanation)
B. Grounding interpretation (bbox, confidence, low-conf warnings, no fabrication)
C. Fusion interpretation (binary, MCQ, probability distributions, close-prob uncertainty)
D. VLM/VQA interpretation (answer, confidence, factual explanation)
E. GEE interpretation (NDVI, NDWI, NDBI, elevation, imagery, non-overclaiming)
F. Multi-model synthesis (agreement, conflict, weak evidence, GEE integration)
G. Mathematical derivation (pixel ratios, percentages)
H. Prevention of unsupported speculative claims & confidence != accuracy distinction
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.result_interpreter import (
    interpret_result,
    synthesize_multi_model_results,
    Interpretation,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. ChangeFormer Interpretation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_changeformer_interpretation_quality():
    """Verify ChangeFormer interpreter extracts pixel counts, percentage ratios, confidence, and non-overclaiming text."""
    result = {
        "answer": "Yes, there are changes detected (2.8% of the area).",
        "confidence": 0.7982,
        "change_ratio": 0.0277,
        "model_name": "ChangeFormerV6",
        "visual_evidence": {
            "type": "change_mask",
            "changed_pixels": 1814,
            "total_pixels": 65536,
            "change_ratio": 0.0277,
        },
        "parameters": {
            "mean_change_probability": 0.100922,
        },
    }

    interp = interpret_result(
        query="Did the area change?",
        task="change_vqa",
        result=result,
        evidence=result["visual_evidence"],
    )

    formatted = interp.to_formatted_answer()

    # Derived values & exact numbers
    assert "2.77%" in formatted or "2.77%" in str(interp.key_findings)
    assert "1,814 of 65,536" in formatted or "1,814" in str(interp.key_findings)
    assert "79.82%" in formatted or "79.82%" in str(interp.key_findings)
    assert "ChangeFormerV6" in formatted or "ChangeFormerV6" in str(interp.key_findings)

    # Distinguish confidence from accuracy
    assert "accuracy" not in interp.confidence_explanation.lower() or "not equivalent to ground-truth accuracy" in interp.confidence_explanation.lower()
    assert "2.77% of the area is newly constructed" not in formatted.lower()


def test_changeformer_building_query_context():
    """Verify ChangeFormer interpreter adds target nuance when asked about buildings specifically."""
    result = {
        "answer": "Yes, changes detected.",
        "confidence": 0.85,
        "change_ratio": 0.05,
        "model_name": "ChangeFormerV6",
        "visual_evidence": {"type": "change_mask", "changed_pixels": 3276, "total_pixels": 65536},
    }

    interp = interpret_result(
        query="Did new buildings appear?",
        task="change_vqa",
        result=result,
        evidence=result["visual_evidence"],
    )

    formatted = interp.to_formatted_answer()
    assert "building" in formatted.lower() or "cannot single-handedly confirm" in formatted.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Grounding Interpretation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_grounding_high_confidence_bbox():
    """Verify Grounding interpreter correctly includes normalized bounding box and confidence."""
    result = {
        "answer": "Localized target spatial region.",
        "confidence": 0.88,
        "normalized_query": "buildings",
        "model_name": "satquery-region-grounding-v1",
        "visual_evidence": {
            "type": "bbox",
            "coordinates": [0.12, 0.25, 0.45, 0.60],
        },
    }

    interp = interpret_result(
        query="Locate the buildings",
        task="grounding",
        result=result,
        evidence=result["visual_evidence"],
    )

    formatted = interp.to_formatted_answer()
    assert "[0.12, 0.25, 0.45, 0.6]" in formatted or "[0.12, 0.25, 0.45, 0.60]" in str(interp.key_findings)
    assert "88.00%" in formatted or "88.00%" in str(interp.key_findings)
    assert interp.limitations == ""


def test_grounding_low_confidence_warning():
    """Verify Grounding interpreter flags low confidence bounding box as unreliable."""
    result = {
        "answer": "Localized target spatial region.",
        "confidence": 0.1141,
        "normalized_query": "buildings",
        "model_name": "satquery-region-grounding-v1",
        "visual_evidence": {
            "type": "bbox",
            "coordinates": [0.1, 0.2, 0.3, 0.4],
        },
    }

    interp = interpret_result(
        query="Locate the buildings",
        task="grounding",
        result=result,
        evidence=result["visual_evidence"],
    )

    formatted = interp.to_formatted_answer()
    assert "unreliable" in formatted.lower() or "low confidence" in formatted.lower()
    assert "11.41%" in formatted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Fusion Interpretation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_fusion_mcq_close_probabilities():
    """Verify Fusion interpreter highlights uncertainty when category probabilities are close."""
    result = {
        "answer": "D (urban)",
        "confidence": 0.2805,
        "task_sub_type": "mcq",
        "model_name": "satquery-optical-sar-fusion-v1",
        "probabilities": {
            "agriculture": 0.2423,
            "forest": 0.2454,
            "water": 0.2319,
            "urban": 0.2805,
        },
    }

    interp = interpret_result(
        query="Which category best describes the image pair?",
        task="fusion",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "urban" in formatted.lower()
    assert "28.05%" in formatted
    assert "uncertainty" in formatted.lower() or "uncertain" in formatted.lower()
    assert "agriculture: 24.23%" in str(interp.key_findings) or "agriculture: 24.23%" in formatted


def test_fusion_binary_high_confidence():
    """Verify Fusion binary interpretation with high confidence."""
    result = {
        "answer": "YES",
        "confidence": 0.9788,
        "task_sub_type": "binary",
        "model_name": "satquery-optical-sar-fusion-v1",
        "probabilities": {"NO": 0.0212, "YES": 0.9788},
    }

    interp = interpret_result(
        query="Does this area contain built-up regions?",
        task="fusion",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "YES" in formatted
    assert "97.88%" in formatted
    assert "assigned high confidence (97.88%)" in interp.confidence_explanation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. VLM / VQA Interpretation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_vlm_vqa_interpretation():
    """Verify VLM interpreter formats answer cleanly with confidence and model attribution."""
    result = {
        "answer": "Dense forest and river canal visible.",
        "confidence": 0.92,
        "model_name": "satquery-vlm-person-a",
    }

    interp = interpret_result(
        query="What vegetation is present?",
        task="vqa",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "dense forest" in formatted.lower()
    assert "92.00%" in formatted
    assert "satquery-vlm-person-a" in str(interp.key_findings)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# E. GEE Supporting Evidence Interpretation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_gee_ndvi_interpretation():
    """Verify GEE NDVI interpreter extracts value, date, and provides non-overclaiming explanation."""
    result = {
        "index_value": 0.7842,
        "date": "2024-06-15",
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
    }

    interp = interpret_result(
        query="What vegetation is present?",
        task="ndvi",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "0.7842" in formatted
    assert "indicates relatively strong vegetation characteristics" in formatted.lower()
    assert "2024-06-15" in formatted
    assert "proved" not in formatted.lower()  # Non-overclaiming guarantee


def test_gee_ndbi_interpretation():
    """Verify GEE NDBI interpreter provides supporting built-up evidence without overclaiming."""
    result = {
        "index_value": 0.1500,
        "date": "2024-06-15",
        "dataset": "COPERNICUS/S2_SR_HARMONIZED",
    }

    interp = interpret_result(
        query="Did new buildings appear?",
        task="ndbi",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "0.1500" in formatted
    assert "built-up" in formatted.lower()
    assert "proved that new buildings were constructed" not in formatted.lower()


def test_gee_elevation_interpretation():
    """Verify GEE Elevation interpreter formats point elevation cleanly."""
    result = {"elevation": 1450.0, "buffer_m": 0.0}

    interp = interpret_result(
        query="What is the elevation?",
        task="elevation",
        result=result,
    )

    formatted = interp.to_formatted_answer()
    assert "1450.00" in formatted
    assert "SRTM" in formatted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F. Multi-Model Synthesis (Agreement, Conflict, GEE Integration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_synthesis_with_gee_supporting_evidence():
    """Verify Multi-Model Synthesizer incorporates GEE evidence alongside DL models."""
    sub_results = [
        {
            "task": "change_vqa",
            "model_name": "ChangeFormerV6",
            "answer": "Significant change detected across 2.77% of the area.",
            "confidence": 0.7982,
            "change_ratio": 0.0277,
            "visual_evidence": {"type": "change_mask", "changed_pixels": 1814, "total_pixels": 65536},
        },
        {
            "task": "grounding",
            "model_name": "satquery-region-grounding-v1",
            "answer": "Localized region",
            "confidence": 0.1141,
            "visual_evidence": {"type": "bbox", "coordinates": [0.1, 0.2, 0.3, 0.4]},
        },
        {
            "task": "fusion",
            "model_name": "satquery-optical-sar-fusion-v1",
            "answer": "NO",
            "confidence": 0.9788,
        },
        {
            "task": "gee",
            "answer": "The NDBI value is 0.1100, providing supporting evidence consistent with built-up characteristics.",
        },
    ]

    synth = synthesize_multi_model_results(
        original_query="Did new buildings appear?",
        sub_task_results=sub_results,
    )

    ans = synth["synthesized_answer"]

    # Key findings structure
    assert "Key findings:" in ans
    assert "ChangeFormer" in ans
    assert "Grounding" in ans
    assert "Fusion" in ans
    assert "GEE Evidence" in ans

    # Synthesis conclusion
    assert synth["synthesis"]["evidence_quality"] == "conflicting"
    assert "Overall conclusion:" in ans


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# G. Mathematical Derivation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_mathematical_derivation_accuracy():
    """Verify pixel ratios and percentage conversions are mathematically accurate."""
    changed_pixels = 32768
    total_pixels = 65536
    expected_ratio = 0.50  # 50.00%

    result = {
        "answer": "Change detected",
        "confidence": 0.90,
        "visual_evidence": {
            "type": "change_mask",
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
        },
    }

    interp = interpret_result(
        query="Did it change?",
        task="change_vqa",
        result=result,
        evidence=result["visual_evidence"],
    )

    formatted = interp.to_formatted_answer()
    assert "50.00%" in formatted
    assert "32,768 of 65,536" in formatted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# H. Distinction Between Confidence and Accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_confidence_not_described_as_accuracy():
    """Verify confidence scores are explicitly distinguished from ground-truth accuracy."""
    result = {"answer": "YES", "confidence": 0.9788}
    interp = interpret_result(query="Is there water?", task="fusion", result=result)

    explanation = interp.confidence_explanation
    assert "97.88%" in explanation
    assert "ground-truth accuracy" in explanation.lower() or "assigned high confidence" in explanation.lower()
    assert "97.88% accurate" not in explanation
