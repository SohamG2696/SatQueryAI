"""
SatQuery AI — Strict Domain Guardrail Test Suite.

Tests:
1. VALID remote-sensing queries (scene description, grounding, change, vqa, spectral).
2. INVALID out-of-domain queries (trivia, how-to, general knowledge, coding, math, recipes).
3. OBJECT-ABOUT vs. OBJECT-IN-IMAGE distinction (e.g. "Is there a dog in the image?" vs "How does a dog walk?").
4. Input / image compatibility checks (e.g. change detection requiring 2+ images).
5. Typo tolerance on visual scene description inquiries.
6. Verification that INVALID queries NEVER invoke LLaVA or any specialist model (models_used == []).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest
from PIL import Image

from app.agent.controller import controller
from app.agent.query_validator import (
    DomainGuardrailResult,
    GuardrailStatus,
    evaluate_domain_guardrail,
    validate_query_intent,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Valid Remote-Sensing Domain Queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_QUERIES = [
    ("Describe the image", 1),
    ("What is in the image?", 1),
    ("What is there in the image?", 1),
    ("What can you see?", 1),
    ("Is there a river?", 1),
    ("Where is the river?", 1),
    ("Describe the buildings.", 1),
    ("What changed between these images?", 2),
    ("Has vegetation decreased?", 2),
    ("Highlight the water body.", 1),
    ("Are there buildings visible in this scene?", 1),
    ("How many airplanes are on the tarmac?", 1),
    ("What color is the largest roof?", 1),
    ("Calculate NDVI for this location", 0),
]


@pytest.mark.parametrize("query,img_count", VALID_QUERIES)
def test_valid_remote_sensing_queries(query: str, img_count: int):
    """Verify supported remote-sensing queries pass the domain guardrail."""
    guardrail = evaluate_domain_guardrail(query, image_count=img_count)
    assert guardrail.domain_valid is True, f"Expected '{query}' to pass guardrail, reason: {guardrail.reason}"
    assert guardrail.domain == "remote_sensing"
    assert guardrail.domain_confidence >= 0.85
    assert guardrail.reason is None

    val = validate_query_intent(query, image_count=img_count)
    assert val.valid is True
    assert val.intent != "invalid"
    assert val.canonical_task != "invalid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Invalid Out-of-Domain Queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INVALID_QUERIES = [
    "How does a dog walk?",
    "Is there a dog in the image?",
    "Is there a dog?",
    "Is there a dog between these images?",
    "Did a cat appear?",
    "Find the dog in this picture",
    "How many people are sitting on chairs?",
    "What is the capital of France?",
    "Who is the Prime Minister of India?",
    "Write Python code.",
    "Tell me a joke.",
    "What is 2 + 2?",
    "Explain quantum mechanics.",
    "How do I cook pasta?",
    "What is today's weather?",
    "Write an email to my boss.",
    "Who won the football match?",
    "How do engines work?",
]


@pytest.mark.parametrize("query", INVALID_QUERIES)
def test_invalid_out_of_domain_queries(query: str):
    """Verify out-of-domain queries are rejected with domain_valid=False."""
    guardrail = evaluate_domain_guardrail(query, image_count=1)
    assert guardrail.domain_valid is False, f"Expected '{query}' to be rejected"
    assert guardrail.domain == "unsupported"
    assert guardrail.domain_confidence >= 0.90
    assert guardrail.reason is not None

    val = validate_query_intent(query, image_count=1)
    assert val.valid is False
    assert val.intent == "invalid"
    assert val.canonical_task == "invalid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Object-in-Image vs. General Knowledge Contrast Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBJECT_CONTRAST_PAIRS = [
    # (Valid: In-Image Visual Question, Invalid: General World Knowledge)
    ("Is there a ship in the harbor?", "How does a boat engine work?"),
    ("Is there a train in the image?", "How fast does a train travel?"),
    ("Is there water in the image?", "How is water purified?"),
    ("Describe the buildings in the image.", "How are buildings constructed?"),
    ("Are there airplanes on the runway?", "How does an airplane fly?"),
    ("Where is the water body?", "Why is water wet?"),
]


@pytest.mark.parametrize("valid_q,invalid_q", OBJECT_CONTRAST_PAIRS)
def test_object_in_image_vs_general_knowledge(valid_q: str, invalid_q: str):
    """Verify guardrail accurately distinguishes in-image visual queries from general knowledge."""
    # 1. Valid in-image visual question
    g_valid = evaluate_domain_guardrail(valid_q, image_count=1)
    assert g_valid.domain_valid is True, f"Query '{valid_q}' should be VALID but got: {g_valid.reason}"
    assert g_valid.domain == "remote_sensing"

    val_valid = validate_query_intent(valid_q, image_count=1)
    assert val_valid.valid is True
    assert val_valid.intent in ("vqa", "scene_description", "grounding")

    # 2. Invalid general knowledge question
    g_invalid = evaluate_domain_guardrail(invalid_q, image_count=1)
    assert g_invalid.domain_valid is False, f"Query '{invalid_q}' should be INVALID"
    assert g_invalid.domain == "unsupported"

    val_invalid = validate_query_intent(invalid_q, image_count=1)
    assert val_invalid.valid is False
    assert val_invalid.intent == "invalid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Input / Image Compatibility Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_image_query_compatibility_single_image_scene_description():
    """One optical image + scene description -> VALID."""
    res = evaluate_domain_guardrail("Describe the image", image_count=1)
    assert res.domain_valid is True
    assert res.domain == "remote_sensing"


def test_image_query_compatibility_two_images_change_detection():
    """Two temporal images + change query -> VALID."""
    res = evaluate_domain_guardrail("What changed between these images?", image_count=2)
    assert res.domain_valid is True
    assert res.domain == "remote_sensing"


def test_image_query_compatibility_single_image_change_detection_rejected():
    """One image + temporal change query → NEEDS_CLARIFICATION (not INVALID)."""
    res = evaluate_domain_guardrail("What changed between these images?", image_count=1)
    assert res.domain_valid is False
    assert res.domain == "incompatible_input"
    # Must be NEEDS_CLARIFICATION, not a hard INVALID rejection
    assert res.status == GuardrailStatus.NEEDS_CLARIFICATION
    assert res.task_detected == "change_analysis"
    assert "required" in res.reason.lower()

    # validate_query_intent must also surface needs_clarification
    val = validate_query_intent("What changed between these images?", image_count=1)
    assert val.valid is False
    assert val.status == GuardrailStatus.NEEDS_CLARIFICATION
    assert val.intent == "change_analysis"
    assert val.canonical_task == "change_analysis"


def test_image_query_compatibility_optical_sar_without_sar_rejected():
    """One optical image + optical-SAR query without SAR modality -> INVALID."""
    res = evaluate_domain_guardrail(
        "Use the optical and SAR images together.",
        image_count=1,
        metadata={"modalities": ["optical"]},
    )
    assert res.domain_valid is False
    assert res.domain == "incompatible_input"
    assert "optical-sar fusion requires" in res.reason.lower()


def test_image_query_compatibility_optical_sar_with_sar_valid():
    """Two images with optical and SAR modalities -> VALID."""
    res = evaluate_domain_guardrail(
        "Use the optical and SAR images together.",
        image_count=2,
        metadata={"modalities": ["optical", "sar"]},
    )
    assert res.domain_valid is True
    assert res.domain == "remote_sensing"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Typo Tolerance for Scene Description Inquiries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TYPO_QUERIES = [
    "whar iaa desceibed i image",
    "desribe the image",
    "what is ther in the image",
    "what is in image",
    "describe scene",
]


@pytest.mark.parametrize("query", TYPO_QUERIES)
def test_typo_tolerant_scene_description(query: str):
    """Verify common typos in scene description inquiries normalize properly."""
    res = validate_query_intent(query, image_count=1)
    assert res.valid is True
    assert res.intent == "scene_description"
    assert res.canonical_task == "scene_description"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Zero Specialist / LLaVA Execution for Invalid Queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.mark.parametrize("invalid_query", INVALID_QUERIES)
def test_controller_zero_model_execution_for_invalid_queries(invalid_query: str):
    """Verify controller immediately rejects invalid queries and NEVER executes LLaVA or specialists."""
    dummy_img = Image.new("RGB", (64, 64), color="green")

    with patch("app.services.inference_service.inference_service.run_inference") as mock_infer, \
         patch("models.vlm.vlm_adapter.VLMAdapter.predict") as mock_vlm:
        
        response = controller.process_query(
            images=[dummy_img],
            query=invalid_query,
        )

        # Confirm zero inference execution
        mock_infer.assert_not_called()
        mock_vlm.assert_not_called()

        # Confirm structured rejection output
        assert response.valid is False
        assert response.task_detected == "invalid"
        assert response.canonical_task is None
        assert response.intent == "invalid"
        assert "Invalid query" in response.answer
        assert response.reason is not None
        assert response.execution_summary.models_used == []
        assert response.execution_summary.parameters.get("valid") is False
        assert any("Domain guardrail" in t for t in response.execution_summary.parameters.get("execution_trace", []))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 3-State Validation: Change-Analysis Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# (query, image_count, expected_status, expected_task_detected)
_THREE_STATE_CASES = [
    # ── VALID: two images + various change/comparison phrasings ────────────
    ("What are the changes?",                   2, GuardrailStatus.VALID,                "change_analysis"),
    ("What are the changes and similar sort of?", 2, GuardrailStatus.VALID,               "change_analysis"),
    ("Compare these images",                     2, GuardrailStatus.VALID,                "change_analysis"),
    ("What changed between these images?",        2, GuardrailStatus.VALID,                "change_analysis"),
    ("How are these images different?",           2, GuardrailStatus.VALID,                "change_analysis"),
    # ── NEEDS_CLARIFICATION: one image + change query ─────────────────
    ("What changed?",                            1, GuardrailStatus.NEEDS_CLARIFICATION,   "change_analysis"),
    ("What are the changes?",                    1, GuardrailStatus.NEEDS_CLARIFICATION,   "change_analysis"),
    ("Compare these images",                     1, GuardrailStatus.NEEDS_CLARIFICATION,   "change_analysis"),
    # ── INVALID: unrelated queries (image_count irrelevant) ───────────
    ("How does a dog walk?",                     1, GuardrailStatus.INVALID,              "invalid"),
    ("What is the capital of France?",            1, GuardrailStatus.INVALID,              "invalid"),
    ("Write Python code.",                        1, GuardrailStatus.INVALID,              "invalid"),
    ("Tell me a joke.",                           1, GuardrailStatus.INVALID,              "invalid"),
]


@pytest.mark.parametrize("query,img_count,expected_status,expected_task", _THREE_STATE_CASES)
def test_three_state_validation(
    query: str,
    img_count: int,
    expected_status: str,
    expected_task: str,
) -> None:
    """Verify the 3-state validation result matches expectation for each scenario."""
    # Test at the guardrail level
    guardrail = evaluate_domain_guardrail(query, image_count=img_count)
    assert guardrail.status == expected_status, (
        f"[guardrail] '{query}' (n={img_count}) → status={guardrail.status!r}, "
        f"expected {expected_status!r}. Reason: {guardrail.reason}"
    )
    assert guardrail.task_detected == expected_task, (
        f"[guardrail] '{query}' task_detected={guardrail.task_detected!r}, "
        f"expected {expected_task!r}"
    )

    # Test at the validation layer
    val = validate_query_intent(query, image_count=img_count)
    assert val.status == expected_status, (
        f"[validate] '{query}' (n={img_count}) → status={val.status!r}, "
        f"expected {expected_status!r}. Reason: {val.reason}"
    )

    if expected_status == GuardrailStatus.VALID:
        assert val.valid is True
        assert val.intent == expected_task
        assert val.canonical_task not in ("invalid", None)
    elif expected_status == GuardrailStatus.NEEDS_CLARIFICATION:
        assert val.valid is False
        assert val.intent == expected_task
        assert val.reason is not None and len(val.reason) > 0
    else:  # INVALID
        assert val.valid is False
        assert val.intent == "invalid"
        assert val.canonical_task == "invalid"


def test_needs_clarification_has_correct_json_shape() -> None:
    """Validate the exact JSON shape for NEEDS_CLARIFICATION as per the spec."""
    val = validate_query_intent("What changed?", image_count=1)
    d = val.to_dict()

    assert d["valid"] is False
    assert d["status"] == "needs_clarification"
    assert d["task_detected"] == "change_analysis"
    assert d["canonical_task"] == "change_analysis"
    assert d["reason"] == "Two compatible images are required for change analysis."


def test_invalid_has_correct_json_shape() -> None:
    """Validate the exact JSON shape for INVALID as per the spec."""
    val = validate_query_intent("How does a dog walk?", image_count=1)
    d = val.to_dict()

    assert d["valid"] is False
    assert d["status"] == "invalid"
    assert d["task_detected"] == "invalid"
    assert d["canonical_task"] is None
    assert d["reason"] is not None


def test_valid_change_analysis_has_correct_json_shape() -> None:
    """Validate the exact JSON shape for VALID change_analysis as per the spec."""
    val = validate_query_intent("What are the changes?", image_count=2)
    d = val.to_dict()

    assert d["valid"] is True
    assert d["status"] == "valid"
    assert d["task_detected"] == "change_analysis"
    assert d["canonical_task"] == "change_vqa"


def test_needs_clarification_zero_model_execution() -> None:
    """NEEDS_CLARIFICATION queries must never invoke any specialist model."""
    dummy_img = Image.new("RGB", (64, 64), color="green")

    with patch("app.services.inference_service.inference_service.run_inference") as mock_infer, \
         patch("models.vlm.vlm_adapter.VLMAdapter.predict") as mock_vlm:

        response = controller.process_query(
            images=[dummy_img],
            query="What changed?",
        )

        # Zero model execution guaranteed
        mock_infer.assert_not_called()
        mock_vlm.assert_not_called()

        assert response.valid is False
        assert response.task_detected == "change_analysis"
        assert response.canonical_task is None
        assert response.execution_summary.models_used == []
        assert "clarification" in response.answer.lower() or "required" in response.answer.lower()
