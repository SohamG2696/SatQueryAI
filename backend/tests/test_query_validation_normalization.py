"""
SatQuery AI — Unit & Integration Tests for Query Validation & Normalization Layer.

Verifies:
1. >= 10 valid scene-description paraphrases normalize to identical canonical intent/prompt.
2. >= 10 valid VQA queries normalize to VQA intent with preserved semantics.
3. >= 10 grounding queries extract structured target and location intent.
4. >= 10 invalid/unrelated queries are rejected with valid=False and reason.
5. Invalid queries bypass all specialist model execution.
6. Inference caching returns consistent cached responses for identical canonical tasks.
7. VLM deterministic decoding settings prevent infinite token repetition loops.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

from app.agent.controller import controller
from app.agent.query_validator import (
    CANONICAL_SCENE_DESCRIPTION_PROMPT,
    extract_grounding_target,
    is_scene_description,
    validate_query_intent,
)
from app.schemas.response import QueryResponse
from app.services.cache_service import inference_cache


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Scene-Description Paraphrases (>= 10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENE_DESCRIPTION_PARAPHRASES = [
    "Describe the image",
    "What is in the image?",
    "What is there in the image?",
    "What can you see?",
    "What can you see here?",
    "Tell me about this image",
    "Describe the scene",
    "Give me an overview of this satellite image",
    "Summarize this image",
    "What does this aerial photograph show?",
    "Provide a detailed description of the landscape",
    "Explain what is visible in this satellite image",
]


@pytest.mark.parametrize("query", SCENE_DESCRIPTION_PARAPHRASES)
def test_scene_description_paraphrases(query: str):
    """Verify all scene description variations normalize to scene_description canonical task."""
    result = validate_query_intent(query, image_count=1)
    assert result.valid is True
    assert result.intent == "scene_description"
    assert result.canonical_task == "scene_description"
    assert result.canonical_prompt == CANONICAL_SCENE_DESCRIPTION_PROMPT
    assert result.confidence >= 0.90


def test_scene_description_identical_canonical_mapping():
    """Verify different paraphrases produce the exact same canonical prompt."""
    results = [validate_query_intent(q, image_count=1) for q in SCENE_DESCRIPTION_PARAPHRASES]
    prompts = {r.canonical_prompt for r in results}
    tasks = {r.canonical_task for r in results}
    intents = {r.intent for r in results}

    assert len(prompts) == 1
    assert len(tasks) == 1
    assert len(intents) == 1
    assert list(prompts)[0] == CANONICAL_SCENE_DESCRIPTION_PROMPT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Valid VQA Queries (>= 10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_VQA_QUERIES = [
    ("Is there a train?", "Is there a train?"),
    ("How many airplanes are visible on the tarmac?", "How many airplanes are visible on the tarmac?"),
    ("Is this area mostly urban or rural?", "Is this area mostly urban or rural?"),
    ("Are there solar panels on the roofs?", "Are there solar panels on the roofs?"),
    ("What color is the roof of the largest building?", "What color is the roof of the largest building?"),
    ("Are there any ships docked at the harbor?", "Are there any ships docked at the harbor?"),
    ("Is there snow cover in this region?", "Is there snow cover in this region?"),
    ("Does this area have industrial storage tanks?", "Does this area have industrial storage tanks?"),
    ("Are there residential houses near the river?", "Are there residential houses near the river?"),
    ("Is the cloud coverage high in this image?", "Is the cloud coverage high in this image?"),
    ("What type of crop is planted in the circular fields?", "What type of crop is planted in the circular fields?"),
]


@pytest.mark.parametrize("query,expected_normalized", VALID_VQA_QUERIES)
def test_valid_vqa_queries(query: str, expected_normalized: str):
    """Verify VQA queries are accepted and preserve question semantics."""
    result = validate_query_intent(query, image_count=1)
    assert result.valid is True
    assert result.intent == "vqa"
    assert result.canonical_task == "vqa"
    assert result.canonical_prompt == expected_normalized
    assert result.confidence >= 0.85


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Grounding Queries (>= 10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_GROUNDING_QUERIES = [
    ("Where is the river?", "river", "locate"),
    ("Locate the water body", "water_body", "locate"),
    ("Highlight the water-covered area", "water_body", "locate"),
    ("Find the runway", "runway", "locate"),
    ("Where are the buildings?", "building", "locate"),
    ("Locate the agricultural fields", "agriculture", "locate"),
    ("Highlight forest area", "forest", "locate"),
    ("Find the sports stadium", "stadium", "locate"),
    ("Where is the bridge?", "bridge", "locate"),
    ("Locate the oil storage tanks", "tanks", "locate"),
    ("Pinpoint the railway tracks", "railway", "locate"),
]


@pytest.mark.parametrize("query,expected_target,expected_op", VALID_GROUNDING_QUERIES)
def test_grounding_queries(query: str, expected_target: str, expected_op: str):
    """Verify spatial grounding queries extract target entity and normalize prompt."""
    result = validate_query_intent(query, image_count=1)
    assert result.valid is True
    assert result.intent == "grounding"
    assert result.canonical_task == "grounding"
    assert result.target == expected_target
    assert result.operation == expected_op
    assert result.canonical_prompt.startswith("Locate and segment the ")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Invalid / Unrelated Queries (>= 10 tests)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INVALID_UNRELATED_QUERIES = [
    "What is the capital of France?",
    "Write Python code to reverse a linked list",
    "Who won the football match?",
    "Write a poem about the sunrise",
    "Tell me a joke",
    "Give me a recipe for chocolate chip cookies",
    "Who was Albert Einstein?",
    "Solve the equation 2x + 5 = 15",
    "Translate hello world to French",
    "How to cook pasta at home?",
    "What is the population of Tokyo?",
    "How do I fix a leaky faucet in my kitchen?",
]


@pytest.mark.parametrize("query", INVALID_UNRELATED_QUERIES)
def test_invalid_unrelated_queries(query: str):
    """Verify out-of-domain queries are rejected cleanly."""
    result = validate_query_intent(query, image_count=1)
    assert result.valid is False
    assert result.intent == "invalid"
    assert result.canonical_task == "invalid"
    assert result.confidence == 0.0
    assert result.reason is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Multi-Image Specialist & Geospatial Intents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_change_analysis_intent():
    """Verify bi-temporal change queries map to change_analysis."""
    queries = [
        "What changed between these images?",
        "Did vegetation increase or decrease between the two dates?",
        "Detect urban expansion between before and after",
    ]
    for q in queries:
        res = validate_query_intent(q, image_count=2)
        assert res.valid is True
        assert res.intent == "change_analysis"
        assert res.canonical_task == "change_vqa"


def test_optical_sar_fusion_intent():
    """Verify cross-modal fusion queries map to optical_sar_analysis."""
    queries = [
        "Use the optical and SAR images together",
        "Fuse optical and radar modalities to analyze structures",
        "Combine Sentinel-1 and Sentinel-2 imagery",
    ]
    for q in queries:
        res = validate_query_intent(q, image_count=2)
        assert res.valid is True
        assert res.intent == "optical_sar_analysis"
        assert res.canonical_task == "fusion"


def test_geospatial_ndvi_intent():
    """Verify NDVI query maps to ndvi_analysis."""
    res = validate_query_intent("Calculate NDVI for this location", image_count=0)
    assert res.valid is True
    assert res.intent == "ndvi_analysis"
    assert res.canonical_task == "gee"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Controller Execution & Zero Model Execution for Invalid Queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_controller_rejects_invalid_query_without_model_execution():
    """Verify controller returns structured rejection and does NOT invoke inference service."""
    dummy_img = Image.new("RGB", (64, 64), color="green")

    with patch("app.services.inference_service.inference_service.run_inference") as mock_infer:
        resp = controller.process_query(
            images=[dummy_img],
            query="What is the capital of France?",
        )

        # Specialist model must NOT have been called
        mock_infer.assert_not_called()

        assert resp.valid is False
        assert resp.task_detected == "invalid"
        assert resp.canonical_task is None or resp.canonical_task == "invalid"
        assert "Invalid query" in resp.answer or "outside SatQuery's supported" in resp.answer
        assert resp.confidence == 0.0
        assert resp.execution_summary.models_used == []
        assert resp.execution_summary.parameters.get("valid") is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Lightweight Result Caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_lightweight_cache_consistency():
    """Verify same image + same canonical task yields cached response."""
    inference_cache.clear()
    img_a = Image.new("RGB", (64, 64), color="blue")
    img_b = Image.new("RGB", (64, 64), color="red")

    mock_output = {
        "answer": "A dense urban landscape with residential zones.",
        "confidence": 0.95,
        "visual_evidence": {"type": "none"},
        "model_name": "satquery-vlm-person-a",
        "parameters": {},
    }

    with patch("app.services.inference_service.inference_service.run_inference", return_value=mock_output) as mock_infer:
        # First call with "Describe the image"
        resp1 = controller.process_query(images=[img_a], query="Describe the image")
        assert mock_infer.call_count == 1
        assert resp1.valid is True

        # Second call with semantically equivalent paraphrase on SAME image -> Should hit cache!
        resp2 = controller.process_query(images=[img_a], query="What is in the image?")
        assert mock_infer.call_count == 1  # No additional model execution!
        assert resp2.answer == resp1.answer
        assert resp2.execution_summary.parameters.get("cached") is True

        # Third call with DIFFERENT image -> Cache miss, should execute model
        resp3 = controller.process_query(images=[img_b], query="Describe the image")
        assert mock_infer.call_count == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Deterministic Generation Anti-Looping Settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_vlm_adapter_generation_settings():
    """Verify VLM adapter invokes generate with deterministic and anti-loop parameters."""
    from models.vlm.vlm_adapter import VLMAdapter

    # Mock model and processor
    with patch.object(VLMAdapter, "__init__", lambda self, *args, **kwargs: None):
        adapter = VLMAdapter()
        adapter.device = "cpu"
        adapter.processor = MagicMock()
        adapter.model = MagicMock()
        adapter.processor.apply_chat_template.return_value = "chat_prompt"
        adapter.processor.return_value.to.return_value = {
            "input_ids": MagicMock(shape=[1, 10]),
        }
        adapter.model.config.eos_token_id = 2
        adapter.model.config.pad_token_id = 2
        adapter.processor.tokenizer = None

        # Return simulated repeating generated tokens to test post-processing
        adapter.model.generate.return_value = [[0] * 20]
        adapter.processor.decode.return_value = "Two large buildings. A river. A river. A train track."

        dummy_img = Image.new("RGB", (384, 384), color="gray")
        res = adapter.predict(image=dummy_img, question="Describe this satellite image.")

        # Verify generate called with do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3
        _, gen_kwargs = adapter.model.generate.call_args
        assert gen_kwargs.get("do_sample") is False
        assert gen_kwargs.get("repetition_penalty") == 1.2
        assert gen_kwargs.get("no_repeat_ngram_size") == 3
        assert gen_kwargs.get("eos_token_id") == 2

        # Verify consecutive duplicate sentence suppression in post-processing
        assert "A river. A river." not in res["prediction"]
