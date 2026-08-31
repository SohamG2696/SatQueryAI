"""
Tests for Agentic Controller & Deterministic Router.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.router import route_request
from app.agent.intent_classifier import classify_query_intent


def test_intent_classifier():
    """Verify semantic classification of queries."""
    assert classify_query_intent("Locate the smallest forest region.") == "grounding"
    assert classify_query_intent("Highlight the water body.") == "grounding"
    assert classify_query_intent("Describe this satellite scene.") == "captioning"
    assert classify_query_intent("Summarize what is in this image.") == "captioning"
    assert classify_query_intent("Has vegetation increased between these two dates?") == "change"
    assert classify_query_intent("What is the difference between before and after?") == "change"
    assert classify_query_intent("Does SAR imagery support optical land cover?") == "fusion"
    assert classify_query_intent("Is there agricultural land in this image?") == "question"
    assert classify_query_intent("") == "empty"


def test_routing_cases():
    """Verify all 5 deterministic routing cases."""
    # Case 1: VQA (1 image + normal question)
    task, route = route_request("Is there an airport visible?", image_count=1)
    assert task == "vqa"
    assert route == "single_image_vqa"

    # Case 2: Captioning (1 image + describe query or empty)
    task, route = route_request("Describe this scene.", image_count=1)
    assert task == "captioning"
    assert route == "single_image_captioning"

    task, route = route_request("", image_count=1)
    assert task == "captioning"

    # Case 3: Grounding (1 image + spatial keywords)
    task, route = route_request("Locate the water body in this image.", image_count=1)
    assert task == "grounding"
    assert route == "single_image_region_grounding"

    # Case 4: Change VQA (2 images, different dates)
    task, route = route_request(
        "Did urban areas expand?",
        image_count=2,
        dates=["2022-01-01", "2025-01-01"],
    )
    assert task == "change_vqa"
    assert route == "bi_temporal_change_analysis"

    # Case 5: Optical-SAR Fusion (2 images, optical + SAR)
    task, route = route_request(
        "Confirm if built-up area is present using both modalities.",
        image_count=2,
        modalities=["optical", "sar"],
    )
    assert task == "fusion"
    assert route == "two_image_cross_modal_fusion"


def test_routing_zero_images():
    """Verify error on zero images."""
    with pytest.raises(ValueError):
        route_request("Test question", image_count=0)
