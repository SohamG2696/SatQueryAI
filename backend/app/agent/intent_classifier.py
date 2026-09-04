"""
SatQuery AI — Query Intent Classifier.

Extracts semantic and task intent keywords from natural-language queries.
"""

from __future__ import annotations

import re
from typing import List

_GROUNDING_KEYWORDS: set[str] = {
    "find",
    "locate",
    "location",
    "highlight",
    "show where",
    "bounding box",
    "bbox",
    "coordinates",
    "segment",
    "detect where",
    "where is",
    "where are",
    "where did",
    "position of",
    "pinpoint",
    "spot",
}

_CHANGE_KEYWORDS: set[str] = {
    "change",
    "changed",
    "changes",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "between",
    "difference",
    "before and after",
    "evolution",
    "growth",
    "loss",
    "deforestation",
    "urbanization",
    "expansion",
    "shrinkage",
}

_FUSION_KEYWORDS: set[str] = {
    "sar",
    "radar",
    "optical",
    "modality",
    "modalities",
    "cross-modal",
    "fuse",
    "fusion",
    "both images",
    "support",
    "confirm",
    "sentinel-1",
    "sentinel-2",
    "compare optical",
    "compare the optical",
    "compare optical and sar",
}

_CAPTION_KEYWORDS: set[str] = {
    "describe",
    "caption",
    "summary",
    "summarize",
    "overview",
    "what is in",
    "what do you see",
    "tell me about this scene",
    "explain this image",
}


def contains_keyword(text: str, keywords: set[str]) -> bool:
    """Check if any keyword is present in the lowercased text as a word or substring."""
    text_lower = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", text_lower) or kw in text_lower for kw in keywords)


def detect_all_intents(query: str | None) -> List[str]:
    """Detect all matching task intents in a query.

    Returns
    -------
    List[str]
        List of detected task names, e.g. ['grounding'], ['change_vqa'], or ['grounding', 'change_vqa'].
    """
    if not query or not query.strip():
        return []

    q = query.strip()
    detected: List[str] = []

    # Check for Grounding
    if contains_keyword(q, _GROUNDING_KEYWORDS):
        detected.append("grounding")

    # Check for Change
    if contains_keyword(q, _CHANGE_KEYWORDS):
        detected.append("change_vqa")

    # Check for Fusion
    if contains_keyword(q, _FUSION_KEYWORDS):
        detected.append("fusion")

    # Check for Captioning (only if no specific visual task detected)
    if not detected and contains_keyword(q, _CAPTION_KEYWORDS):
        detected.append("captioning")

    # Default fallback if nothing matched
    if not detected:
        detected.append("vqa")

    return detected


def classify_query_intent(query: str | None) -> str:
    """Identify the primary intent category of the natural language query.

    Returns
    -------
    str
        One of: 'grounding', 'captioning', 'change', 'fusion', 'question', 'multi_model', 'empty'
    """
    intents = detect_all_intents(query)
    if not intents:
        return "empty"
    if len(intents) > 1:
        return "multi_model"
    first = intents[0]
    if first == "change_vqa":
        return "change"
    if first == "vqa":
        return "question"
    return first
