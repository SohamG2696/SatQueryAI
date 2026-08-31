"""
SatQuery AI — Query Intent Classifier.

Extracts semantic and task intent keywords from natural-language queries.
"""

from __future__ import annotations

import re

# Keywords indicating spatial region grounding
_GROUNDING_KEYWORDS: set[str] = {
    "where",
    "locate",
    "location",
    "region",
    "highlight",
    "show",
    "find",
    "point",
    "mark",
    "bounding",
    "bbox",
    "area",
    "zone",
    "coordinates",
    "segment",
    "detect",
}

# Keywords indicating captioning / scene summarization
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

# Keywords indicating change detection
_CHANGE_KEYWORDS: set[str] = {
    "change",
    "changed",
    "increased",
    "decreased",
    "between",
    "difference",
    "before",
    "after",
    "evolution",
    "growth",
    "loss",
    "deforestation",
    "urbanization",
    "flood extent",
    "expansion",
}

# Keywords indicating optical-sar cross-modal fusion
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
}


def contains_keyword(text: str, keywords: set[str]) -> bool:
    """Check if any keyword is present in the lowercased text as a word or substring."""
    text_lower = text.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", text_lower) or kw in text_lower for kw in keywords)


def classify_query_intent(query: str | None) -> str:
    """Identify the primary intent category of the natural language query.

    Returns
    -------
    str
        One of: 'grounding', 'captioning', 'change', 'fusion', 'question', 'empty'
    """
    if not query or not query.strip():
        return "empty"

    q = query.strip()

    # Priority 1: Grounding / Localization
    if contains_keyword(q, _GROUNDING_KEYWORDS):
        return "grounding"

    # Priority 2: Change keywords
    if contains_keyword(q, _CHANGE_KEYWORDS):
        return "change"

    # Priority 3: Fusion keywords
    if contains_keyword(q, _FUSION_KEYWORDS):
        return "fusion"

    # Priority 4: Captioning keywords
    if contains_keyword(q, _CAPTION_KEYWORDS):
        return "captioning"

    return "question"
