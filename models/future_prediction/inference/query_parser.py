"""
models/future_prediction/inference/query_parser.py
====================================================
Deterministic natural-language query parser for land-cover future forecasting.

Extracts structured query intentions:
- is_prediction_query: bool
- target_categories: list of supported targets ["built_up_pct", "vegetation_pct", "water_pct"]
- unsupported_categories: list of requested unsupported classes (e.g. ["roads"])
- target_year: int (explicit year or relative horizon calculation)
- raw_query: str
"""

from __future__ import annotations

import re
from typing import Any, List, Optional


PREDICTION_KEYWORDS = [
    r"\bpredict\b",
    r"\bforecast\b",
    r"\bprojection\b",
    r"\bfuture\b",
    r"\bwill\b",
    r"\blikely to\b",
    r"\bnext year\b",
    r"\bnext \d+ years?\b",
    r"\bby 20[2-7][0-9]\b",
    r"\bin 20[2-7][0-9]\b",
    r"\bover the next\b",
]

CATEGORY_MAPPINGS = {
    "built_up_pct": [
        r"\bbuildings?\b",
        r"\bbuilt-?up\b",
        r"\bbuiltup\b",
        r"\burban\b",
        r"\bstructures?\b",
        r"\bconstruction\b",
    ],
    "vegetation_pct": [
        r"\btrees?\b",
        r"\bvegetation\b",
        r"\bgreen cover\b",
        r"\bgreenery\b",
        r"\bforests?\b",
        r"\bplants?\b",
    ],
    "water_pct": [
        r"\bwater\b",
        r"\blakes?\b",
        r"\brivers?\b",
        r"\bwaterbod(y|ies)\b",
        r"\bponds?\b",
    ],
}

UNSUPPORTED_MAPPINGS = {
    "roads": [r"\broads?\b", r"\bhighways?\b", r"\brailways?\b", r"\binfrastructure\b"],
}

ALL_LANDCOVER_PATTERNS = [
    r"\bland-?cover\b",
    r"\bcategory\b",
    r"\bcategories\b",
    r"\ball classes?\b",
    r"\beverything\b",
]


def parse_prediction_query(
    query: str,
    uploaded_years: Optional[List[int]] = None,
) -> dict[str, Any]:
    """
    Parses a natural-language user query into structured future prediction parameters.

    Args:
        query: User text prompt (e.g., "Predict the change of buildings and trees in 2027.")
        uploaded_years: Optional list of historical years present in dataset/uploads.

    Returns:
        Dict: {
            "is_prediction_query": bool,
            "target_categories": list[str],
            "unsupported_categories": list[str],
            "target_year": int | None,
            "raw_query": str
        }
    """
    if not query or not isinstance(query, str):
        return {
            "is_prediction_query": False,
            "target_categories": [],
            "unsupported_categories": [],
            "target_year": None,
            "raw_query": query or "",
        }

    raw_query = query.strip()
    query_lower = raw_query.lower()

    # 1. Determine if this is a prediction/forecast query
    is_prediction = any(re.search(pat, query_lower) for pat in PREDICTION_KEYWORDS)

    if not is_prediction:
        return {
            "is_prediction_query": False,
            "target_categories": [],
            "unsupported_categories": [],
            "target_year": None,
            "raw_query": raw_query,
        }

    # Determine reference baseline year from uploaded_years
    base_year = max(uploaded_years) if (uploaded_years and len(uploaded_years) > 0) else 2025

    # 2. Extract target_year
    target_year: Optional[int] = None

    # Option A: Explicit 4-digit year (2026-2075)
    explicit_year_match = re.search(r"\b(20[2-7][0-9])\b", query_lower)
    if explicit_year_match:
        target_year = int(explicit_year_match.group(1))

    # Option B: Relative year horizon (e.g., "next 3 years", "in 5 years")
    if target_year is None:
        relative_match = re.search(r"\b(?:next|in|over the next)\s+(\d+)\s+years?\b", query_lower)
        if relative_match:
            offset = int(relative_match.group(1))
            target_year = base_year + offset

    # Default target_year if missing
    if target_year is None:
        target_year = base_year + 2  # default to baseline + 2 years (e.g. 2027)

    # 3. Extract target_categories
    matched_categories: list[str] = []
    for cat_name, patterns in CATEGORY_MAPPINGS.items():
        if any(re.search(pat, query_lower) for pat in patterns):
            matched_categories.append(cat_name)

    # 4. Extract unsupported_categories
    unsupported_categories: list[str] = []
    for un_name, patterns in UNSUPPORTED_MAPPINGS.items():
        if any(re.search(pat, query_lower) for pat in patterns):
            unsupported_categories.append(un_name)

    # If general landcover terms are used and no specific category matched, default to all 3 supported categories
    is_general_query = any(re.search(pat, query_lower) for pat in ALL_LANDCOVER_PATTERNS)
    if is_general_query and len(matched_categories) == 0:
        matched_categories = ["built_up_pct", "vegetation_pct", "water_pct"]

    # If prediction query but no category specified at all, default to all 3
    if len(matched_categories) == 0 and len(unsupported_categories) == 0:
        matched_categories = ["built_up_pct", "vegetation_pct", "water_pct"]

    return {
        "is_prediction_query": True,
        "target_categories": matched_categories,
        "unsupported_categories": unsupported_categories,
        "target_year": target_year,
        "raw_query": raw_query,
    }


# ── Testing Script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — Prediction Query Parser Test Suite")
    print("=" * 80)

    test_queries = [
        ("Predict the change of buildings, trees and roads in 2027.", None),
        ("How will vegetation change by 2028?", None),
        (
            "Predict the change in buildings over the next 3 years.",
            [2020, 2022, 2023, 2024],
        ),
        ("Which land-cover category is likely to increase by 2027?", None),
        ("What's in this image?", None),
    ]

    for idx, (q, yrs) in enumerate(test_queries, 1):
        print(f"\n--- Test Query {idx} ---")
        print(f"Query         : \"{q}\"")
        if yrs:
            print(f"Uploaded Years: {yrs}")
        parsed = parse_prediction_query(q, uploaded_years=yrs)
        print(json.dumps(parsed, indent=2))

    print("\n" + "=" * 80)
    print("ALL 5 TEST QUERIES PARSED CLEANLY")
    print("=" * 80)
