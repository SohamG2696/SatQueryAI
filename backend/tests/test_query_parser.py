"""
Unit tests for future prediction natural language query parser.
"""

from pathlib import Path
import pytest
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.future_prediction.inference.query_parser import parse_prediction_query


def test_parse_explicit_year_and_unsupported_category():
    q = "Predict the change of buildings, trees and roads in 2027."
    res = parse_prediction_query(q)
    assert res["is_prediction_query"] is True
    assert "built_up_pct" in res["target_categories"]
    assert "vegetation_pct" in res["target_categories"]
    assert "roads" in res["unsupported_categories"]
    assert res["target_year"] == 2027


def test_parse_single_category():
    q = "How will vegetation change by 2028?"
    res = parse_prediction_query(q)
    assert res["is_prediction_query"] is True
    assert res["target_categories"] == ["vegetation_pct"]
    assert res["unsupported_categories"] == []
    assert res["target_year"] == 2028


def test_parse_relative_year():
    q = "Predict the change in buildings over the next 3 years."
    uploaded = [2020, 2022, 2023, 2024]
    res = parse_prediction_query(q, uploaded_years=uploaded)
    assert res["is_prediction_query"] is True
    assert res["target_categories"] == ["built_up_pct"]
    assert res["target_year"] == 2027  # 2024 + 3


def test_parse_general_category_query():
    q = "Which land-cover category is likely to increase by 2027?"
    res = parse_prediction_query(q)
    assert res["is_prediction_query"] is True
    assert set(res["target_categories"]) == {"built_up_pct", "vegetation_pct", "water_pct"}
    assert res["target_year"] == 2027


def test_parse_non_prediction_query():
    q = "What's in this image?"
    res = parse_prediction_query(q)
    assert res["is_prediction_query"] is False
    assert res["target_categories"] == []
    assert res["unsupported_categories"] == []
    assert res["target_year"] is None
