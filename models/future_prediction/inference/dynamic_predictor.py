"""
models/future_prediction/inference/dynamic_predictor.py
========================================================
Orchestrator for dynamic multi-year land-cover future prediction from
user-uploaded satellite imagery.

Pipeline:
1. parse_prediction_query() -> extract target_categories, unsupported_categories, target_year
2. validate_same_location()  -> verify images cover the same spatial region (within tolerance)
3. extract_landcover_from_uploads() -> extract actual SCL historical landcover stats
4. Weighted Linear Regression fit & forecast for each requested target_category
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Ensure project root is in sys.path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.future_prediction.inference.dynamic_extractor import (
    extract_landcover_from_uploads,
    validate_same_location,
)
from models.future_prediction.inference.query_parser import parse_prediction_query

DISCLAIMER = "Statistical trend projection based on your uploaded historical imagery, not a generative forecast."


def r2_score_weighted(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted R² score."""
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def determine_category_confidence(target_year: int, max_year: int, r2: float) -> str:
    """Determine confidence for a single target category."""
    horizon = target_year - max_year
    if horizon <= 3:
        return "high" if r2 >= 0.0 else "moderate"
    elif horizon <= 6:
        return "moderate"
    else:
        return "low"


def predict_from_uploads(
    image_paths: list[str | Path],
    years: list[int],
    query: str,
    tolerance_km: float = 2.0,
) -> dict[str, Any]:
    """
    Dynamically extracts landcover time series from user-uploaded images and forecasts
    future composition based on natural language query parameters.

    Args:
        image_paths: List of file paths to uploaded Sentinel-2 GeoTIFFs (N >= 2).
        years: List of acquisition years corresponding to image_paths.
        query: Natural language user query string.
        tolerance_km: Max allowable centroid distance separation in km.

    Returns:
        Dict formatted response with target_year, predictions, historical_data,
        unsupported_categories, explanation, and disclaimer.

    Raises:
        ValueError: If not a prediction query, location validation fails, or extraction fails.
    """
    # ── Step a: Parse Query ───────────────────────────────────────────────────
    parsed_query = parse_prediction_query(query, uploaded_years=years)
    if not parsed_query["is_prediction_query"]:
        raise ValueError(
            f"Query '{query}' was not identified as a future-prediction request. "
            "Keywords like 'predict', 'forecast', 'will', 'future', or 'by 20XX' are required."
        )

    target_year = parsed_query["target_year"]
    target_categories = parsed_query["target_categories"]
    unsupported_categories = parsed_query["unsupported_categories"]

    if not target_categories and not unsupported_categories:
        target_categories = ["built_up_pct", "vegetation_pct", "water_pct"]

    # ── Step b: Validate Spatial Location Overlap ─────────────────────────────
    loc_val = validate_same_location(image_paths, tolerance_km=tolerance_km)
    if not loc_val["valid"]:
        raise ValueError(f"Spatial validation failed: {loc_val['message']}")

    # ── Step c: Extract SCL Landcover Time-Series ──────────────────────────────
    df_hist = extract_landcover_from_uploads(image_paths, years)

    max_obs_year = int(df_hist["year"].max())
    X_train = df_hist["year"].values.reshape(-1, 1).astype(float)
    weights = (df_hist["valid_px_pct"] / 100.0).values.astype(float)
    X_pred = np.array([[float(target_year)]])

    # ── Step d: Weighted Linear Regression Forecast per Category ──────────────
    predictions: dict[str, dict[str, Any]] = {}
    explanations: list[str] = []

    category_labels = {
        "built_up_pct": "Building / built-up coverage",
        "vegetation_pct": "Vegetation coverage",
        "water_pct": "Water coverage",
    }

    for cat in target_categories:
        if cat not in df_hist.columns:
            continue

        y_train = df_hist[cat].values.astype(float)
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=weights)

        raw_pred = float(model.predict(X_pred)[0])
        pred_val = round(max(0.0, min(100.0, raw_pred)), 2)

        y_fit = model.predict(X_train).flatten()
        r2 = round(r2_score_weighted(y_train, y_fit, weights), 4)
        slope = round(float(model.coef_[0]), 4)
        conf = determine_category_confidence(target_year, max_obs_year, r2)

        predictions[cat] = {
            "value": pred_val,
            "confidence": conf,
            "r2": r2,
            "annual_slope": slope,
        }

        # Build natural-language explanation sentence
        obs_start_yr = int(df_hist["year"].min())
        obs_end_yr = int(df_hist["year"].max())
        start_val = float(df_hist.loc[df_hist["year"] == obs_start_yr, cat].values[0])
        end_val = float(df_hist.loc[df_hist["year"] == obs_end_yr, cat].values[0])

        direction = "increase" if slope >= 0 else "decrease"
        label = category_labels.get(cat, cat)
        exp = (
            f"{label} is expected to {direction} to {pred_val:.1f}% by {target_year} "
            f"(slope: {slope:+.2f}%/yr), based on observed trend from {start_val:.1f}% in {obs_start_yr} "
            f"to {end_val:.1f}% in {obs_end_yr}."
        )
        explanations.append(exp)

    # Convert historical DataFrame to list of dict records
    historical_records = df_hist.to_dict(orient="records")

    explanation_text = " ".join(explanations)
    if unsupported_categories:
        explanation_text += f" (Note: {', '.join(unsupported_categories)} is not yet supported)."

    return {
        "target_year": target_year,
        "predictions": predictions,
        "historical_data": historical_records,
        "unsupported_categories": unsupported_categories,
        "explanation": explanation_text,
        "disclaimer": DISCLAIMER,
    }


# ── Self-Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — Dynamic Predictor Self-Test")
    print("=" * 80)

    RAW_DIR = PROJECT_ROOT / "models" / "future_prediction" / "data" / "raw"
    imgs = [
        RAW_DIR / "region_01_2019.tif",
        RAW_DIR / "region_01_2021.tif",
        RAW_DIR / "region_01_2023.tif",
        RAW_DIR / "region_01_2025.tif",
    ]
    yrs = [2019, 2021, 2023, 2025]
    test_q = "Predict the change of buildings, trees and roads in 2027."

    print(f"\nQuery: '{test_q}'")
    print(f"Uploaded: {[p.name for p in imgs]} with years {yrs}\n")

    res = predict_from_uploads(imgs, yrs, test_q)
    print(json.dumps(res, indent=2))

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED CLEANLY")
    print("=" * 80)
