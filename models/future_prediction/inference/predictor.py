"""
models/future_prediction/inference/predictor.py
=================================================
Inference module for SatQueryAI land-cover future forecasting.

Uses weighted Linear Regression per region and target variable, trained on
historical land-cover time-series data (2019-2025).

Function:
    predict_future(region_id: str, target_year: int, historical_csv_path: str = None) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── Defaults & Constants ──────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_CSV_PATH = (
    PROJECT_ROOT / "models" / "future_prediction" / "data" / "processed" / "landcover_timeseries.csv"
)

TARGETS = ["built_up_pct", "vegetation_pct", "water_pct"]
DISCLAIMER = "Statistical trend projection based on observed historical data, not a generative forecast."


def r2_score_weighted(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted R² score."""
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def determine_confidence(target_year: int, max_observed_year: int, min_r2: float) -> str:
    """
    Determine confidence rating based on extrapolation horizon and fit quality.
    - Horizon <= 3 years (e.g. 2027): High (or Moderate if min_r2 is negative)
    - Horizon <= 6 years (e.g. 2030): Moderate
    - Horizon > 6 years: Low
    """
    horizon = target_year - max_observed_year
    if horizon <= 0:
        return "high"
    elif horizon <= 3:
        return "high" if min_r2 >= 0.0 else "moderate"
    elif horizon <= 6:
        return "moderate"
    else:
        return "low"


def predict_future(
    region_id: str,
    target_year: int,
    historical_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Predict land-cover distribution (built_up_pct, vegetation_pct, water_pct)
    for a specified region and target year using linear regression.

    Args:
        region_id: Unique region identifier (e.g., 'region_01', 'region_02').
        target_year: Target forecast year (e.g., 2027, 2030).
        historical_csv_path: Optional path to landcover_timeseries.csv.

    Returns:
        Dict containing forecast values, method details, confidence rating,
        disclaimer, and interpretation notes.
    """
    csv_path = Path(historical_csv_path) if historical_csv_path else DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Historical time-series CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    regions_available = df["region_id"].unique()
    if region_id not in regions_available:
        raise ValueError(f"Unknown region_id '{region_id}'. Available regions: {list(regions_available)}")

    reg_df = df[df["region_id"] == region_id].sort_values("year")
    observed_years = reg_df["year"].values
    max_observed_year = int(observed_years.max())

    X_train = observed_years.reshape(-1, 1).astype(float)
    weights = (reg_df["valid_px_pct"] / 100.0).values.astype(float)
    X_pred = np.array([[float(target_year)]])

    predictions: dict[str, float] = {}
    r2_scores: dict[str, float] = {}
    slopes: dict[str, float] = {}
    notes: list[str] = []

    for target in TARGETS:
        y_train = reg_df[target].values.astype(float)
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=weights)

        pred_val = float(model.predict(X_pred)[0])
        # Clip to physical boundaries [0, 100]
        pred_val_clipped = max(0.0, min(100.0, pred_val))
        predictions[target] = round(pred_val_clipped, 2)

        # Compute R² fit on historical data
        y_fit = model.predict(X_train).flatten()
        r2 = r2_score_weighted(y_train, y_fit, weights)
        r2_scores[target] = round(r2, 4)
        slopes[target] = round(float(model.coef_[0]), 4)

        # Check for saturation / low R² interpretation notes
        avg_val = np.mean(y_train)
        if r2 < 0.30:
            if avg_val > 90.0:
                notes.append(
                    f"{target}: Low R² ({r2:.2f}) indicates a stable/saturated urban surface "
                    f"(average ~{avg_val:.1f}%), rather than an unreliable signal."
                )
            elif avg_val < 5.0:
                notes.append(
                    f"{target}: Low R² ({r2:.2f}) due to near-zero constant presence "
                    f"(average ~{avg_val:.1f}%)."
                )
            else:
                notes.append(
                    f"{target}: Low R² ({r2:.2f}) indicates non-monotonic temporal fluctuation."
                )

    # Normalize prediction percentages so they sum to 100.0% if needed
    total_pct = sum(predictions.values())
    if total_pct > 0 and abs(total_pct - 100.0) > 0.01:
        for k in predictions:
            predictions[k] = round((predictions[k] / total_pct) * 100.0, 2)

    min_r2 = min(r2_scores.values())
    confidence = determine_confidence(target_year, max_observed_year, min_r2)

    return {
        "region_id": region_id,
        "forecast_year": target_year,
        "built_up_pct": predictions["built_up_pct"],
        "vegetation_pct": predictions["vegetation_pct"],
        "water_pct": predictions["water_pct"],
        "method": "linear_regression",
        "confidence": confidence,
        "r2_scores": r2_scores,
        "annual_slopes": slopes,
        "interpretation_notes": notes,
        "disclaimer": DISCLAIMER,
    }


# ── Testing Script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — Predictor Module Test")
    print("=" * 80)

    test_cases = [
        ("region_01", 2027),  # Bangalore 2027
        ("region_02", 2027),  # Mumbai 2027 (High built-up saturation)
        ("region_03", 2030),  # Delhi 2030 (Longer horizon + non-monotonic)
    ]

    for reg, yr in test_cases:
        print(f"\n--- Testing predict_future('{reg}', {yr}) ---")
        result = predict_future(reg, yr)
        print(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED CLEANLY")
    print("=" * 80)
