"""
models/future_prediction/training/train_baseline.py
=====================================================
Baseline Linear Regression forecaster for land-cover time series.

For each of 8 regions, fits separate per-target LinearRegression models:
    year -> built_up_pct
    year -> vegetation_pct
    year -> water_pct

Evaluation :  train on [2019, 2021, 2023], predict 2025 vs actual.
Production  :  train on [2019, 2021, 2023, 2025], forecast 2027.

sample_weight = valid_px_pct / 100  (low-confidence rows count less, not dropped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
CSV_PATH     = PROJECT_ROOT / "models" / "future_prediction" / "data" / "processed" / "landcover_timeseries.csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "future_prediction" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV  = CKPT_DIR / "baseline_results.csv"

TARGETS = ["built_up_pct", "vegetation_pct", "water_pct"]
EVAL_YEARS   = [2019, 2021, 2023]   # training years for evaluation
EVAL_TARGET  = 2025                 # held-out test year
PROD_YEARS   = [2019, 2021, 2023, 2025]  # all observed years for production model
PROD_FORECAST= 2027                 # one-step-ahead forecast

R2_LOW_THRESHOLD = 0.30             # warn if R² < this


# ── Helpers ───────────────────────────────────────────────────────────────────

def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_pred: np.ndarray,
) -> tuple[float, LinearRegression]:
    """Fit weighted LinearRegression and predict X_pred."""
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=w_train)
    pred = float(model.predict(X_pred)[0])
    return pred, model


def r2_score_weighted(y_true, y_pred, weights):
    """Weighted R² = 1 - SS_res / SS_tot (both weighted)."""
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def quality_flag(r2: float) -> str:
    if r2 < 0:
        return "NEGATIVE_R2"
    if r2 < R2_LOW_THRESHOLD:
        return "LOW_R2"
    return "OK"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — Baseline Linear Regression Forecaster")
    print("=" * 80)

    df = pd.read_csv(CSV_PATH)
    print(f"\nLoaded {len(df)} rows from {CSV_PATH}")
    print(f"Regions: {sorted(df['region_id'].unique())}")
    print(f"Years  : {sorted(df['year'].unique())}")
    print(f"Targets: {TARGETS}")

    regions = sorted(df["region_id"].unique())

    eval_rows    = []   # evaluation results (train 2019/21/23, predict 2025)
    forecast_rows = []  # production forecasts (train all 4, predict 2027)
    warn_rows    = []   # quality warnings

    # ── Per-region, per-target modelling ──────────────────────────────────────
    for reg in regions:
        reg_df = df[df["region_id"] == reg].set_index("year").sort_index()

        for target in TARGETS:

            # ── EVALUATION MODEL (train 2019/2021/2023, test 2025) ────────────
            eval_df = reg_df.loc[reg_df.index.isin(EVAL_YEARS)]
            X_train = eval_df.index.values.reshape(-1, 1).astype(float)
            y_train = eval_df[target].values.astype(float)
            w_train = (eval_df["valid_px_pct"] / 100.0).values.astype(float)

            X_test  = np.array([[float(EVAL_TARGET)]])
            pred_2025, eval_model = fit_predict(X_train, y_train, w_train, X_test)

            # In-sample R² on training years (weighted)
            y_train_pred = eval_model.predict(X_train).flatten()
            r2_eval = r2_score_weighted(y_train, y_train_pred, w_train)

            # Actual 2025 value
            if EVAL_TARGET in reg_df.index:
                actual_2025 = float(reg_df.loc[EVAL_TARGET, target])
                w_test      = float(reg_df.loc[EVAL_TARGET, "valid_px_pct"]) / 100.0
            else:
                actual_2025 = float("nan")
                w_test = 1.0

            error       = pred_2025 - actual_2025
            abs_error   = abs(error)

            qflag = quality_flag(r2_eval)
            if qflag != "OK":
                warn_rows.append({
                    "region_id": reg, "target": target,
                    "model": "eval", "r2": round(r2_eval, 4), "flag": qflag,
                })

            eval_rows.append({
                "region_id":    reg,
                "target":       target,
                "pred_2025":    round(pred_2025, 3),
                "actual_2025":  round(actual_2025, 3),
                "error":        round(error, 3),
                "abs_error":    round(abs_error, 3),
                "r2_train":     round(r2_eval, 4),
                "slope":        round(float(eval_model.coef_[0]), 4),
                "quality_flag": qflag,
            })

            # ── PRODUCTION MODEL (train all 4 years, forecast 2027) ───────────
            prod_df = reg_df.loc[reg_df.index.isin(PROD_YEARS)]
            X_prod  = prod_df.index.values.reshape(-1, 1).astype(float)
            y_prod  = prod_df[target].values.astype(float)
            w_prod  = (prod_df["valid_px_pct"] / 100.0).values.astype(float)

            X_fore  = np.array([[float(PROD_FORECAST)]])
            fore_2027, prod_model = fit_predict(X_prod, y_prod, w_prod, X_fore)

            y_prod_pred = prod_model.predict(X_prod).flatten()
            r2_prod = r2_score_weighted(y_prod, y_prod_pred, w_prod)
            qflag_p = quality_flag(r2_prod)
            if qflag_p != "OK":
                warn_rows.append({
                    "region_id": reg, "target": target,
                    "model": "prod", "r2": round(r2_prod, 4), "flag": qflag_p,
                })

            forecast_rows.append({
                "region_id":      reg,
                "target":         target,
                "forecast_2027":  round(fore_2027, 3),
                "r2_prod":        round(r2_prod, 4),
                "slope_pct_yr":   round(float(prod_model.coef_[0]), 4),
                "quality_flag":   qflag_p,
            })

    eval_df_out = pd.DataFrame(eval_rows)
    fore_df_out = pd.DataFrame(forecast_rows)

    # ── Compute per-target aggregate MAE / RMSE ───────────────────────────────
    agg_rows = []
    for target in TARGETS:
        t_df = eval_df_out[eval_df_out["target"] == target]
        valid = t_df.dropna(subset=["actual_2025"])
        mae  = mean_absolute_error(valid["actual_2025"], valid["pred_2025"])
        rmse = np.sqrt(mean_squared_error(valid["actual_2025"], valid["pred_2025"]))
        agg_rows.append({
            "target": target,
            "MAE_across_regions": round(mae, 3),
            "RMSE_across_regions": round(rmse, 3),
        })
    agg_df = pd.DataFrame(agg_rows)

    # ── Print Table 1: Evaluation ─────────────────────────────────────────────
    SEP = "─" * 100

    print("\n" + "═" * 100)
    print("TABLE 1: EVALUATION — Train on [2019, 2021, 2023], Predict 2025 vs Actual 2025")
    print("         sample_weight = valid_px_pct / 100")
    print("═" * 100)

    for target in TARGETS:
        t_df = eval_df_out[eval_df_out["target"] == target].copy()
        print(f"\n  Target: {target}")
        print(f"  {'Region':<12} {'Pred_2025':>10} {'Actual_2025':>12} {'Error':>8} {'AbsErr':>8} {'R2_train':>9} {'Slope/yr':>9} {'Flag'}")
        print("  " + SEP[:95])
        for _, row in t_df.iterrows():
            flag_str = f"  *** {row['quality_flag']}" if row['quality_flag'] != "OK" else ""
            print(f"  {row['region_id']:<12} {row['pred_2025']:>10.3f} {row['actual_2025']:>12.3f} "
                  f"{row['error']:>8.3f} {row['abs_error']:>8.3f} "
                  f"{row['r2_train']:>9.4f} {row['slope']:>9.4f}{flag_str}")
        # Per-target aggregate
        valid = t_df.dropna(subset=["actual_2025"])
        mae  = mean_absolute_error(valid["actual_2025"], valid["pred_2025"])
        rmse = np.sqrt(mean_squared_error(valid["actual_2025"], valid["pred_2025"]))
        print("  " + "·" * 95)
        print(f"  {'AGGREGATE (8 regions)':<12}  MAE = {mae:.3f}   RMSE = {rmse:.3f}")

    print("\n" + "─" * 60)
    print("  AGGREGATE MAE / RMSE BY TARGET (across all 8 regions):")
    print(f"  {'Target':<20} {'MAE':>8} {'RMSE':>8}")
    print("  " + "─" * 38)
    for _, row in agg_df.iterrows():
        print(f"  {row['target']:<20} {row['MAE_across_regions']:>8.3f} {row['RMSE_across_regions']:>8.3f}")

    # ── Print Table 2: Forecast 2027 ──────────────────────────────────────────
    print("\n" + "═" * 100)
    print("TABLE 2: PRODUCTION FORECAST — Train on [2019, 2021, 2023, 2025], Predict 2027")
    print("         sample_weight = valid_px_pct / 100")
    print("═" * 100)

    for target in TARGETS:
        t_df = fore_df_out[fore_df_out["target"] == target].copy()
        print(f"\n  Target: {target}")
        print(f"  {'Region':<12} {'Forecast_2027':>14} {'R2_prod':>9} {'Slope/yr':>9} {'Flag'}")
        print("  " + SEP[:65])
        for _, row in t_df.iterrows():
            flag_str = f"  *** {row['quality_flag']}" if row['quality_flag'] != "OK" else ""
            print(f"  {row['region_id']:<12} {row['forecast_2027']:>14.3f} "
                  f"{row['r2_prod']:>9.4f} {row['slope_pct_yr']:>9.4f}{flag_str}")

    # ── Quality Warning Summary ───────────────────────────────────────────────
    print("\n" + "═" * 100)
    print("QUALITY WARNINGS (R² < 0.30 or negative R²)")
    print("═" * 100)
    if warn_rows:
        warn_df = pd.DataFrame(warn_rows).drop_duplicates()
        for _, w in warn_df.iterrows():
            print(f"  *** {w['flag']:15s} | {w['region_id']} | target={w['target']:<16} | "
                  f"model={w['model']} | R2={w['r2']:.4f}")
        print(f"\n  Total unique warnings: {len(warn_df)}")
        print("  Forecast for flagged region/targets should be treated as indicative only.")
    else:
        print("  No quality warnings — all regions/targets have R2 >= 0.30.")

    # ── Save Results ──────────────────────────────────────────────────────────
    # Merge eval + forecast into one wide results CSV
    eval_save = eval_df_out.copy()
    eval_save.columns = [f"eval_{c}" if c not in ("region_id","target") else c
                         for c in eval_save.columns]
    fore_save = fore_df_out.copy()
    fore_save.columns = [f"prod_{c}" if c not in ("region_id","target") else c
                         for c in fore_save.columns]
    results = pd.merge(eval_save, fore_save, on=["region_id", "target"])
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved full results to:\n  {RESULTS_CSV}")

    # ── Execution summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("DONE")
    print(f"  Models trained : {len(regions) * len(TARGETS) * 2} "
          f"(8 regions x 3 targets x 2 [eval+prod])")
    n_warn = len(pd.DataFrame(warn_rows).drop_duplicates()) if warn_rows else 0
    print(f"  Quality flags  : {n_warn}")
    print(f"  Forecast year  : {PROD_FORECAST}")
    print("=" * 80)


if __name__ == "__main__":
    main()
