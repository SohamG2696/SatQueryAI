"""
models/future_prediction/training/train_gru.py
==============================================
GRU-based land-cover forecaster with Leave-One-Region-Out (LORO) cross-validation.

Pools data across all 8 regions so the model learns shared temporal patterns,
then evaluates each region as the held-out test region once.

Architecture:
  Input : [built_up_pct, veg_pct, water_pct] for years [2019, 2021, 2023]
          → shape (seq_len=3, n_features=3), per region
  GRU   : hidden_size=16 (deliberately tiny — 8 training sequences, no room to grow)
  Output: predicted [built_up_pct, veg_pct, water_pct] for 2025

Comparison: prints GRU vs Linear Regression side-by-side using baseline_results.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
CSV_PATH     = PROJECT_ROOT / "models" / "future_prediction" / "data" / "processed" / "landcover_timeseries.csv"
BASELINE_CSV = PROJECT_ROOT / "models" / "future_prediction" / "checkpoints" / "baseline_results.csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "future_prediction" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GRU_RESULTS  = CKPT_DIR / "gru_results.csv"

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_SIZE  = 16       # small: prevents overfitting on 8 training sequences
N_LAYERS     = 1
DROPOUT      = 0.0      # no dropout — with 7 training seqs, dropout hurts more than helps
LR           = 1e-3
EPOCHS       = 800      # enough to converge on 7 sequences; monitored per fold
PATIENCE     = 150      # early stopping on training loss plateau

TARGETS       = ["built_up_pct", "vegetation_pct", "water_pct"]
INPUT_YEARS   = [2019, 2021, 2023]
TARGET_YEAR   = 2025
PROD_YEARS    = [2019, 2021, 2023, 2025]
PROD_FORECAST = 2027
R2_THRESH     = 0.30

torch.manual_seed(42)
np.random.seed(42)


# ── Model ──────────────────────────────────────────────────────────────────────

class LandCoverGRU(nn.Module):
    """
    Single-layer GRU: sequence of 3 land-cover snapshots → next-year prediction.
    Takes the last hidden state and passes it through a linear projection.
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 16,
                 n_layers: int = 1, output_size: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,  # (batch, seq, features)
            dropout=0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])   # last timestep hidden state


# ── Normalisation helpers ──────────────────────────────────────────────────────

class Normalizer:
    """Min-max normaliser fit on training data; applies inverse for output."""
    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        # X shape: (n_samples, seq_len, n_features) or (n_samples, n_features)
        flat = X.reshape(-1, X.shape[-1])
        self.min_ = flat.min(axis=0)
        self.max_ = flat.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.range_ + self.min_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ── Training ───────────────────────────────────────────────────────────────────

def train_gru(
    X_train: np.ndarray,   # (n_train, seq_len, 3)
    y_train: np.ndarray,   # (n_train, 3)
    w_train: np.ndarray,   # (n_train,)
    X_test:  np.ndarray,   # (1, seq_len, 3)
    epochs:  int = EPOCHS,
    lr:      float = LR,
    patience: int = PATIENCE,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float], LandCoverGRU]:
    """
    Train a fresh LandCoverGRU on (X_train, y_train) with sample weights,
    then predict X_test. Returns (prediction, loss_history, model).
    """
    # Normalise inputs and outputs jointly from training data
    norm_X  = Normalizer()
    norm_y  = Normalizer()
    X_tr    = norm_X.fit_transform(X_train)   # (n, seq, 3)
    y_tr    = norm_y.fit_transform(y_train)   # (n, 3)
    X_te    = norm_X.transform(X_test)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    wt = torch.tensor(w_train, dtype=torch.float32)
    Xte= torch.tensor(X_te,  dtype=torch.float32)

    model = LandCoverGRU(hidden_size=HIDDEN_SIZE, n_layers=N_LAYERS)
    opt   = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")

    loss_history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(Xt)                            # (n_train, 3)
        loss_raw = criterion(pred, yt)              # (n_train, 3)
        # Weighted mean: weight each sample, then mean across features
        weighted = (loss_raw * wt.unsqueeze(-1)).mean()
        weighted.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_val = float(weighted.item())
        loss_history.append(loss_val)

        if verbose and epoch % 200 == 0:
            print(f"    epoch {epoch:4d}  loss={loss_val:.6f}")

        # Early stopping on training loss plateau
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch} (patience={patience})")
                break

    model.eval()
    with torch.no_grad():
        pred_norm = model(Xte).numpy()              # (1, 3)
    pred_raw = norm_y.inverse_transform(pred_norm)  # back to original scale
    return pred_raw[0], loss_history, model


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(csv_path: Path):
    """
    Returns:
        regions      : list of region_id strings (8 regions, sorted)
        X_input      : (8, 3, 3) — 8 regions × 3 input years × 3 targets
        y_target     : (8, 3)    — 8 regions × 3 targets for 2025
        weights_input: (8, 3)    — valid_px_pct / 100 for each input year
        weights_2025 : (8,)      — valid_px_pct / 100 for 2025
        df           : full raw dataframe
    """
    df = pd.read_csv(csv_path)
    regions = sorted(df["region_id"].unique())

    X_input       = np.zeros((len(regions), len(INPUT_YEARS), 3), dtype=np.float32)
    y_target      = np.zeros((len(regions), 3), dtype=np.float32)
    weights_input = np.zeros((len(regions), len(INPUT_YEARS)), dtype=np.float32)
    weights_2025  = np.zeros(len(regions), dtype=np.float32)

    for ri, reg in enumerate(regions):
        rdf = df[df["region_id"] == reg].set_index("year")
        for yi, yr in enumerate(INPUT_YEARS):
            X_input[ri, yi, 0] = float(rdf.loc[yr, "built_up_pct"])
            X_input[ri, yi, 1] = float(rdf.loc[yr, "vegetation_pct"])
            X_input[ri, yi, 2] = float(rdf.loc[yr, "water_pct"])
            weights_input[ri, yi] = float(rdf.loc[yr, "valid_px_pct"]) / 100.0
        y_target[ri, 0] = float(rdf.loc[TARGET_YEAR, "built_up_pct"])
        y_target[ri, 1] = float(rdf.loc[TARGET_YEAR, "vegetation_pct"])
        y_target[ri, 2] = float(rdf.loc[TARGET_YEAR, "water_pct"])
        weights_2025[ri] = float(rdf.loc[TARGET_YEAR, "valid_px_pct"]) / 100.0

    return regions, X_input, y_target, weights_input, weights_2025, df


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def quality_flag(r2: float) -> str:
    if r2 < 0:
        return "NEGATIVE_R2"
    if r2 < R2_THRESH:
        return "LOW_R2"
    return "OK"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("SatQueryAI — GRU Land-Cover Forecaster (LORO Cross-Validation)")
    print(f"  Architecture : GRU hidden={HIDDEN_SIZE}, layers={N_LAYERS}, seq_len={len(INPUT_YEARS)}")
    print(f"  Eval scheme  : Leave-One-Region-Out (LORO), {8} folds")
    print(f"  Train years  : {INPUT_YEARS}  →  Predict year: {TARGET_YEAR}")
    print("=" * 80)

    regions, X_all, y_all, w_in_all, w_25_all, df_raw = load_data(CSV_PATH)
    n_regions = len(regions)

    # Collapse 3-timestep weights to a single representative weight per region
    # (use minimum valid_px across input years — conservative)
    w_train_region = w_in_all.min(axis=1)  # (8,)

    # ── LORO Cross-Validation ─────────────────────────────────────────────────
    print(f"\nRunning {n_regions}-fold LORO cross-validation...")
    print("(train on 7 regions, predict held-out region's 2025 value)\n")

    cv_results = []
    all_train_losses = []

    for held_out in range(n_regions):
        train_idx = [i for i in range(n_regions) if i != held_out]
        reg_name  = regions[held_out]

        X_train = X_all[train_idx]         # (7, 3, 3)
        y_train = y_all[train_idx]         # (7, 3)
        w_train = w_train_region[train_idx]# (7,)
        X_test  = X_all[[held_out]]        # (1, 3, 3)
        y_true  = y_all[held_out]          # (3,)

        t0 = time.time()
        pred, loss_hist, model = train_gru(
            X_train, y_train, w_train, X_test,
            epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
        )
        elapsed = time.time() - t0

        final_train_loss = loss_hist[-1]
        min_train_loss   = min(loss_hist)
        all_train_losses.append(final_train_loss)

        for ti, target in enumerate(TARGETS):
            abs_err = abs(float(pred[ti]) - float(y_true[ti]))
            cv_results.append({
                "region_id":       reg_name,
                "target":          target,
                "gru_pred_2025":   round(float(pred[ti]), 3),
                "actual_2025":     round(float(y_true[ti]), 3),
                "gru_error":       round(float(pred[ti]) - float(y_true[ti]), 3),
                "gru_abs_error":   round(abs_err, 3),
                "train_loss_final":round(final_train_loss, 6),
                "train_loss_min":  round(min_train_loss, 6),
                "n_epochs":        len(loss_hist),
            })

        print(f"  Fold {held_out+1}/8 | held-out={reg_name} | "
              f"train_loss={final_train_loss:.5f} | "
              f"pred_built={pred[0]:.2f}% actual={y_true[0]:.2f}% | "
              f"pred_veg={pred[1]:.2f}% actual={y_true[1]:.2f}% | "
              f"{elapsed:.1f}s")

    cv_df = pd.DataFrame(cv_results)

    # ── Load baseline for comparison ───────────────────────────────────────────
    baseline_df = pd.read_csv(BASELINE_CSV)
    # Flatten to (region_id, target) → eval_pred_2025, eval_actual_2025, eval_abs_error
    bl = baseline_df[["region_id", "target",
                       "eval_pred_2025", "eval_actual_2025", "eval_abs_error"]].copy()

    # ── Print Table 1: GRU Evaluation by Target ────────────────────────────────
    SEP = "─" * 110

    print("\n\n" + "═" * 110)
    print("TABLE 1: GRU EVALUATION — LORO Cross-Validation (Train 7, Predict held-out 2025)")
    print("=" * 110)

    warn_rows = []
    for target in TARGETS:
        t_df = cv_df[cv_df["target"] == target].copy()
        bl_t = bl[bl["target"] == target].set_index("region_id")

        print(f"\n  Target: {target}")
        print(f"  {'Region':<12} {'GRU_Pred':>9} {'Actual':>9} {'GRU_Err':>9} "
              f"{'GRU_AbsErr':>11} {'LR_AbsErr':>11} {'Delta':>8}  Note")
        print("  " + SEP[:105])

        gru_abs = []
        lr_abs  = []
        for _, row in t_df.iterrows():
            reg = row["region_id"]
            lr_abserr = float(bl_t.loc[reg, "eval_abs_error"]) if reg in bl_t.index else float("nan")
            delta = row["gru_abs_error"] - lr_abserr
            sign  = "▲WORSE" if delta > 0.5 else ("▼BETTER" if delta < -0.5 else "~SAME")
            gru_abs.append(row["gru_abs_error"])
            lr_abs.append(lr_abserr)
            print(f"  {reg:<12} {row['gru_pred_2025']:>9.3f} {row['actual_2025']:>9.3f} "
                  f"{row['gru_error']:>9.3f} {row['gru_abs_error']:>11.3f} "
                  f"{lr_abserr:>11.3f} {delta:>+8.3f}  {sign}")

        gru_mae  = np.mean(gru_abs)
        gru_rmse = np.sqrt(np.mean(np.array(gru_abs) ** 2))
        lr_mae   = np.nanmean(lr_abs)
        lr_rmse  = np.sqrt(np.nanmean(np.array(lr_abs) ** 2))
        mae_win  = "GRU" if gru_mae < lr_mae else "LinearReg"
        print("  " + "·" * 105)
        print(f"  {'GRU AGGREGATE':<12}  MAE={gru_mae:.3f}  RMSE={gru_rmse:.3f}"
              f"   vs   LR:  MAE={lr_mae:.3f}  RMSE={lr_rmse:.3f}"
              f"   →  Winner: {mae_win}")

        # R² across 8 LORO predictions vs actuals (cross-region R²)
        gru_preds_arr = cv_df[cv_df["target"] == target]["gru_pred_2025"].values.astype(float)
        actuals_arr   = cv_df[cv_df["target"] == target]["actual_2025"].values.astype(float)
        r2 = r2_score(actuals_arr, gru_preds_arr)
        qf = quality_flag(r2)
        print(f"  Cross-region R² (GRU): {r2:.4f}  [{qf}]")
        if qf != "OK":
            warn_rows.append({"target": target, "r2": round(r2, 4), "flag": qf})

    # ── Overfitting diagnostic ─────────────────────────────────────────────────
    mean_final_loss = np.mean(all_train_losses)
    min_final_loss  = min(all_train_losses)
    max_final_loss  = max(all_train_losses)

    print("\n" + "─" * 60)
    print("OVERFITTING DIAGNOSTIC")
    print("─" * 60)
    print(f"  Final training loss stats across 8 LORO folds:")
    print(f"    Mean: {mean_final_loss:.5f}  Min: {min_final_loss:.5f}  Max: {max_final_loss:.5f}")
    print()

    # Compute CV MAE for built_up_pct as the primary indicator
    cv_mae_built = cv_df[cv_df["target"] == "built_up_pct"]["gru_abs_error"].mean()
    lr_mae_built = bl[bl["target"] == "built_up_pct"]["eval_abs_error"].mean()

    if mean_final_loss < 1e-4 and cv_mae_built > lr_mae_built * 1.5:
        overfit_verdict = ("LIKELY OVERFITTING — training loss is near-zero but CV error is "
                           "significantly worse than linear regression.")
    elif mean_final_loss < 5e-4 and cv_mae_built > 4.0:
        overfit_verdict = ("POSSIBLE OVERFITTING — low training loss but high CV MAE. "
                           "With only 7 training sequences, the GRU may be memorising patterns.")
    elif cv_mae_built < lr_mae_built:
        overfit_verdict = (f"NO CLEAR OVERFITTING — GRU CV MAE ({cv_mae_built:.3f}%) < "
                           f"LR MAE ({lr_mae_built:.3f}%). Pooling helps.")
    else:
        overfit_verdict = (f"MILD OVERFITTING RISK — GRU CV MAE ({cv_mae_built:.3f}%) >= "
                           f"LR MAE ({lr_mae_built:.3f}%). 7 training sequences is very limited.")

    print(f"  Verdict: {overfit_verdict}")
    print()
    print("  Interpretation guide:")
    print("    - Training loss near-zero + high CV error = classic overfit")
    print("    - Training loss moderate + CV error < LR = GRU generalising well")
    print(f"    - With n=7 training sequences, treat all GRU results with caution.")

    # ── Table 2: Head-to-head comparison ─────────────────────────────────────
    print("\n" + "═" * 110)
    print("TABLE 2: HEAD-TO-HEAD COMPARISON — GRU (LORO) vs Linear Regression")
    print("  built_up_pct only (the variable with highest LR error)")
    print("=" * 110)

    t_gru = cv_df[cv_df["target"] == "built_up_pct"].set_index("region_id")
    t_lr  = bl[bl["target"] == "built_up_pct"].set_index("region_id")

    print(f"\n  {'Region':<12} {'Actual':>8} {'GRU Pred':>10} {'GRU Err':>9} "
          f"{'LR Pred':>9} {'LR Err':>9}  Winner")
    print("  " + "─" * 80)
    for reg in regions:
        gru_pred = t_gru.loc[reg, "gru_pred_2025"]
        gru_err  = t_gru.loc[reg, "gru_error"]
        gru_ae   = t_gru.loc[reg, "gru_abs_error"]
        actual   = t_gru.loc[reg, "actual_2025"]
        lr_pred  = float(t_lr.loc[reg, "eval_pred_2025"])
        lr_ae    = float(t_lr.loc[reg, "eval_abs_error"])
        lr_err   = float(lr_pred - actual)
        winner   = "GRU" if gru_ae < lr_ae else ("LR" if lr_ae < gru_ae else "TIE")
        print(f"  {reg:<12} {actual:>8.3f} {gru_pred:>10.3f} {gru_err:>+9.3f} "
              f"{lr_pred:>9.3f} {lr_err:>+9.3f}  {winner}")

    gru_mae_b  = t_gru["gru_abs_error"].mean()
    gru_rmse_b = np.sqrt((t_gru["gru_abs_error"] ** 2).mean())
    lr_mae_b   = t_lr["eval_abs_error"].mean()
    lr_rmse_b  = np.sqrt((t_lr["eval_abs_error"] ** 2).mean())
    overall_winner = "GRU" if gru_mae_b < lr_mae_b else "Linear Regression"
    print("  " + "─" * 80)
    print(f"  {'AGGREGATE':<12}          {'GRU':>10}  MAE={gru_mae_b:.3f} RMSE={gru_rmse_b:.3f}"
          f"  {'LR':>9}  MAE={lr_mae_b:.3f} RMSE={lr_rmse_b:.3f}  → {overall_winner} wins")

    # ── Production forecast (all 8 regions, predict 2027) ────────────────────
    print("\n" + "═" * 100)
    print("TABLE 3: PRODUCTION FORECAST 2027 — GRU trained on ALL 8 regions × 4 years")
    print("  (Train: [2019,2021,2023,2025], Forecast: 2027)")
    print("=" * 100)

    # Build production inputs: [2021, 2023, 2025] as input (most recent 3 years)
    prod_input_years = [2021, 2023, 2025]
    X_prod = np.zeros((n_regions, 3, 3), dtype=np.float32)
    y_prod = np.zeros((n_regions, 3), dtype=np.float32)  # 2025 as "target" for training
    w_prod = np.zeros(n_regions, dtype=np.float32)

    for ri, reg in enumerate(regions):
        rdf = df_raw[df_raw["region_id"] == reg].set_index("year")
        for yi, yr in enumerate(prod_input_years):
            X_prod[ri, yi, 0] = float(rdf.loc[yr, "built_up_pct"])
            X_prod[ri, yi, 1] = float(rdf.loc[yr, "vegetation_pct"])
            X_prod[ri, yi, 2] = float(rdf.loc[yr, "water_pct"])
        y_prod[ri, 0] = float(rdf.loc[TARGET_YEAR, "built_up_pct"])
        y_prod[ri, 1] = float(rdf.loc[TARGET_YEAR, "vegetation_pct"])
        y_prod[ri, 2] = float(rdf.loc[TARGET_YEAR, "water_pct"])
        w_prod[ri] = float(rdf.loc[TARGET_YEAR, "valid_px_pct"]) / 100.0

    # Train one final model on all 8 regions
    # Use [2019,2021,2023] → predict 2025 for training (same as eval)
    # Then shift: use [2021,2023,2025] → predict 2027 for production
    # But since we only have 8 samples, train on all 8 with the shifted window
    print("\n  [Training production model on all 8 regions (2021/2023/2025 → 2027 forecast)...]")
    # For production model, X = [2021,2023,2025] as input, y = [2025] as proxy target
    # This isn't ideal but with 8 samples we use what we have
    # Better: train on BOTH windows [2019→2025 and 2021→?] — but we only have truth at 2025
    # So: train same model as eval (on [2019,2021,2023]→2025, all 8 regions),
    #     then feed [2021,2023,2025] as the input sequence for 2027 prediction
    X_eval_all = X_all          # (8, 3, 3) — [2019,2021,2023]
    y_eval_all = y_all          # (8, 3)    — 2025

    # Dummy X_test is just a placeholder; we'll re-feed X_prod for the actual prediction
    X_dummy = X_eval_all[[0]]
    _, _, prod_model_obj = train_gru(
        X_eval_all, y_eval_all, w_train_region, X_dummy,
        epochs=EPOCHS, lr=LR, patience=PATIENCE, verbose=False,
    )

    # Now normalise X_prod with the same stats and get 2027 forecasts
    norm_X_prod = Normalizer()
    norm_y_prod = Normalizer()
    norm_X_prod.fit(X_eval_all)   # fit on training distribution
    norm_y_prod.fit(y_eval_all)

    X_prod_norm = norm_X_prod.transform(X_prod)
    Xp_t = torch.tensor(X_prod_norm, dtype=torch.float32)
    prod_model_obj.eval()
    with torch.no_grad():
        fore_norm = prod_model_obj(Xp_t).numpy()  # (8, 3)
    fore_raw = norm_y_prod.inverse_transform(fore_norm)

    fore_rows = []
    print(f"\n  {'Region':<12} {'built_up_2027':>14} {'veg_2027':>10} {'water_2027':>11}")
    print("  " + "─" * 55)
    for ri, reg in enumerate(regions):
        b27, v27, w27 = fore_raw[ri]
        b27 = max(0.0, min(100.0, b27))
        v27 = max(0.0, min(100.0, v27))
        w27 = max(0.0, min(100.0, w27))
        print(f"  {reg:<12} {b27:>14.3f} {v27:>10.3f} {w27:>11.3f}")
        fore_rows.append({
            "region_id": reg,
            "gru_forecast_built_2027": round(b27, 3),
            "gru_forecast_veg_2027":   round(v27, 3),
            "gru_forecast_water_2027": round(w27, 3),
        })

    # ── Save Results ───────────────────────────────────────────────────────────
    save_df = cv_df.copy()
    save_df.to_csv(GRU_RESULTS, index=False)
    print(f"\nSaved LORO CV results to:\n  {GRU_RESULTS}")

    fore_df = pd.DataFrame(fore_rows)
    fore_csv = CKPT_DIR / "gru_forecast_2027.csv"
    fore_df.to_csv(fore_csv, index=False)
    print(f"Saved 2027 forecasts to:\n  {fore_csv}")

    # ── Quality Warnings ───────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("QUALITY FLAGS")
    print("=" * 80)
    if warn_rows:
        for w in warn_rows:
            print(f"  *** {w['flag']:<15} | target={w['target']:<20} | cross-region R²={w['r2']:.4f}")
    else:
        print("  No cross-region R² warnings (all targets R² >= 0.30).")

    print(f"\n  Note: with only 7 training sequences per LORO fold, all results")
    print(f"  should be treated as directional indicators, not precise forecasts.")
    print(f"  A larger dataset (more regions or more years) is needed for reliable GRU training.")

    print("\n" + "=" * 80)
    print("DONE")
    print(f"  GRU built_up MAE : {gru_mae_b:.3f}%   LR built_up MAE : {lr_mae_b:.3f}%")
    print(f"  GRU built_up RMSE: {gru_rmse_b:.3f}%   LR built_up RMSE: {lr_rmse_b:.3f}%")
    print(f"  Overall winner   : {overall_winner}")
    print("=" * 80)


if __name__ == "__main__":
    main()
