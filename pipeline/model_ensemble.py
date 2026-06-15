"""
model_ensemble.py
Weighted ensemble of XGBoost + Prophet forecasts.

Strategy
────────
- Weight each model inversely by its holdout RMSE
  (lower RMSE → higher weight)
- Combine yhat; propagate uncertainty bands from Prophet
- Produce a single model_comparison.json report for the dashboard
- Saves: models/ensemble_forecast.csv, models/model_comparison.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"


def rmse_weight(rmse_a: float, rmse_b: float):
    """Inverse-RMSE weights — better model gets higher weight."""
    inv_a = 1 / rmse_a
    inv_b = 1 / rmse_b
    total = inv_a + inv_b
    return inv_a / total, inv_b / total


def load_forecasts():
    xgb   = pd.read_csv(MODELS_DIR / "xgb_forecast.csv",    parse_dates=["date"])
    proph = pd.read_csv(MODELS_DIR / "prophet_forecast.csv", parse_dates=["date"])
    return xgb, proph


def load_cv_results():
    with open(MODELS_DIR / "xgb_cv_results.json") as f:
        xgb_cv = json.load(f)
    with open(MODELS_DIR / "prophet_cv_results.json") as f:
        proph_cv = json.load(f)
    return xgb_cv, proph_cv


def load_holdouts():
    xgb_h = pd.read_csv(MODELS_DIR / "xgb_holdout.csv",    parse_dates=["date"])
    proph_h = pd.read_csv(MODELS_DIR / "prophet_holdout.csv", parse_dates=["ds"])
    proph_h = proph_h.rename(columns={"ds": "date", "y_true": "revenue"})
    return xgb_h, proph_h


def run(save: bool = True):
    log.info("═══ Ensemble + Comparison ════════════════")

    xgb_fc, proph_fc = load_forecasts()
    xgb_cv, proph_cv = load_cv_results()
    xgb_h,  proph_h  = load_holdouts()

    # ── Weights based on holdout RMSE ────────────────────────────
    rmse_xgb   = xgb_cv["holdout_rmse"]
    rmse_proph = proph_cv["holdout_rmse"]
    w_xgb, w_proph = rmse_weight(rmse_xgb, rmse_proph)
    log.info(f"Weights  XGBoost: {w_xgb:.3f}  Prophet: {w_proph:.3f}")

    # ── Merge forecasts on date ───────────────────────────────────
    merged = xgb_fc.rename(columns={"yhat": "yhat_xgb"}).merge(
        proph_fc[["date","yhat","yhat_lower","yhat_upper"]].rename(
            columns={"yhat": "yhat_proph"}
        ),
        on="date", how="inner"
    )

    merged["yhat_ensemble"] = (
        w_xgb * merged["yhat_xgb"] + w_proph * merged["yhat_proph"]
    ).round(2)

    # Widen Prophet's uncertainty band slightly for ensemble
    merged["ensemble_lower"] = (merged["yhat_lower"] * 0.95).clip(lower=0).round(2)
    merged["ensemble_upper"] = (merged["yhat_upper"] * 1.05).round(2)

    log.info(f"Ensemble forecast: {len(merged)} days")
    log.info(f"  Mean: £{merged['yhat_ensemble'].mean():,.2f}")
    log.info(f"  Range: £{merged['yhat_ensemble'].min():,.2f} – £{merged['yhat_ensemble'].max():,.2f}")

    # ── Ensemble holdout metrics ──────────────────────────────────
    # Align holdout dates
    xgb_h_sub = xgb_h[["date", TARGET := "revenue", "yhat"]].rename(
        columns={"yhat": "yhat_xgb", "revenue": "y_true"}
    )
    proph_dates = set(proph_h["date"])
    xgb_aligned = xgb_h_sub[xgb_h_sub["date"].isin(proph_dates)]
    proph_aligned = proph_h[proph_h["date"].isin(set(xgb_h_sub["date"]))].rename(
        columns={"revenue": "y_true", "yhat": "yhat_proph"}
    )[["date","y_true","yhat_proph"]]

    ens_h = xgb_aligned.merge(proph_aligned[["date","yhat_proph"]], on="date", how="inner")
    ens_h["yhat_ens"] = w_xgb * ens_h["yhat_xgb"] + w_proph * ens_h["yhat_proph"]

    ens_rmse = float(np.sqrt(mean_squared_error(ens_h["y_true"], ens_h["yhat_ens"])))
    ens_mae  = float(mean_absolute_error(ens_h["y_true"], ens_h["yhat_ens"]))
    ens_mape = float((ens_h["y_true"] - ens_h["yhat_ens"]).abs().div(
        ens_h["y_true"].clip(lower=1)).mean() * 100)

    # ── Model comparison report ───────────────────────────────────
    comparison = {
        "models": {
            "xgboost": {
                "holdout_rmse":  xgb_cv["holdout_rmse"],
                "holdout_mae":   xgb_cv["holdout_mae"],
                "holdout_mape":  xgb_cv.get("holdout_mape", None),
                "cv_rmse_mean":  xgb_cv["cv_rmse_mean"],
                "cv_rmse_std":   xgb_cv["cv_rmse_std"],
                "cv_mae_mean":   xgb_cv["cv_mae_mean"],
                "weight":        round(w_xgb, 4),
                "top_features":  list(xgb_cv.get("feature_importance", {}).keys())[:10],
            },
            "prophet": {
                "holdout_rmse":  proph_cv["holdout_rmse"],
                "holdout_mae":   proph_cv["holdout_mae"],
                "holdout_mape":  proph_cv.get("holdout_mape", None),
                "cv_rmse_mean":  proph_cv.get("cv_rmse_mean", None),
                "cv_mae_mean":   proph_cv.get("cv_mae_mean",  None),
                "weight":        round(w_proph, 4),
            },
            "ensemble": {
                "holdout_rmse": round(ens_rmse, 2),
                "holdout_mae":  round(ens_mae,  2),
                "holdout_mape": round(ens_mape, 2),
                "strategy":     "inverse-RMSE weighted average",
            },
        },
        "winner": "ensemble" if ens_rmse < min(rmse_xgb, rmse_proph) else (
            "xgboost" if rmse_xgb < rmse_proph else "prophet"
        ),
        "forecast_horizon_days": 90,
    }

    log.info(f"\n{'─'*48}")
    log.info(f"  {'Model':<12} {'RMSE':>10}  {'MAE':>10}  {'MAPE':>8}")
    log.info(f"{'─'*48}")
    for name, m in comparison["models"].items():
        rmse = m['holdout_rmse']
        mae  = m['holdout_mae']
        mape = m.get('holdout_mape') or 0
        log.info(f"  {name:<12} £{rmse:>9,.2f}  £{mae:>9,.2f}  {mape:>7.2f}%")
    log.info(f"{'─'*48}")
    log.info(f"  Winner: {comparison['winner'].upper()}")

    if save:
        merged.to_csv(MODELS_DIR / "ensemble_forecast.csv", index=False)
        with open(MODELS_DIR / "model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        log.info(f"Saved ensemble forecast + comparison report to {MODELS_DIR}/")

    return merged, comparison
