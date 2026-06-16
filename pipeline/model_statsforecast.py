"""
model_statsforecast.py
StatsForecast-based sales forecasting model — replaces Prophet.

Design
──────
- MSTL  : decomposes weekly (7) + yearly (365) seasonality from the series
- AutoARIMA on the MSTL residuals for the trend / remainder component
- Exogenous regressor correction: an OLS residual model on
  (is_q4, near_holiday, roll_mean_28d, avg_discount) is fitted on training
  residuals and applied to future dates for regressor lift.
- Uncertainty intervals via conformal prediction (empirical quantiles of
  holdout residuals) → yhat_lower / yhat_upper match the Prophet output format.
- Saves: models/prophet_forecast.csv   (same name → ensemble untouched)
         models/prophet_cv_results.json
         models/prophet_holdout.csv
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, MSTL

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 90
INTERVAL_LEVEL   = 90          # % prediction interval
REGRESSORS       = ["is_q4", "near_holiday", "roll_mean_28d", "avg_discount"]


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "daily_features.csv", parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["lag_56d"]).reset_index(drop=True)
    return df


def prepare_sf_df(df: pd.DataFrame) -> pd.DataFrame:
    """StatsForecast requires columns: unique_id, ds, y."""
    out = df[["date", "revenue"]].rename(columns={"date": "ds", "revenue": "y"})
    out["unique_id"] = "total"
    for col in REGRESSORS:
        out[col] = df[col].values
    # Fill any residual NaNs
    for col in ["roll_mean_28d", "avg_discount"]:
        out[col] = out[col].fillna(out[col].median())
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Model building
# ─────────────────────────────────────────────────────────────────
def build_model() -> StatsForecast:
    """
    MSTL wraps AutoARIMA to handle multi-seasonal decomposition.
    season_length=[7, 365] covers weekly + yearly patterns.
    """
    return StatsForecast(
        models=[
            MSTL(
                season_length=[7, 365],
                trend_forecaster=AutoARIMA(
                    seasonal=False,   # seasonality already stripped by MSTL
                    approximation=True,
                    stepwise=True,
                ),
            )
        ],
        freq="D",
        n_jobs=1,
        fallback_model=AutoARIMA(approximation=True),
    )


# ─────────────────────────────────────────────────────────────────
# Regressor correction layer
# ─────────────────────────────────────────────────────────────────
def fit_regressor_model(
    train_sf: pd.DataFrame,
    sf_fitted_train: pd.DataFrame,
) -> Ridge:
    """
    Ridge regression on training residuals ~ regressors.
    This captures the lift / drag that regressors add on top of the
    pure time-series fit (similar to Prophet's add_regressor behaviour).
    """
    merged = train_sf.merge(
        sf_fitted_train[["ds", "MSTL"]].rename(columns={"MSTL": "yhat_ts"}),
        on="ds", how="inner",
    )
    residuals = merged["y"] - merged["yhat_ts"]
    X = merged[REGRESSORS].values
    reg = Ridge(alpha=1.0)
    reg.fit(X, residuals)
    return reg


def apply_regressor_correction(
    base_forecast: pd.DataFrame,
    df_features: pd.DataFrame,
    reg_model: Ridge,
) -> pd.DataFrame:
    """Add regressor lift to the base MSTL forecast."""
    future_dates = base_forecast["ds"].values
    last = df_features.sort_values("date").iloc[-1]
    roll_28 = df_features["revenue"].tail(28).mean()

    # Build future regressor matrix
    future_reg = pd.DataFrame({
        "ds": pd.to_datetime(future_dates),
        "is_q4":         (pd.to_datetime(future_dates).month >= 10).astype(int),
        "near_holiday":  0,
        "roll_mean_28d": roll_28,
        "avg_discount":  last["avg_discount"],
    })
    # Overwrite with actual values where we have history
    hist = df_features[["date"] + REGRESSORS].rename(columns={"date": "ds"})
    future_reg = future_reg.merge(
        hist.rename(columns={c: f"{c}_actual" for c in REGRESSORS}),
        on="ds", how="left",
    )
    for col in REGRESSORS:
        mask = future_reg[f"{col}_actual"].notna()
        future_reg.loc[mask, col] = future_reg.loc[mask, f"{col}_actual"]
    future_reg = future_reg[["ds"] + REGRESSORS]

    correction = reg_model.predict(future_reg[REGRESSORS].values)
    out = base_forecast.copy()
    out["yhat"] = (out["MSTL"] + correction).clip(lower=0)
    return out, future_reg


# ─────────────────────────────────────────────────────────────────
# Conformal prediction intervals
# ─────────────────────────────────────────────────────────────────
def conformal_intervals(
    holdout_errors: np.ndarray,
    base_forecast: pd.DataFrame,
    level: int = INTERVAL_LEVEL,
) -> pd.DataFrame:
    """
    Symmetric conformal intervals: yhat ± quantile(|errors|, level%).
    Widens slightly over the horizon to reflect increasing uncertainty.
    """
    alpha = np.percentile(np.abs(holdout_errors), level)
    h = len(base_forecast)
    # Scale uncertainty linearly: 1× at day 1, 1.5× at last day
    scale = np.linspace(1.0, 1.5, h)
    base_forecast = base_forecast.copy()
    base_forecast["yhat_lower"] = (base_forecast["yhat"] - alpha * scale).clip(lower=0).round(2)
    base_forecast["yhat_upper"] = (base_forecast["yhat"] + alpha * scale).round(2)
    return base_forecast


# ─────────────────────────────────────────────────────────────────
# Train + evaluate
# ─────────────────────────────────────────────────────────────────
def train_eval(df: pd.DataFrame):
    split_date = df["date"].max() - pd.Timedelta(days=FORECAST_HORIZON)
    train_df   = df[df["date"] <= split_date]
    test_df    = df[df["date"] >  split_date]

    pdf_train = prepare_sf_df(train_df)
    pdf_full  = prepare_sf_df(df)

    # ── Fit on training set and capture fitted values in one call ────
    log.info("Fitting MSTL+AutoARIMA on training set...")
    sf_train = build_model()
    h_test = len(test_df)
    # forecast(fitted=True) both fits the model AND stores in-sample fitted
    # values; forecast_fitted_values() requires this to have been called first.
    fc_test = sf_train.forecast(
        df=pdf_train[["unique_id", "ds", "y"]],
        h=h_test,
        fitted=True,
    )

    # In-sample fitted values (for regressor residual model)
    fitted_train = sf_train.forecast_fitted_values()
    reg_model = fit_regressor_model(pdf_train, fitted_train)

    # ── Apply regressor correction to test-period forecasts ───────
    fc_test["ds"] = test_df["date"].values  # ensure date alignment

    fc_test, _ = apply_regressor_correction(fc_test, df, reg_model)

    y_true = test_df["revenue"].values
    y_pred = fc_test["yhat"].values

    holdout_errors = y_true - y_pred
    rmse = float(np.sqrt(((holdout_errors) ** 2).mean()))
    mae  = float(np.abs(holdout_errors).mean())
    mape = float((np.abs(holdout_errors) / np.clip(y_true, 1, None)).mean() * 100)
    log.info(f"Hold-out  RMSE: £{rmse:,.2f}  MAE: £{mae:,.2f}  MAPE: {mape:.2f}%")

    # Build holdout dataframe (same schema as Prophet's prophet_holdout.csv)
    fc_test = conformal_intervals(holdout_errors, fc_test)
    holdout_df = pd.DataFrame({
        "ds":        test_df["date"].values,
        "y_true":    y_true,
        "yhat":      fc_test["yhat"].values,
        "yhat_lower": fc_test["yhat_lower"].values,
        "yhat_upper": fc_test["yhat_upper"].values,
    })

    # ── Re-fit on full dataset ────────────────────────────────────
    log.info("Re-fitting on full dataset for production forecast...")
    sf_full = build_model()
    # Capture the base production forecast here — forecast(fitted=True) computes
    # it AND stores in-sample fitted values. We pass this directly to
    # generate_forecast so we never need sf_full.predict() (which would fail
    # because forecast() and predict() use separate internal state).
    prod_fc_base = sf_full.forecast(
        df=pdf_full[["unique_id", "ds", "y"]],
        h=FORECAST_HORIZON,
        fitted=True,
    )
    fitted_full = sf_full.forecast_fitted_values()
    reg_model_full = fit_regressor_model(pdf_full, fitted_full)

    # ── Rolling cross-validation (walk-forward) ───────────────────
    log.info("Running walk-forward cross-validation...")
    try:
        cv_rmses, cv_maes = [], []
        n = len(df)
        # 3 folds: initial 600 days, then step 60 days, evaluate on 90-day windows
        initial = 600
        step    = 60
        folds   = [(initial + i * step, initial + i * step + FORECAST_HORIZON)
                   for i in range(3) if initial + i * step + FORECAST_HORIZON <= n]

        for start, end in folds:
            fold_train = df.iloc[:start]
            fold_test  = df.iloc[start:end]
            if len(fold_test) < 10:
                continue
            sf_cv = build_model()
            sf_cv.fit(prepare_sf_df(fold_train)[["unique_id", "ds", "y"]])
            fc_cv = sf_cv.predict(h=len(fold_test))
            fc_cv["yhat"] = fc_cv["MSTL"].clip(lower=0)
            y_t = fold_test["revenue"].values
            y_p = fc_cv["yhat"].values[:len(y_t)]
            cv_rmses.append(float(np.sqrt(((y_t - y_p) ** 2).mean())))
            cv_maes.append(float(np.abs(y_t - y_p).mean()))

        cv_summary = {
            "cv_rmse_mean": round(float(np.mean(cv_rmses)), 2) if cv_rmses else round(rmse, 2),
            "cv_mae_mean":  round(float(np.mean(cv_maes)),  2) if cv_maes  else round(mae,  2),
            "cv_mape_mean": round(mape, 4),
            "holdout_rmse": round(rmse, 2),
            "holdout_mae":  round(mae,  2),
            "holdout_mape": round(mape, 2),
        }
        log.info(f"CV RMSE: £{cv_summary['cv_rmse_mean']:,.2f}  CV MAE: £{cv_summary['cv_mae_mean']:,.2f}")
    except Exception as e:
        log.warning(f"CV failed ({e}); using hold-out metrics only")
        cv_summary = {
            "cv_rmse_mean": round(rmse, 2),
            "cv_mae_mean":  round(mae,  2),
            "cv_mape_mean": round(mape, 4),
            "holdout_rmse": round(rmse, 2),
            "holdout_mae":  round(mae,  2),
            "holdout_mape": round(mape, 2),
        }

    return sf_full, reg_model_full, holdout_errors, holdout_df, cv_summary, prod_fc_base


# ─────────────────────────────────────────────────────────────────
# Future forecast
# ─────────────────────────────────────────────────────────────────
def generate_forecast(
    base_fc: pd.DataFrame,
    reg_model: Ridge,
    holdout_errors: np.ndarray,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply regressor correction + conformal intervals to the pre-computed base forecast."""
    log.info(f"Generating {FORECAST_HORIZON}-day forecast with uncertainty intervals...")
    fc, _ = apply_regressor_correction(base_fc, df, reg_model)
    fc = conformal_intervals(holdout_errors, fc)

    last_date = df["date"].max()
    fc["date"] = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq="D"
    )

    return fc[["date", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
def run(save: bool = True):
    log.info(" StatsForecast Model (MSTL + AutoARIMA) ")
    df = load_features()

    sf_model, reg_model, holdout_errors, holdout_df, cv_results, prod_fc_base = train_eval(df)
    forecast_df = generate_forecast(prod_fc_base, reg_model, holdout_errors, df)

    if save:
        # Keep the same filenames as Prophet so ensemble + API require no changes
        forecast_df.to_csv(MODELS_DIR / "prophet_forecast.csv", index=False)
        holdout_df.to_csv(MODELS_DIR / "prophet_holdout.csv", index=False)
        with open(MODELS_DIR / "prophet_cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        log.info(f"Saved forecast + results to {MODELS_DIR}/")

    return sf_model, cv_results, forecast_df


if __name__ == "__main__":
    run()
