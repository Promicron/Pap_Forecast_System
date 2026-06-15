"""
model_prophet.py
Facebook Prophet sales forecasting model.

Design
──────
- Prophet handles trend + weekly + yearly seasonality natively
- Additional regressors: is_q4, near_holiday, roll_mean_28d, avg_discount
- Prophet's built-in cross_validation for RMSE/MAE/MAPE
- Uncertainty intervals (yhat_lower, yhat_upper) for the dashboard band
- Saves: models/prophet_model.json, models/prophet_cv_results.json, models/prophet_forecast.csv
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 90


# ─────────────────────────────────────────────
# US public holidays (Superstore is US retail)
# ─────────────────────────────────────────────
def make_holidays() -> pd.DataFrame:
    years = [2023, 2024, 2025, 2026]
    rows  = []
    for y in years:
        rows += [
            {"holiday": "New Year",        "ds": f"{y}-01-01"},
            {"holiday": "MLK Day",         "ds": f"{y}-01-15"},  # approx
            {"holiday": "Presidents Day",  "ds": f"{y}-02-19"},  # approx
            {"holiday": "Memorial Day",    "ds": f"{y}-05-27"},  # approx
            {"holiday": "Independence Day","ds": f"{y}-07-04"},
            {"holiday": "Labor Day",       "ds": f"{y}-09-02"},  # approx
            {"holiday": "Thanksgiving",    "ds": f"{y}-11-28"},  # approx
            {"holiday": "Black Friday",    "ds": f"{y}-11-29"},
            {"holiday": "Cyber Monday",    "ds": f"{y}-12-02"},
            {"holiday": "Christmas",       "ds": f"{y}-12-25"},
        ]
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    df["lower_window"] = -1
    df["upper_window"] =  1
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "daily_features.csv", parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=["lag_56d"]).reset_index(drop=True)
    return df


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prophet requires columns: ds, y — plus any extra regressors."""
    pdf = df[["date", "revenue", "is_q4", "near_holiday",
              "roll_mean_28d", "avg_discount"]].rename(
        columns={"date": "ds", "revenue": "y"}
    )
    # Fill any residual NaNs in regressors with their column median
    for col in ["roll_mean_28d", "avg_discount"]:
        pdf[col] = pdf[col].fillna(pdf[col].median())
    return pdf


def build_model() -> Prophet:
    return Prophet(
        yearly_seasonality  = True,
        weekly_seasonality  = True,
        daily_seasonality   = False,
        seasonality_mode    = "multiplicative",  # fits retail data better
        changepoint_prior_scale = 0.15,           # flexibility of trend
        seasonality_prior_scale = 10.0,
        holidays            = make_holidays(),
        interval_width      = 0.90,               # 90% uncertainty interval
    )


def train_eval(df: pd.DataFrame):
    split_date = df["date"].max() - pd.Timedelta(days=FORECAST_HORIZON)
    train_df   = df[df["date"] <= split_date]
    test_df    = df[df["date"] >  split_date]

    pdf_train = prepare_prophet_df(train_df)
    pdf_full  = prepare_prophet_df(df)

    m = build_model()
    for reg in ["is_q4", "near_holiday", "roll_mean_28d", "avg_discount"]:
        m.add_regressor(reg, standardize=True)

    log.info("Fitting Prophet on training set...")
    m.fit(pdf_train)

    # Predict on test period using actual regressor values
    pdf_test = prepare_prophet_df(test_df).rename(columns={"y": "y_true"})
    forecast_test = m.predict(pdf_test.drop(columns=["y_true"]))
    forecast_test = forecast_test.merge(
        pdf_test[["ds", "y_true"]], on="ds", how="left"
    )

    y_true = forecast_test["y_true"]
    y_pred = forecast_test["yhat"].clip(lower=0)

    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae  = float((y_true - y_pred).abs().mean())
    mape = float(((y_true - y_pred).abs() / y_true.clip(lower=1)).mean() * 100)
    log.info(f"Hold-out  RMSE: £{rmse:,.2f}  MAE: £{mae:,.2f}  MAPE: {mape:.2f}%")

    # Re-fit on full dataset for final forecast
    log.info("Re-fitting Prophet on full dataset for production forecast...")
    m_full = build_model()
    for reg in ["is_q4", "near_holiday", "roll_mean_28d", "avg_discount"]:
        m_full.add_regressor(reg, standardize=True)
    m_full.fit(pdf_full)

    # Prophet cross-validation
    log.info("Running Prophet built-in cross-validation (this may take ~30s)...")
    try:
        df_cv = cross_validation(
            m_full,
            initial  = "600 days",
            period   = "60 days",
            horizon  = "90 days",
            parallel = None,
        )
        pm = performance_metrics(df_cv)
        cv_summary = {
            "cv_rmse_mean": round(pm["rmse"].mean(), 2),
            "cv_mae_mean":  round(pm["mae"].mean(),  2),
            "cv_mape_mean": round(pm["mape"].mean() * 100, 4),
            "holdout_rmse": round(rmse, 2),
            "holdout_mae":  round(mae,  2),
            "holdout_mape": round(mape, 2),
        }
        log.info(f"CV RMSE: £{cv_summary['cv_rmse_mean']:,.2f}  CV MAE: £{cv_summary['cv_mae_mean']:,.2f}")
    except Exception as e:
        log.warning(f"Prophet CV failed ({e}); using hold-out only")
        cv_summary = {
            "cv_rmse_mean": round(rmse, 2),
            "cv_mae_mean":  round(mae,  2),
            "holdout_rmse": round(rmse, 2),
            "holdout_mae":  round(mae,  2),
            "holdout_mape": round(mape, 2),
        }

    return m_full, forecast_test, cv_summary


def generate_forecast(model: Prophet, df: pd.DataFrame) -> pd.DataFrame:
    """Make future dataframe and populate regressors for the forecast horizon."""
    future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq="D")

    # Fill regressor values for future dates
    last = df.sort_values("date").iloc[-1]
    roll_28 = df["revenue"].tail(28).mean()

    future["is_q4"]         = (pd.to_datetime(future["ds"]).dt.month >= 10).astype(int)
    future["near_holiday"]  = 0  # conservative
    future["roll_mean_28d"] = roll_28
    future["avg_discount"]  = last["avg_discount"]

    # Use actual values for historical dates
    hist = df[["date", "is_q4", "near_holiday", "roll_mean_28d", "avg_discount"]].rename(
        columns={"date": "ds"}
    )
    for col in ["is_q4", "near_holiday", "roll_mean_28d", "avg_discount"]:
        mask = future["ds"].isin(hist["ds"])
        future.loc[mask, col] = future.loc[mask, "ds"].map(
            hist.set_index("ds")[col]
        ).values

    forecast = model.predict(future)
    forecast_out = forecast[forecast["ds"] > df["date"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    forecast_out.columns = ["date", "yhat", "yhat_lower", "yhat_upper"]
    forecast_out["yhat"]       = forecast_out["yhat"].clip(lower=0).round(2)
    forecast_out["yhat_lower"] = forecast_out["yhat_lower"].clip(lower=0).round(2)
    forecast_out["yhat_upper"] = forecast_out["yhat_upper"].clip(lower=0).round(2)
    return forecast_out


def run(save: bool = True):
    log.info(" Prophet Model ")
    df = load_features()

    model, holdout_df, cv_results = train_eval(df)

    log.info("Generating 90-day forecast with uncertainty intervals...")
    forecast_df = generate_forecast(model, df)

    if save:
        with open(MODELS_DIR / "prophet_model.json", "w") as f:
            f.write(model_to_json(model))
        with open(MODELS_DIR / "prophet_cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        forecast_df.to_csv(MODELS_DIR / "prophet_forecast.csv", index=False)
        holdout_df[["ds","y_true","yhat","yhat_lower","yhat_upper"]].to_csv(
            MODELS_DIR / "prophet_holdout.csv", index=False
        )
        log.info(f"Saved model + results to {MODELS_DIR}/")

    return model, cv_results, forecast_df


if __name__ == "__main__":
    run()
