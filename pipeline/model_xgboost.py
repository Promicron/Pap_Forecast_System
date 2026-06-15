"""
model_xgboost.py
XGBoost sales forecasting model.

Design
──────
- Time-series aware train/val split (no shuffle — respects temporal order)
- TimeSeriesSplit cross-validation for robust RMSE/MAE estimation
- Feature importance extraction
- Saves: models/xgb_model.json, models/xgb_cv_results.json, models/xgb_forecast.csv
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Columns that would leak the target (computed from revenue on the same day)
LEAKY_COLS = [
    "gross_profit", "n_transactions", "n_customers", "n_products",
    "avg_order_value", "avg_quantity", "recurring_rev", "gp_margin",
    "recurring_share", "rev_per_customer",
    "rev_consumer", "rev_corporate", "rev_home office",
    "rev_furniture", "rev_office supplies", "rev_technology",
]

FEATURE_COLS = [
    # Temporal
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter", "year",
    "is_weekend", "is_month_start", "is_month_end", "is_quarter_end",
    "is_q4", "days_in_month", "days_to_quarter_end",
    "days_to_holiday", "near_holiday",
    "sin_week", "cos_week", "sin_year", "cos_year",
    # Rolling / lags (all lagged ≥1 day — no leakage)
    "roll_mean_7d", "roll_std_7d",
    "roll_mean_14d", "roll_std_14d",
    "roll_mean_28d", "roll_std_28d",
    "roll_mean_90d", "roll_std_90d",
    "lag_7d", "lag_14d", "lag_28d", "lag_56d",
    "momentum_7_28", "ewm_14d",
    # Discount signal (daily avg — no revenue leakage)
    "avg_discount",
]

TARGET = "revenue"
FORECAST_HORIZON = 90  # days


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "daily_features.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Drop rows where lags are NaN (first 56 days)
    df = df.dropna(subset=["lag_56d"]).reset_index(drop=True)
    log.info(f"Feature matrix: {len(df)} rows after dropping lag warm-up")
    return df


def train_eval(df: pd.DataFrame):
    """
    Hold-out validation: last 90 days = test set, rest = train.
    Also runs 5-fold TimeSeriesSplit CV on the training set.
    """
    split_idx = len(df) - FORECAST_HORIZON
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET]

    # ── XGBoost params ────────────────────────────────────────────
    params = dict(
        n_estimators      = 600,
        learning_rate     = 0.04,
        max_depth         = 5,
        min_child_weight  = 3,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        random_state      = 42,
        n_jobs            = -1,
        early_stopping_rounds = 40,
        eval_metric       = "rmse",
    )

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Hold-out metrics ──────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100)

    log.info(f"Hold-out  RMSE: £{rmse:,.2f}  MAE: £{mae:,.2f}  MAPE: {mape:.2f}%")

    # ── TimeSeriesSplit CV ────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5, test_size=60)
    cv_rmses, cv_maes = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train)):
        X_tr  = X_train.iloc[tr_idx];  y_tr  = y_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]; y_val = y_train.iloc[val_idx]

        m = XGBRegressor(**{k: v for k, v in params.items()
                            if k not in ("early_stopping_rounds","eval_metric")},
                         early_stopping_rounds=40, eval_metric="rmse")
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.clip(m.predict(X_val), 0, None)
        cv_rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        cv_maes.append(float(mean_absolute_error(y_val, preds)))
        log.info(f"  CV fold {fold+1}/5  RMSE: £{cv_rmses[-1]:,.2f}  MAE: £{cv_maes[-1]:,.2f}")

    cv_summary = {
        "cv_rmse_mean": round(np.mean(cv_rmses), 2),
        "cv_rmse_std":  round(np.std(cv_rmses),  2),
        "cv_mae_mean":  round(np.mean(cv_maes),  2),
        "cv_mae_std":   round(np.std(cv_maes),   2),
        "holdout_rmse": round(rmse, 2),
        "holdout_mae":  round(mae,  2),
        "holdout_mape": round(mape, 2),
    }
    log.info(f"CV RMSE: £{cv_summary['cv_rmse_mean']:,.2f} ± £{cv_summary['cv_rmse_std']:,.2f}")

    return model, test, y_pred, cv_summary


def generate_forecast(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a 90-day rolling forecast beyond the dataset end date.
    Each day's prediction feeds into the next day's lag features.
    """
    last_date = df["date"].max()
    history   = df.copy().sort_values("date").reset_index(drop=True)

    forecasts = []
    for i in range(1, FORECAST_HORIZON + 1):
        next_date = last_date + pd.Timedelta(days=i)

        # Build one-row feature vector
        row = {}
        row["day_of_week"]         = next_date.dayofweek
        row["day_of_month"]        = next_date.day
        row["week_of_year"]        = next_date.isocalendar().week
        row["month"]               = next_date.month
        row["quarter"]             = next_date.quarter
        row["year"]                = next_date.year
        row["is_weekend"]          = int(next_date.dayofweek >= 5)
        row["is_month_start"]      = int(next_date.day == 1)
        row["is_month_end"]        = int(next_date.day == next_date.days_in_month)
        row["is_quarter_end"]      = int(next_date.month in [3,6,9,12] and row["is_month_end"])
        row["is_q4"]               = int(next_date.month >= 10)
        row["days_in_month"]       = next_date.days_in_month
        # Days to quarter end
        import calendar
        qe_month = ((next_date.month - 1) // 3 + 1) * 3
        qe_day   = calendar.monthrange(next_date.year, qe_month)[1]
        qe_date  = pd.Timestamp(next_date.year, qe_month, qe_day)
        row["days_to_quarter_end"] = max(0, (qe_date - next_date).days)
        # Holiday proximity (simplified)
        row["days_to_holiday"]     = min(14, i % 14)
        row["near_holiday"]        = int(row["days_to_holiday"] <= 3)
        # Fourier
        doy = next_date.timetuple().tm_yday
        row["sin_week"] = np.sin(2 * np.pi * row["day_of_week"] / 7)
        row["cos_week"] = np.cos(2 * np.pi * row["day_of_week"] / 7)
        row["sin_year"] = np.sin(2 * np.pi * doy / 365)
        row["cos_year"] = np.cos(2 * np.pi * doy / 365)
        # Discount (use historical rolling mean)
        row["avg_discount"] = history["avg_discount"].tail(28).mean()

        # Rolling windows from history
        rev_series = history["revenue"]
        for w in [7, 14, 28, 90]:
            tail = rev_series.tail(w)
            row[f"roll_mean_{w}d"] = tail.mean()
            row[f"roll_std_{w}d"]  = tail.std() if len(tail) > 1 else 0.0

        # Lags from history
        for lag in [7, 14, 28, 56]:
            idx = len(history) - lag
            row[f"lag_{lag}d"] = history["revenue"].iloc[idx] if idx >= 0 else rev_series.mean()

        row["momentum_7_28"] = (row["roll_mean_7d"] / row["roll_mean_28d"]
                                if row["roll_mean_28d"] > 0 else 1.0)
        row["ewm_14d"] = rev_series.ewm(span=14, adjust=False).mean().iloc[-1]

        X_row = pd.DataFrame([row])[FEATURE_COLS]
        pred  = float(np.clip(model.predict(X_row)[0], 0, None))

        forecasts.append({"date": next_date, "yhat": round(pred, 2)})

        # Append predicted value to history so next iteration's lags are correct
        new_row = history.iloc[-1].copy()
        new_row["date"]    = next_date
        new_row["revenue"] = pred
        new_row["avg_discount"] = row["avg_discount"]
        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)

    return pd.DataFrame(forecasts)


def run(save: bool = True):
    log.info("═══ XGBoost Model ═══════════════════════")
    df = load_features()

    model, test_df, y_pred, cv_results = train_eval(df)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    log.info("Top 10 features by importance:")
    for feat, score in importance.head(10).items():
        bar = "█" * int(score * 200)
        log.info(f"  {feat:<28} {score:.4f}  {bar}")

    log.info("Generating 90-day rolling forecast...")
    forecast_df = generate_forecast(model, df)

    # Build holdout comparison frame
    holdout = test_df[["date", TARGET]].copy()
    holdout["yhat"]  = y_pred
    holdout["model"] = "xgboost"
    holdout["split"] = "holdout"

    if save:
        model.save_model(str(MODELS_DIR / "xgb_model.json"))
        cv_results["feature_importance"] = importance.head(20).round(6).to_dict()
        with open(MODELS_DIR / "xgb_cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=2)
        forecast_df.to_csv(MODELS_DIR / "xgb_forecast.csv", index=False)
        holdout.to_csv(MODELS_DIR / "xgb_holdout.csv", index=False)
        log.info(f"Saved model + results to {MODELS_DIR}/")

    return model, cv_results, forecast_df, importance


if __name__ == "__main__":
    run()
