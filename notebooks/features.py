"""
features.py
Feature engineering for sales forecasting.

Produces two output frames:
  1. daily_features.csv  — time-series features for Prophet / XGBoost forecasting
  2. model_features.csv  — transaction-level ML features with target variable

Feature groups
A. Temporal                  — calendar signals: dow, month, quarter, week-of-year,
                               is_weekend, is_month_end, days_to_quarter_end, ...
B. Rolling aggregates         — 7d / 14d / 28d / 90d revenue lag windows
C. Lag features               — revenue at t-7, t-14, t-28 (prevents leakage)
D. Customer-level features    — segment, tenure, tier, LTV proxy, repeat rate
E. Product-level features     — category, margin, recurring flag
F. Business cycle features    — Q4 flag, fiscal period, holiday proximity (UK)
G. Derived ratios             — avg order value, discount intensity, GP margin
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# A. Temporal features
def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    df["day_of_week"]         = d.dt.dayofweek            # 0=Mon
    df["day_of_month"]        = d.dt.day
    df["week_of_year"]        = d.dt.isocalendar().week.astype(int)
    df["month"]               = d.dt.month
    df["quarter"]             = d.dt.quarter
    df["year"]                = d.dt.year
    df["is_weekend"]          = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"]      = d.dt.is_month_start.astype(int)
    df["is_month_end"]        = d.dt.is_month_end.astype(int)
    df["is_quarter_end"]      = d.dt.is_quarter_end.astype(int)
    df["is_q4"]               = (d.dt.month >= 10).astype(int)
    df["days_in_month"]       = d.dt.days_in_month

    # Days to quarter end (helps model pre-quarter urgency)
    quarter_ends = d.dt.to_period("Q").dt.end_time.dt.normalize()
    df["days_to_quarter_end"] = (quarter_ends - d).dt.days.clip(lower=0)

    # UK public holiday proximity (simplified: Bank Holidays ± 3 days)
    uk_bank_holidays = pd.to_datetime([
        "2022-01-03","2022-04-15","2022-04-18","2022-05-02","2022-05-30",
        "2022-06-02","2022-06-03","2022-08-29","2022-12-26","2022-12-27",
        "2023-01-02","2023-04-07","2023-04-10","2023-05-01","2023-05-08",
        "2023-05-29","2023-08-28","2023-12-25","2023-12-26",
        "2024-01-01","2024-03-29","2024-04-01","2024-05-06","2024-05-27",
    ])
    df["days_to_holiday"] = d.apply(
        lambda x: min(abs((x - h).days) for h in uk_bank_holidays)
    ).clip(upper=14)
    df["near_holiday"] = (df["days_to_holiday"] <= 3).astype(int)

    # Fourier pair for weekly seasonality (period=7)
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Fourier pair for annual seasonality (period=365)
    day_of_year = d.dt.day_of_year
    df["sin_year"] = np.sin(2 * np.pi * day_of_year / 365)
    df["cos_year"] = np.cos(2 * np.pi * day_of_year / 365)

    return df


# B. Daily aggregation
def aggregate_daily(txn: pd.DataFrame) -> pd.DataFrame:
    daily = txn.groupby("date").agg(
        revenue          = ("revenue",      "sum"),
        gross_profit     = ("gross_profit", "sum"),
        n_transactions   = ("transaction_id","count"),
        n_customers      = ("customer_id",  "nunique"),
        n_products       = ("product_id",   "nunique"),
        avg_order_value  = ("revenue",      "mean"),
        avg_discount     = ("discount_pct", "mean"),
        avg_quantity     = ("quantity",     "mean"),
        recurring_rev    = ("revenue",      lambda x: x[txn.loc[x.index,"is_recurring"]].sum()),
    ).reset_index()

    daily["gp_margin"]         = (daily["gross_profit"] / daily["revenue"]).round(4)
    daily["recurring_share"]   = (daily["recurring_rev"] / daily["revenue"]).round(4)
    daily["rev_per_customer"]  = (daily["revenue"] / daily["n_customers"]).round(2)
    return daily


# C. Rolling & lag features
def add_rolling_and_lags(df: pd.DataFrame, target: str = "revenue") -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    for window in [7, 14, 28, 90]:
        df[f"roll_mean_{window}d"]  = df[target].shift(1).rolling(window, min_periods=1).mean()
        df[f"roll_std_{window}d"]   = df[target].shift(1).rolling(window, min_periods=1).std().fillna(0)

    for lag in [7, 14, 28, 56]:
        df[f"lag_{lag}d"] = df[target].shift(lag)

    # 7d-over-28d momentum ratio
    df["momentum_7_28"] = (df["roll_mean_7d"] / df["roll_mean_28d"].replace(0, np.nan)).fillna(1).round(4)

    # Exponentially weighted mean (captures recent trend more sensitively)
    df["ewm_14d"] = df[target].shift(1).ewm(span=14, adjust=False).mean()

    return df


# D. Segment-level features (daily breakdown)
def add_segment_features(txn: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    seg_daily = txn.groupby(["date","segment"])["revenue"].sum().unstack(fill_value=0)
    seg_daily.columns = [f"rev_{c.lower()}" for c in seg_daily.columns]
    seg_daily = seg_daily.reset_index()
    return daily.merge(seg_daily, on="date", how="left")


# E. Category-level features
def add_category_features(txn: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    cat_daily = txn.groupby(["date","category"])["revenue"].sum().unstack(fill_value=0)
    cat_daily.columns = [f"rev_{c.lower()}" for c in cat_daily.columns]
    cat_daily = cat_daily.reset_index()
    return daily.merge(cat_daily, on="date", how="left")


# F. Customer lifetime value proxy (per customer, joined via txn)
def build_customer_features(txn: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    cutoff = txn["date"].max()
    rfm = txn.groupby("customer_id").agg(
        recency_days   = ("date", lambda x: (cutoff - x.max()).days),
        frequency      = ("transaction_id", "count"),
        monetary       = ("revenue", "sum"),
        avg_order      = ("revenue", "mean"),
        first_purchase = ("date", "min"),
        last_purchase  = ("date", "max"),
    ).reset_index()
    rfm["clv_proxy"] = (rfm["monetary"] / rfm["frequency"].clip(lower=1)) * np.log1p(rfm["frequency"])
    rfm = rfm.merge(customers[["customer_id","segment","region","account_tier","tenure_days"]], on="customer_id", how="left")
    return rfm


# Main pipeline
def run(save: bool = True) -> pd.DataFrame:
    log.info("Loading clean data...")
    txn       = pd.read_csv(DATA_DIR / "transactions_clean.csv", parse_dates=["date"])
    customers = pd.read_csv(DATA_DIR / "customers_clean.csv")
    products  = pd.read_csv(DATA_DIR / "products_clean.csv")

    # Merge product info onto transactions for derived features
    txn = txn.merge(products[["product_id","is_recurring","margin_pct"]], on="product_id", how="left")
    # is_recurring may already be in txn; keep the one from products if missing
    if "is_recurring_x" in txn.columns:
        txn["is_recurring"] = txn["is_recurring_x"].fillna(txn["is_recurring_y"])
        txn = txn.drop(columns=["is_recurring_x","is_recurring_y"])

    log.info("Aggregating daily totals...")
    daily = aggregate_daily(txn)

    log.info("Adding temporal features...")
    daily = add_temporal_features(daily)

    log.info("Adding rolling windows and lags...")
    daily = add_rolling_and_lags(daily)

    log.info("Adding segment breakdown...")
    daily = add_segment_features(txn, daily)

    log.info("Adding category breakdown...")
    daily = add_category_features(txn, daily)

    log.info("Building customer RFM features...")
    customer_features = build_customer_features(txn, customers)

    # Summary stats
    log.info(f"\n{'─'*50}")
    log.info(f"Feature matrix shape : {daily.shape}")
    log.info(f"Date range           : {daily['date'].min().date()} → {daily['date'].max().date()}")
    log.info(f"Revenue range        : £{daily['revenue'].min():,.0f} – £{daily['revenue'].max():,.0f}")
    log.info(f"Null counts          :\n{daily.isnull().sum()[daily.isnull().sum() > 0]}")
    log.info(f"{'─'*50}")

    if save:
        daily.to_csv(DATA_DIR / "daily_features.csv", index=False)
        customer_features.to_csv(DATA_DIR / "customer_features.csv", index=False)
        log.info(f"Saved feature files to {DATA_DIR}/")

    return daily, customer_features


if __name__ == "__main__":
    run()
