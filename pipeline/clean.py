"""
Handles: nulls, outliers, type coercion, deduplication, referential integrity.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# Load raw data
def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers    = pd.read_csv(DATA_DIR / "customers.csv")
    products     = pd.read_csv(DATA_DIR / "products.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    log.info(f"Loaded {len(transactions):,} raw transactions")
    return customers, products, transactions


# Cleaning steps
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    original_len = len(df)

    # 1. Drop exact duplicates
    df = df.drop_duplicates(subset="transaction_id")
    log.info(f"  Dedup: removed {original_len - len(df)} duplicates")

    # 2. Enforce types
    df["date"]         = pd.to_datetime(df["date"])
    df["quantity"]     = pd.to_numeric(df["quantity"],     errors="coerce").astype("Int64")
    df["revenue"]      = pd.to_numeric(df["revenue"],      errors="coerce")
    df["unit_price"]   = pd.to_numeric(df["unit_price"],   errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    df["gross_profit"] = pd.to_numeric(df["gross_profit"], errors="coerce")

    # 3. Remove non-positive revenue (returns/voids are out of scope for forecasting)
    neg_mask = df["revenue"] <= 0
    df = df[~neg_mask]
    log.info(f"  Removed {neg_mask.sum()} non-positive revenue rows")

    # 4. Outlier capping: Winsorise revenue at 99.5th percentile per category
    upper_caps = df.groupby("category")["revenue"].quantile(0.995)
    def cap_revenue(row):
        cap = upper_caps.get(row["category"], np.inf)
        return min(row["revenue"], cap)
    before_mean = df["revenue"].mean()
    df["revenue"] = df.apply(cap_revenue, axis=1)
    log.info(f"  Winsorise: revenue mean {before_mean:.0f} → {df['revenue'].mean():.0f}")

    # 5. Drop nulls in critical columns
    critical = ["date", "customer_id", "product_id", "revenue"]
    null_before = len(df)
    df = df.dropna(subset=critical)
    log.info(f"  Dropped {null_before - len(df)} rows with null critical fields")

    # 6. Discount sanity check
    df["discount_pct"] = df["discount_pct"].clip(0.0, 0.50)

    # 7. Quantity sanity check
    df["quantity"] = df["quantity"].clip(lower=1)

    # 8. Recompute gross profit from actuals (in case of rounding drift)
    df["gross_profit"] = (df["revenue"] - df["cost"]).round(2)

    log.info(f"  Clean complete: {len(df):,} rows remain ({original_len - len(df):,} removed total)")
    return df.reset_index(drop=True)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customer_id")
    df["tenure_days"] = df["tenure_days"].clip(lower=0)
    df["segment"]     = df["segment"].str.strip()
    df["region"]      = df["region"].str.strip()
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="product_id")
    df["base_price"]  = df["base_price"].clip(lower=0.01)
    df["margin_pct"]  = df["margin_pct"].clip(0.0, 1.0)
    return df


# ─────────────────────────────────────────────
# Referential integrity
# ─────────────────────────────────────────────
def enforce_referential_integrity(
    txn: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    valid_customers = set(customers["customer_id"])
    valid_products  = set(products["product_id"])

    before = len(txn)
    txn = txn[
        txn["customer_id"].isin(valid_customers) &
        txn["product_id"].isin(valid_products)
    ]
    log.info(f"  Referential integrity: removed {before - len(txn)} orphan rows")
    return txn


# ─────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────
def run(save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers_raw, products_raw, txn_raw = load_raw()

    log.info("Cleaning customers...")
    customers = clean_customers(customers_raw)

    log.info("Cleaning products...")
    products = clean_products(products_raw)

    log.info("Cleaning transactions...")
    txn = clean_transactions(txn_raw)
    txn = enforce_referential_integrity(txn, customers, products)

    if save:
        customers.to_csv(DATA_DIR / "customers_clean.csv", index=False)
        products.to_csv(DATA_DIR / "products_clean.csv",   index=False)
        txn.to_csv(DATA_DIR / "transactions_clean.csv",    index=False)
        log.info(f"Saved clean data to {DATA_DIR}/")

    return customers, products, txn


if __name__ == "__main__":
    run()
