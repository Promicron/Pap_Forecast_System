"""
Generates realistic synthetic sales + customer transactional data.
Includes seasonality, trends, noise, and anonymised customer records.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import hashlib
import random
import os

fake = Faker("en_GB")
rng = np.random.default_rng(42)
random.seed(42)

# Config
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 6, 30)
N_CUSTOMERS = 800
N_PRODUCTS  = 60
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "../data")

SEGMENTS = ["Enterprise", "SMB", "Consumer", "Partner"]
SEG_WEIGHTS = [0.15, 0.35, 0.40, 0.10]

CATEGORIES = ["Software", "Hardware", "Services", "Support", "Training"]
CAT_WEIGHTS = [0.30, 0.20, 0.25, 0.15, 0.10]

REGIONS = ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol", "Leeds"]
REG_WEIGHTS = [0.30, 0.18, 0.15, 0.12, 0.13, 0.12]


# Anonymisation helper
def anonymise_id(raw_id: str) -> str:
    """One-way hash of a customer identifier. Non-reversible."""
    return "CUST_" + hashlib.sha256(raw_id.encode()).hexdigest()[:10].upper()


# Customer dimension table
def build_customers(n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        raw_email = fake.company_email()
        rows.append({
            "customer_id": anonymise_id(raw_email),   # PII removed
            "segment":     rng.choice(SEGMENTS, p=SEG_WEIGHTS),
            "region":      rng.choice(REGIONS,  p=REG_WEIGHTS),
            "tenure_days": int(rng.integers(30, 1200)),
            "account_tier": rng.choice(["Gold", "Silver", "Bronze"], p=[0.2, 0.35, 0.45]),
        })
    df = pd.DataFrame(rows).drop_duplicates("customer_id").reset_index(drop=True)
    return df


# Product dimension table
def build_products(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cat = rng.choice(CATEGORIES, p=CAT_WEIGHTS)
        base_price = {
            "Software": rng.uniform(200, 3500),
            "Hardware": rng.uniform(500, 8000),
            "Services": rng.uniform(1000, 15000),
            "Support":  rng.uniform(150, 1200),
            "Training": rng.uniform(300, 2500),
        }[cat]
        rows.append({
            "product_id":   f"PROD_{i+1:04d}",
            "category":     cat,
            "base_price":   round(base_price, 2),
            "margin_pct":   round(rng.uniform(0.15, 0.65), 3),
            "is_recurring": bool(rng.choice([True, False], p=[0.4, 0.6])),
        })
    return pd.DataFrame(rows)


# Sales seasonality + trend signal
def sales_multiplier(date: datetime) -> float:
    """
    Encodes:
      - Yearly upward trend
      - Q4 spike (Oct-Dec strong), Q1 dip (Jan weak)
      - Monthly cycle with slight mid-month peak
      - Random noise
    """
    day_of_year = date.timetuple().tm_yday
    year_frac   = (date - START_DATE).days / 365.0

    trend      = 1.0 + 0.18 * year_frac                      # +18% per year
    seasonality = 1.0 + 0.25 * np.sin((day_of_year / 365) * 2 * np.pi - np.pi / 2)
    q4_boost   = 1.25 if date.month in [10, 11, 12] else 1.0
    q1_drag    = 0.85 if date.month in [1, 2]       else 1.0
    mid_month  = 1.0 + 0.05 * np.sin((date.day / 30) * 2 * np.pi)
    noise      = rng.normal(1.0, 0.06)

    return max(0.3, trend * seasonality * q4_boost * q1_drag * mid_month * noise)


# Transaction fact table
def build_transactions(customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []

    for date in date_range:
        multiplier = sales_multiplier(date.to_pydatetime())
        n_orders   = max(1, int(rng.poisson(6 * multiplier)))

        for _ in range(n_orders):
            cust    = customers.sample(1, random_state=None).iloc[0]
            prod    = products.sample(1, random_state=None).iloc[0]
            qty     = int(rng.integers(1, 6))
            discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20],
                                   p=[0.55, 0.20, 0.12, 0.08, 0.05])

            unit_price = prod["base_price"] * rng.uniform(0.90, 1.10)
            revenue    = round(unit_price * qty * (1 - discount), 2)
            cost       = round(revenue * (1 - prod["margin_pct"]), 2)

            rows.append({
                "transaction_id":  f"TXN_{len(rows)+1:08d}",
                "date":            date.date(),
                "customer_id":     cust["customer_id"],
                "product_id":      prod["product_id"],
                "segment":         cust["segment"],
                "region":          cust["region"],
                "category":        prod["category"],
                "quantity":        qty,
                "unit_price":      round(unit_price, 2),
                "discount_pct":    discount,
                "revenue":         revenue,
                "cost":            cost,
                "gross_profit":    round(revenue - cost, 2),
                "is_recurring":    prod["is_recurring"],
                "account_tier":    cust["account_tier"],
            })

    return pd.DataFrame(rows)


# Main
def generate(save: bool = True):
    print("Building customer dimension...")
    customers = build_customers(N_CUSTOMERS)

    print("Building product dimension...")
    products  = build_products(N_PRODUCTS)

    print("Generating transactions (this takes ~20s)...")
    transactions = build_transactions(customers, products)

    print(f"  → {len(transactions):,} transactions across {len(transactions['date'].unique())} days")

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
        products.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
        transactions.to_csv(f"{OUTPUT_DIR}/transactions.csv", index=False)
        print(f"  → Saved to {OUTPUT_DIR}/")

    return customers, products, transactions


if __name__ == "__main__":
    generate()
