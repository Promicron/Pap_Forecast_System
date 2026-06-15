"""
kaggle_ingest.py
Ingestion adapter for the Kaggle Superstore Sales dataset.

Maps Superstore columns → pipeline schema and writes the three
dimension/fact tables that clean.py, features.py, and eda.py expect:
  data/customers.csv
  data/products.csv
  data/transactions.csv

Column mapping
──────────────
Superstore              → Pipeline schema
─────────────────────────────────────────
Customer ID             → customer_id  (SHA-256 anonymised)
Customer Name           → [dropped — PII]
Segment                 → segment
State/Province          → region
(tenure derived)        → tenure_days  (days since first order)
(tier derived)          → account_tier (by lifetime spend quartile)

Product ID              → product_id
Category                → category
(price derived)         → base_price  (median unit price per product)
(margin derived)        → margin_pct  (Profit / Sales per product, clipped)
(recurrence heuristic)  → is_recurring (Technology category OR reordered 3+ times)

Order ID + Row ID       → transaction_id
Order Date              → date
Sales                   → revenue
Sales - Profit          → cost
Profit                  → gross_profit
Quantity                → quantity
Discount                → discount_pct
"""

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

RAW_PATH = Path("data/samplesuperstore.csv")
OUT_DIR  = Path(__file__).parent.parent / "data"


def anonymise_id(raw_id: str) -> str:
    return "CUST_" + hashlib.sha256(raw_id.encode()).hexdigest()[:10].upper()


def derive_account_tier(lifetime_spend: pd.Series) -> pd.Series:
    q66 = lifetime_spend.quantile(0.66)
    q33 = lifetime_spend.quantile(0.33)
    return pd.cut(
        lifetime_spend,
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Bronze", "Silver", "Gold"],
    ).astype(str)


def derive_tenure(df: pd.DataFrame) -> pd.Series:
    span = df.groupby("Customer ID")["Order Date"].agg(first="min", last="max")
    span["tenure_days"] = (span["last"] - span["first"]).dt.days.clip(lower=1)
    return span["tenure_days"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=False)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=False)
    log.info(f"Loaded {len(df):,} rows  |  {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")
    return df


def build_customers(df: pd.DataFrame) -> pd.DataFrame:
    ltv    = df.groupby("Customer ID")["Sales"].sum().rename("lifetime_spend")
    tenure = derive_tenure(df)

    customers = (
        df[["Customer ID", "Segment", "State/Province"]]
        .drop_duplicates("Customer ID")
        .set_index("Customer ID")
        .join(ltv)
        .join(tenure)
        .reset_index()
    )

    customers["account_tier"] = derive_account_tier(customers["lifetime_spend"])
    customers["customer_id"]  = customers["Customer ID"].apply(anonymise_id)

    customers = customers.rename(columns={
        "Segment":        "segment",
        "State/Province": "region",
    })[["customer_id", "segment", "region", "tenure_days", "account_tier"]]

    log.info(f"Customer dimension: {len(customers):,} rows")
    return customers


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["unit_price"] = df["Sales"] / df["Quantity"].clip(lower=1)
    df["margin_tx"]  = (df["Profit"] / df["Sales"].replace(0, np.nan)).fillna(0)

    prod_agg = df.groupby("Product ID").agg(
        category   = ("Category",   "first"),
        base_price = ("unit_price", "median"),
        margin_pct = ("margin_tx",  "median"),
        n_orders   = ("Order ID",   "nunique"),
    ).reset_index()

    cust_per_prod = df.groupby("Product ID")["Customer ID"].nunique().rename("unique_customers")
    prod_agg = prod_agg.join(cust_per_prod, on="Product ID")
    prod_agg["is_recurring"] = (
        (prod_agg["category"] == "Technology") |
        (prod_agg["unique_customers"] >= 3)
    )

    prod_agg["margin_pct"] = prod_agg["margin_pct"].clip(-0.5, 0.9).round(4)
    prod_agg["base_price"] = prod_agg["base_price"].round(2)

    prod_agg = prod_agg.rename(columns={"Product ID": "product_id"})[
        ["product_id", "category", "base_price", "margin_pct", "is_recurring"]
    ]

    log.info(f"Product dimension: {len(prod_agg):,} rows")
    return prod_agg


def build_transactions(df: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    cust_lookup = customers.set_index("customer_id")[["segment", "region", "account_tier"]]
    prod_lookup = products.set_index("product_id")[["is_recurring"]]

    txn = df.copy()
    txn["customer_id"]    = txn["Customer ID"].apply(anonymise_id)
    txn["product_id"]     = txn["Product ID"]
    txn["transaction_id"] = "TXN_SS_" + txn["Row ID"].astype(str).str.zfill(6)
    txn["date"]           = txn["Order Date"].dt.normalize()
    txn["revenue"]        = txn["Sales"].round(2)
    txn["cost"]           = (txn["Sales"] - txn["Profit"]).round(2)
    txn["gross_profit"]   = txn["Profit"].round(2)
    txn["quantity"]       = txn["Quantity"]
    txn["discount_pct"]   = txn["Discount"].round(4)
    txn["unit_price"]     = (txn["Sales"] / txn["Quantity"].clip(lower=1)).round(2)
    txn["category"]       = txn["Category"]

    txn = txn.join(cust_lookup, on="customer_id", how="left")
    txn = txn.join(prod_lookup, on="product_id",  how="left")
    txn["is_recurring"] = txn["is_recurring"].fillna(False)

    txn = txn[[
        "transaction_id", "date", "customer_id", "product_id",
        "segment", "region", "category",
        "quantity", "unit_price", "discount_pct",
        "revenue", "cost", "gross_profit",
        "is_recurring", "account_tier",
    ]].reset_index(drop=True)

    log.info(f"Transaction fact table: {len(txn):,} rows")
    return txn


def ingest(save: bool = True):
    log.info(" Superstore ingest ")
    raw = load_raw()

    log.info("Building customer dimension...")
    customers = build_customers(raw)

    log.info("Building product dimension...")
    products = build_products(raw)

    log.info("Building transaction fact table...")
    transactions = build_transactions(raw, customers, products)

    assert transactions["customer_id"].isin(customers["customer_id"]).all(), \
        "Referential integrity fail: orphan customer_ids"
    assert transactions["product_id"].isin(products["product_id"]).all(), \
        "Referential integrity fail: orphan product_ids"

    expected = {
        "transaction_id","date","customer_id","product_id","segment","region",
        "category","quantity","unit_price","discount_pct","revenue","cost",
        "gross_profit","is_recurring","account_tier",
    }
    missing = expected - set(transactions.columns)
    assert not missing, f"Missing expected columns: {missing}"

    log.info("Referential integrity + schema: OK")

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        customers.to_csv(OUT_DIR / "customers.csv",    index=False)
        products.to_csv(OUT_DIR  / "products.csv",     index=False)
        transactions.to_csv(OUT_DIR / "transactions.csv", index=False)
        log.info(f"Saved to {OUT_DIR}/")

    return customers, products, transactions


if __name__ == "__main__":
    ingest()
