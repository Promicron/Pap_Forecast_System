"""
test_pipeline.py
Unit + integration tests for the data pipeline.
Run with: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from notebooks.generate_data import build_customers, build_products, anonymise_id
from clean import clean_transactions, clean_customers, enforce_referential_integrity
from notebooks.features import add_temporal_features, add_rolling_and_lags, aggregate_daily


# ── Anonymisation ────────────────────────────────────────────────────────────
class TestAnonymisation:
    def test_id_is_deterministic(self):
        assert anonymise_id("test@example.com") == anonymise_id("test@example.com")

    def test_id_does_not_contain_original(self):
        raw = "john.smith@company.co.uk"
        hashed = anonymise_id(raw)
        assert "john" not in hashed.lower()
        assert "smith" not in hashed.lower()
        assert "company" not in hashed.lower()

    def test_id_has_prefix(self):
        assert anonymise_id("any@email.com").startswith("CUST_")

    def test_different_inputs_differ(self):
        assert anonymise_id("a@x.com") != anonymise_id("b@x.com")


# ── Customer generation ────────────────────────────────────────────────────
class TestCustomerGeneration:
    def test_correct_columns(self):
        df = build_customers(50)
        expected = {"customer_id","segment","region","tenure_days","account_tier"}
        assert expected.issubset(df.columns)

    def test_no_pii_in_ids(self):
        df = build_customers(50)
        for cid in df["customer_id"]:
            assert "@" not in cid
            assert "." not in cid.replace("CUST_","")

    def test_valid_segments(self):
        df = build_customers(100)
        valid = {"Enterprise","SMB","Consumer","Partner"}
        assert set(df["segment"].unique()).issubset(valid)

    def test_tenure_non_negative(self):
        df = build_customers(100)
        assert (df["tenure_days"] >= 0).all()


# ── Data cleaning ─────────────────────────────────────────────────────────
class TestCleaning:
    def _make_txn(self, n=50):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "transaction_id": [f"TXN_{i}" for i in range(n)],
            "date":           pd.date_range("2023-01-01", periods=n, freq="D"),
            "customer_id":    [f"CUST_{i%10}" for i in range(n)],
            "product_id":     [f"PROD_{i%5}" for i in range(n)],
            "segment":        ["SMB"] * n,
            "region":         ["London"] * n,
            "category":       ["Software"] * n,
            "quantity":       rng.integers(1, 5, size=n),
            "unit_price":     rng.uniform(100, 1000, size=n).round(2),
            "discount_pct":   rng.uniform(0, 0.2, size=n).round(3),
            "revenue":        rng.uniform(100, 5000, size=n).round(2),
            "cost":           rng.uniform(50, 2500, size=n).round(2),
            "gross_profit":   rng.uniform(50, 2500, size=n).round(2),
            "is_recurring":   [False] * n,
            "account_tier":   ["Silver"] * n,
        })

    def test_removes_duplicates(self):
        df = self._make_txn(20)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        cleaned = clean_transactions(df)
        assert cleaned["transaction_id"].is_unique

    def test_removes_non_positive_revenue(self):
        df = self._make_txn(20)
        df.loc[0, "revenue"] = -100
        df.loc[1, "revenue"] = 0
        cleaned = clean_transactions(df)
        assert (cleaned["revenue"] > 0).all()

    def test_discount_capped_at_50pct(self):
        df = self._make_txn(10)
        df.loc[0, "discount_pct"] = 0.99
        cleaned = clean_transactions(df)
        assert cleaned["discount_pct"].max() <= 0.50

    def test_referential_integrity(self):
        txn = self._make_txn(20)
        customers = pd.DataFrame({"customer_id": [f"CUST_{i}" for i in range(5)]})
        products  = pd.DataFrame({"product_id":  [f"PROD_{i}" for i in range(5)]})
        result = enforce_referential_integrity(txn, customers, products)
        assert result["customer_id"].isin(customers["customer_id"]).all()
        assert result["product_id"].isin(products["product_id"]).all()


# ── Feature engineering ────────────────────────────────────────────────────
class TestFeatureEngineering:
    def _make_daily(self):
        return pd.DataFrame({
            "date":    pd.date_range("2023-01-01", periods=60, freq="D"),
            "revenue": np.random.default_rng(1).uniform(10000, 50000, 60).round(2),
        })

    def test_temporal_features_present(self):
        df = add_temporal_features(self._make_daily())
        for col in ["day_of_week","month","quarter","is_weekend","is_q4","sin_year","cos_year"]:
            assert col in df.columns, f"Missing: {col}"

    def test_is_weekend_correct(self):
        df = add_temporal_features(self._make_daily())
        for _, row in df.iterrows():
            expected = 1 if row["date"].dayofweek >= 5 else 0
            assert row["is_weekend"] == expected

    def test_rolling_features_present(self):
        df = add_rolling_and_lags(self._make_daily())
        for col in ["roll_mean_7d","roll_mean_28d","lag_7d","lag_28d","momentum_7_28","ewm_14d"]:
            assert col in df.columns, f"Missing: {col}"

    def test_no_future_leakage_in_lags(self):
        df = add_rolling_and_lags(self._make_daily())
        # lag_7d on day index 10 should equal revenue on day index 3
        assert df.loc[10, "lag_7d"] == pytest.approx(df.loc[3, "revenue"], rel=1e-3)

    def test_agg_revenue_matches_sum(self):
        txn = pd.DataFrame({
            "date":           pd.to_datetime(["2023-01-01","2023-01-01","2023-01-02"]),
            "transaction_id": ["T1","T2","T3"],
            "customer_id":    ["C1","C2","C1"],
            "product_id":     ["P1","P1","P2"],
            "revenue":        [100.0, 200.0, 150.0],
            "gross_profit":   [40.0, 80.0, 60.0],
            "quantity":       [1, 2, 1],
            "discount_pct":   [0.0, 0.05, 0.0],
            "is_recurring":   [False, False, True],
        })
        daily = aggregate_daily(txn)
        assert daily.loc[daily["date"] == pd.Timestamp("2023-01-01"), "revenue"].values[0] == pytest.approx(300.0)
        assert daily.loc[daily["date"] == pd.Timestamp("2023-01-02"), "revenue"].values[0] == pytest.approx(150.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
