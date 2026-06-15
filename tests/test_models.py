"""
test_models.py
Unit tests for model modules — fast, no full training required.
Tests feature contract, forecast shape, and ensemble weighting logic.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from model_xgboost  import FEATURE_COLS, LEAKY_COLS, generate_forecast as xgb_forecast_fn
from model_prophet  import make_holidays, prepare_prophet_df, build_model
from model_ensemble import rmse_weight


# ── XGBoost feature contract ────────────────────────────────────────────────
class TestXGBoostFeatures:
    def test_no_leaky_cols_in_features(self):
        """Leaky columns must not appear in the XGBoost feature set."""
        overlap = set(FEATURE_COLS) & set(LEAKY_COLS)
        assert not overlap, f"Leaky columns in feature set: {overlap}"

    def test_feature_cols_are_unique(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_required_temporal_features_present(self):
        required = {"day_of_week", "month", "quarter", "is_q4", "sin_year", "cos_year"}
        assert required.issubset(set(FEATURE_COLS))

    def test_required_lag_features_present(self):
        required = {"lag_7d", "lag_14d", "lag_28d", "lag_56d"}
        assert required.issubset(set(FEATURE_COLS))

    def test_required_rolling_features_present(self):
        required = {"roll_mean_7d", "roll_mean_28d", "ewm_14d"}
        assert required.issubset(set(FEATURE_COLS))


# ── Prophet helpers ─────────────────────────────────────────────────────────
class TestProphetHelpers:
    def test_holidays_has_required_columns(self):
        h = make_holidays()
        assert {"holiday", "ds", "lower_window", "upper_window"}.issubset(h.columns)

    def test_holidays_are_datetime(self):
        h = make_holidays()
        assert pd.api.types.is_datetime64_any_dtype(h["ds"])

    def test_holidays_cover_multiple_years(self):
        h = make_holidays()
        years = h["ds"].dt.year.unique()
        assert len(years) >= 3

    def test_prepare_prophet_df_columns(self):
        df = pd.DataFrame({
            "date":         pd.date_range("2023-01-01", periods=30, freq="D"),
            "revenue":      np.random.default_rng(0).uniform(500, 3000, 30),
            "is_q4":        [0] * 30,
            "near_holiday": [0] * 30,
            "roll_mean_28d":[1000.0] * 30,
            "avg_discount": [0.1] * 30,
        })
        pdf = prepare_prophet_df(df)
        assert "ds" in pdf.columns
        assert "y"  in pdf.columns
        assert "is_q4" in pdf.columns

    def test_prepare_prophet_df_no_raw_pii_columns(self):
        """customer_id, Customer Name etc must not leak into Prophet df."""
        df = pd.DataFrame({
            "date":          pd.date_range("2023-01-01", periods=10, freq="D"),
            "revenue":       [1000.0] * 10,
            "is_q4":         [0] * 10,
            "near_holiday":  [0] * 10,
            "roll_mean_28d": [1000.0] * 10,
            "avg_discount":  [0.1] * 10,
            "customer_id":   ["CUST_ABC"] * 10,  # should be ignored
        })
        pdf = prepare_prophet_df(df)
        assert "customer_id" not in pdf.columns

    def test_model_instantiates(self):
        m = build_model()
        assert m is not None
        assert m.seasonality_mode == "multiplicative"
        assert m.interval_width   == 0.90


# ── Ensemble weighting ───────────────────────────────────────────────────────
class TestEnsembleWeights:
    def test_weights_sum_to_one(self):
        w_a, w_b = rmse_weight(100.0, 200.0)
        assert abs(w_a + w_b - 1.0) < 1e-9

    def test_lower_rmse_gets_higher_weight(self):
        w_a, w_b = rmse_weight(100.0, 300.0)
        assert w_a > w_b

    def test_equal_rmse_gives_equal_weights(self):
        w_a, w_b = rmse_weight(150.0, 150.0)
        assert abs(w_a - w_b) < 1e-9

    def test_weights_are_positive(self):
        w_a, w_b = rmse_weight(500.0, 1500.0)
        assert w_a > 0
        assert w_b > 0

    def test_extreme_difference_caps_sensibly(self):
        """Even with 10x RMSE gap, neither weight should be zero."""
        w_a, w_b = rmse_weight(10.0, 10000.0)
        assert w_a > 0.99
        assert w_b > 0


# ── Integration: saved artefacts ────────────────────────────────────────────
class TestSavedArtifacts:
    MODELS_DIR = Path(__file__).parent.parent / "models"

    def test_xgb_model_saved(self):
        assert (self.MODELS_DIR / "xgb_model.json").exists()

    def test_xgb_forecast_saved(self):
        path = self.MODELS_DIR / "xgb_forecast.csv"
        assert path.exists()
        df = pd.read_csv(path, parse_dates=["date"])
        assert len(df) == 90
        assert "yhat" in df.columns
        assert (df["yhat"] >= 0).all()

    def test_prophet_forecast_saved(self):
        path = self.MODELS_DIR / "prophet_forecast.csv"
        assert path.exists()
        df = pd.read_csv(path, parse_dates=["date"])
        assert len(df) == 90
        assert {"yhat", "yhat_lower", "yhat_upper"}.issubset(df.columns)
        assert (df["yhat_upper"] >= df["yhat"]).all()
        assert (df["yhat"] >= df["yhat_lower"]).all()

    def test_ensemble_forecast_saved(self):
        path = self.MODELS_DIR / "ensemble_forecast.csv"
        assert path.exists()
        df = pd.read_csv(path, parse_dates=["date"])
        assert "yhat_ensemble" in df.columns
        assert (df["yhat_ensemble"] >= 0).all()

    def test_model_comparison_saved(self):
        path = self.MODELS_DIR / "model_comparison.json"
        assert path.exists()
        import json
        with open(path) as f:
            comp = json.load(f)
        assert "models" in comp
        assert "winner" in comp
        assert {"xgboost", "prophet", "ensemble"}.issubset(comp["models"].keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
