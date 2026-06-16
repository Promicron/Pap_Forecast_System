"""
model_registry.py
Loads all ML artefacts once at startup and exposes them as a singleton.
Avoids reloading XGBoost / Prophet on every request.
"""

from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
import json

import pandas as pd
from xgboost import XGBRegressor

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"


class ModelRegistry:
    """Singleton that holds all loaded model artefacts."""

    def __init__(self):
        self.xgb_model:         Optional[XGBRegressor] = None
        self.xgb_forecast:      Optional[pd.DataFrame] = None
        self.prophet_forecast:  Optional[pd.DataFrame] = None
        self.ensemble_forecast: Optional[pd.DataFrame] = None
        self.model_comparison:  Optional[dict] = None
        self.daily_features:    Optional[pd.DataFrame] = None
        self.transactions:      Optional[pd.DataFrame] = None
        self._loaded: list[str] = []

    def load(self):
        log.info("Loading model registry...")

        # XGBoost
        try:
            self.xgb_model = XGBRegressor()
            self.xgb_model.load_model(str(MODELS_DIR / "xgb_model.json"))
            self._loaded.append("xgboost")
            log.info("  ✓ XGBoost model loaded")
        except Exception as e:
            log.error(f"  ✗ XGBoost load failed: {e}")

        # StatsForecast (MSTL+AutoARIMA) — forecast is loaded as CSV below;
        # the model object is not persisted between pipeline runs.

        # Forecast CSVs
        try:
            self.xgb_forecast = pd.read_csv(
                MODELS_DIR / "xgb_forecast.csv", parse_dates=["date"]
            )
            self.prophet_forecast = pd.read_csv(
                MODELS_DIR / "prophet_forecast.csv", parse_dates=["date"]
            )
            self.ensemble_forecast = pd.read_csv(
                MODELS_DIR / "ensemble_forecast.csv", parse_dates=["date"]
            )
            self._loaded.append("forecasts")
            log.info("  ✓ Forecast CSVs loaded")
        except Exception as e:
            log.error(f"  ✗ Forecast CSV load failed: {e}")

        # Model comparison report
        try:
            with open(MODELS_DIR / "model_comparison.json") as f:
                self.model_comparison = json.load(f)
            log.info("  ✓ Model comparison loaded")
        except Exception as e:
            log.error(f"  ✗ Model comparison load failed: {e}")

        # Feature data (for actuals + insights)
        try:
            self.daily_features = pd.read_csv(
                DATA_DIR / "daily_features.csv", parse_dates=["date"]
            ).sort_values("date").reset_index(drop=True)
            self._loaded.append("features")
            log.info(f"  ✓ Daily features loaded ({len(self.daily_features)} rows)")
        except Exception as e:
            log.error(f"  ✗ Daily features load failed: {e}")

        # Transactions (for segment breakdown)
        try:
            self.transactions = pd.read_csv(
                DATA_DIR / "transactions_clean.csv", parse_dates=["date"]
            )
            self._loaded.append("transactions")
            log.info(f"  ✓ Transactions loaded ({len(self.transactions)} rows)")
        except Exception as e:
            log.error(f"  ✗ Transactions load failed: {e}")

        log.info(f"Registry ready. Loaded: {self._loaded}")
        return self

    @property
    def models_loaded(self) -> list[str]:
        return self._loaded


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry().load()
