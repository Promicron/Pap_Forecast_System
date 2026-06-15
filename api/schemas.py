"""
schemas.py
Pydantic v2 request and response models for the SalesCast API.
All monetary values in USD (Superstore dataset).
"""

from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Shared primitives
# ─────────────────────────────────────────────
ModelName = Literal["xgboost", "prophet", "ensemble"]


class DailyPoint(BaseModel):
    date:  date
    value: float = Field(..., description="Daily revenue in USD")


class ForecastPoint(BaseModel):
    date:        date
    yhat:        float  = Field(..., description="Point forecast")
    yhat_lower:  Optional[float] = Field(None, description="Lower uncertainty bound (90%)")
    yhat_upper:  Optional[float] = Field(None, description="Upper uncertainty bound (90%)")
    model:       ModelName


# ─────────────────────────────────────────────
# /forecast
# ─────────────────────────────────────────────
class ForecastRequest(BaseModel):
    model:   ModelName  = Field("ensemble", description="Which model to serve")
    horizon: int        = Field(90, ge=1, le=90, description="Days to forecast (max 90)")

    @field_validator("horizon")
    @classmethod
    def horizon_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("horizon must be ≥ 1")
        return v


class ForecastResponse(BaseModel):
    model:        ModelName
    horizon:      int
    generated_at: str
    forecast:     list[ForecastPoint]


# ─────────────────────────────────────────────
# /actuals
# ─────────────────────────────────────────────
class ActualsRequest(BaseModel):
    start_date: Optional[date] = None
    end_date:   Optional[date] = None
    granularity: Literal["daily", "weekly", "monthly"] = "daily"

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v, info):
        start = info.data.get("start_date")
        if start and v and v < start:
            raise ValueError("end_date must be after start_date")
        return v


class ActualsResponse(BaseModel):
    start_date:  date
    end_date:    date
    granularity: str
    points:      list[DailyPoint]
    total:       float
    mean:        float
    std:         float


# ─────────────────────────────────────────────
# /kpis
# ─────────────────────────────────────────────
class KPIResponse(BaseModel):
    total_revenue:       float
    forecast_revenue_90d: float
    avg_daily_revenue:   float
    revenue_std:         float
    gp_margin_pct:       float
    recurring_share_pct: float
    discount_rate_pct:   float
    n_customers:         int
    n_products:          int
    n_transactions:      int
    date_range_start:    date
    date_range_end:      date


# ─────────────────────────────────────────────
# /models
# ─────────────────────────────────────────────
class ModelMetrics(BaseModel):
    holdout_rmse:  float
    holdout_mae:   float
    holdout_mape:  Optional[float]
    cv_rmse_mean:  Optional[float]
    cv_rmse_std:   Optional[float]
    cv_mae_mean:   Optional[float]
    weight:        Optional[float]
    top_features:  Optional[list[str]]


class ModelComparisonResponse(BaseModel):
    winner:                 ModelName
    forecast_horizon_days:  int
    models:                 dict[str, ModelMetrics]


# ─────────────────────────────────────────────
# /insights
# ─────────────────────────────────────────────
class Insight(BaseModel):
    type:    Literal["trend", "risk", "opportunity", "info"]
    title:   str
    detail:  str
    metric:  Optional[str] = None
    value:   Optional[float] = None


class InsightsResponse(BaseModel):
    generated_at: str
    insights:     list[Insight]


# ─────────────────────────────────────────────
# /segments
# ─────────────────────────────────────────────
class SegmentBreakdown(BaseModel):
    segment:  str
    revenue:  float
    share_pct: float
    n_transactions: int


class SegmentsResponse(BaseModel):
    segments: list[SegmentBreakdown]
    total_revenue: float


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:       Literal["ok", "degraded"]
    version:      str
    models_loaded: list[str]
    data_rows:    int
    last_data_date: Optional[date]
