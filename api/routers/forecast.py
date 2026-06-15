"""
routers/forecast.py
POST /api/v1/forecast  — returns point forecasts ± uncertainty bands
GET  /api/v1/forecast  — convenience GET with query params
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd

from api.model_registry import ModelRegistry, get_registry
from api.schemas import (
    ForecastPoint, ForecastRequest, ForecastResponse, ModelName
)

router = APIRouter(prefix="/forecast", tags=["Forecast"])


def _build_forecast_points(
    registry: ModelRegistry,
    model: ModelName,
    horizon: int,
) -> list[ForecastPoint]:

    if model == "xgboost":
        df = registry.xgb_forecast.head(horizon).copy()
        return [
            ForecastPoint(
                date=row["date"].date(),
                yhat=round(row["yhat"], 2),
                yhat_lower=None,
                yhat_upper=None,
                model="xgboost",
            )
            for _, row in df.iterrows()
        ]

    elif model == "prophet":
        df = registry.prophet_forecast.head(horizon).copy()
        return [
            ForecastPoint(
                date=row["date"].date(),
                yhat=round(row["yhat"], 2),
                yhat_lower=round(row["yhat_lower"], 2),
                yhat_upper=round(row["yhat_upper"], 2),
                model="prophet",
            )
            for _, row in df.iterrows()
        ]

    elif model == "ensemble":
        df = registry.ensemble_forecast.head(horizon).copy()
        return [
            ForecastPoint(
                date=row["date"].date(),
                yhat=round(row["yhat_ensemble"], 2),
                yhat_lower=round(row.get("ensemble_lower", row["yhat_ensemble"] * 0.85), 2),
                yhat_upper=round(row.get("ensemble_upper", row["yhat_ensemble"] * 1.15), 2),
                model="ensemble",
            )
            for _, row in df.iterrows()
        ]

    raise HTTPException(status_code=400, detail=f"Unknown model: {model}")


@router.post("", response_model=ForecastResponse, summary="Get sales forecast")
def get_forecast_post(
    req: ForecastRequest,
    registry: ModelRegistry = Depends(get_registry),
):
    """
    Returns a daily sales forecast for the requested model and horizon.

    - **model**: `xgboost` | `prophet` | `ensemble` (default: ensemble)
    - **horizon**: number of days to forecast, 1–90 (default: 90)

    Prophet and ensemble responses include `yhat_lower` / `yhat_upper`
    uncertainty bands at the 90% interval.
    """
    if registry.ensemble_forecast is None:
        raise HTTPException(status_code=503, detail="Forecast data not loaded")

    points = _build_forecast_points(registry, req.model, req.horizon)

    return ForecastResponse(
        model=req.model,
        horizon=req.horizon,
        generated_at=datetime.now(timezone.utc).isoformat(),
        forecast=points,
    )


@router.get("", response_model=ForecastResponse, summary="Get sales forecast (GET)")
def get_forecast_get(
    model: ModelName = Query("ensemble", description="Model to use"),
    horizon: int     = Query(90, ge=1, le=90, description="Days to forecast"),
    registry: ModelRegistry = Depends(get_registry),
):
    """Convenience GET endpoint — same response as POST /forecast."""
    if registry.ensemble_forecast is None:
        raise HTTPException(status_code=503, detail="Forecast data not loaded")

    points = _build_forecast_points(registry, model, horizon)

    return ForecastResponse(
        model=model,
        horizon=horizon,
        generated_at=datetime.now(timezone.utc).isoformat(),
        forecast=points,
    )
