"""
routers/analytics.py
GET /api/v1/actuals   — historical daily/weekly/monthly revenue
GET /api/v1/kpis      — headline KPI card values
GET /api/v1/segments  — revenue breakdown by customer segment
GET /api/v1/insights  — auto-generated business insights
GET /api/v1/models    — model comparison metrics
"""

from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.model_registry import ModelRegistry, get_registry
from api.schemas import (
    ActualsResponse, DailyPoint,
    InsightsResponse, Insight,
    KPIResponse,
    ModelComparisonResponse, ModelMetrics,
    SegmentBreakdown, SegmentsResponse,
)

router = APIRouter(tags=["Analytics"])


# ─────────────────────────────────────────────
# /actuals
# ─────────────────────────────────────────────
@router.get("/actuals", response_model=ActualsResponse, summary="Historical revenue")
def get_actuals(
    start_date:  Optional[date] = Query(None, description="Start date (inclusive)"),
    end_date:    Optional[date] = Query(None, description="End date (inclusive)"),
    granularity: str            = Query("daily", pattern="^(daily|weekly|monthly)$"),
    registry:    ModelRegistry  = Depends(get_registry),
):
    """
    Returns historical daily revenue from the processed feature matrix.

    Supports aggregation to **weekly** or **monthly** granularity.
    Defaults to the last 180 days if no date range is provided.
    """
    df = registry.daily_features
    if df is None:
        raise HTTPException(503, "Feature data not loaded")

    # Default window: last 180 days
    if start_date is None:
        start_date = (df["date"].max() - pd.Timedelta(days=180)).date()
    if end_date is None:
        end_date = df["date"].max().date()

    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    filtered = df[mask].copy()

    if filtered.empty:
        raise HTTPException(404, "No data for the specified date range")

    # Aggregate
    if granularity == "weekly":
        filtered["period"] = filtered["date"].dt.to_period("W").dt.start_time
        agg = filtered.groupby("period")["revenue"].sum().reset_index()
        agg.columns = ["date", "value"]
    elif granularity == "monthly":
        filtered["period"] = filtered["date"].dt.to_period("M").dt.start_time
        agg = filtered.groupby("period")["revenue"].sum().reset_index()
        agg.columns = ["date", "value"]
    else:
        agg = filtered[["date", "revenue"]].rename(columns={"revenue": "value"})

    points = [DailyPoint(date=r["date"].date(), value=round(r["value"], 2))
              for _, r in agg.iterrows()]

    return ActualsResponse(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        points=points,
        total=round(float(agg["value"].sum()), 2),
        mean=round(float(agg["value"].mean()), 2),
        std=round(float(agg["value"].std()), 2),
    )


# ─────────────────────────────────────────────
# /kpis
# ─────────────────────────────────────────────
@router.get("/kpis", response_model=KPIResponse, summary="Headline KPIs")
def get_kpis(registry: ModelRegistry = Depends(get_registry)):
    """
    Returns headline KPI card values for the dashboard:
    total revenue, forecast revenue (90d), GP margin, recurring share, etc.
    """
    df  = registry.daily_features
    txn = registry.transactions
    ens = registry.ensemble_forecast

    if df is None or txn is None:
        raise HTTPException(503, "Data not loaded")

    forecast_90d = float(ens["yhat_ensemble"].sum()) if ens is not None else 0.0

    return KPIResponse(
        total_revenue=round(float(df["revenue"].sum()), 2),
        forecast_revenue_90d=round(forecast_90d, 2),
        avg_daily_revenue=round(float(df["revenue"].mean()), 2),
        revenue_std=round(float(df["revenue"].std()), 2),
        gp_margin_pct=round(float(df["gp_margin"].mean() * 100), 2),
        recurring_share_pct=round(float(df["recurring_share"].mean() * 100), 2),
        discount_rate_pct=round(float(df["avg_discount"].mean() * 100), 2),
        n_customers=int(txn["customer_id"].nunique()),
        n_products=int(txn["product_id"].nunique()),
        n_transactions=int(len(txn)),
        date_range_start=df["date"].min().date(),
        date_range_end=df["date"].max().date(),
    )


# ─────────────────────────────────────────────
# /segments
# ─────────────────────────────────────────────
@router.get("/segments", response_model=SegmentsResponse, summary="Revenue by segment")
def get_segments(
    start_date: Optional[date] = Query(None),
    end_date:   Optional[date] = Query(None),
    registry:   ModelRegistry  = Depends(get_registry),
):
    """Revenue breakdown by customer segment (Consumer / Corporate / Home Office)."""
    txn = registry.transactions
    if txn is None:
        raise HTTPException(503, "Transaction data not loaded")

    filtered = txn.copy()
    if start_date:
        filtered = filtered[filtered["date"].dt.date >= start_date]
    if end_date:
        filtered = filtered[filtered["date"].dt.date <= end_date]

    agg = (
        filtered.groupby("segment")
        .agg(revenue=("revenue", "sum"), n_transactions=("transaction_id", "count"))
        .reset_index()
    )
    total = float(agg["revenue"].sum())
    agg["share_pct"] = (agg["revenue"] / total * 100).round(2)

    segments = [
        SegmentBreakdown(
            segment=r["segment"],
            revenue=round(float(r["revenue"]), 2),
            share_pct=float(r["share_pct"]),
            n_transactions=int(r["n_transactions"]),
        )
        for _, r in agg.sort_values("revenue", ascending=False).iterrows()
    ]

    return SegmentsResponse(segments=segments, total_revenue=round(total, 2))


# ─────────────────────────────────────────────
# /insights
# ─────────────────────────────────────────────
def _generate_insights(df: pd.DataFrame, ens: pd.DataFrame) -> list[Insight]:
    insights = []

    # 1. Trend — compare last 30d vs prior 30d
    last_date = df["date"].max()
    recent = df[df["date"] >= last_date - pd.Timedelta(days=30)]["revenue"]
    prior  = df[(df["date"] >= last_date - pd.Timedelta(days=60)) &
                (df["date"] <  last_date - pd.Timedelta(days=30))]["revenue"]

    if len(recent) > 0 and len(prior) > 0:
        pct_change = (recent.mean() - prior.mean()) / prior.mean() * 100
        direction  = "up" if pct_change > 0 else "down"
        insights.append(Insight(
            type="trend" if pct_change > 0 else "risk",
            title=f"Revenue trending {direction} {abs(pct_change):.1f}%",
            detail=(
                f"Average daily revenue over the last 30 days is "
                f"${recent.mean():,.0f}, compared to ${prior.mean():,.0f} "
                f"in the prior 30-day period."
            ),
            metric="revenue_30d_change_pct",
            value=round(pct_change, 2),
        ))

    # 2. Discount pressure
    recent_disc = df[df["date"] >= last_date - pd.Timedelta(days=30)]["avg_discount"].mean()
    if recent_disc > 0.15:
        insights.append(Insight(
            type="risk",
            title=f"High discount intensity: {recent_disc*100:.1f}%",
            detail=(
                "Average transaction discount over the last 30 days exceeds 15%. "
                "Heavy discounting may be compressing gross profit margin."
            ),
            metric="avg_discount_30d",
            value=round(recent_disc * 100, 2),
        ))

    # 3. Forecast vs recent actuals
    if ens is not None and len(ens) > 0:
        forecast_mean = ens["yhat_ensemble"].mean()
        actual_mean   = df.tail(30)["revenue"].mean()
        fcast_uplift  = (forecast_mean - actual_mean) / actual_mean * 100
        insights.append(Insight(
            type="opportunity" if fcast_uplift > 0 else "risk",
            title=f"90-day forecast {'above' if fcast_uplift > 0 else 'below'} recent run-rate",
            detail=(
                f"The ensemble model projects average daily revenue of "
                f"${forecast_mean:,.0f} over the next 90 days, "
                f"{'up' if fcast_uplift > 0 else 'down'} "
                f"{abs(fcast_uplift):.1f}% from the recent 30-day average of "
                f"${actual_mean:,.0f}."
            ),
            metric="forecast_vs_actuals_pct",
            value=round(fcast_uplift, 2),
        ))

    # 4. Weekday vs weekend revenue
    df["dow"] = df["date"].dt.dayofweek
    weekday_rev = df[df["dow"] < 5]["revenue"].mean()
    weekend_rev = df[df["dow"] >= 5]["revenue"].mean()
    wk_ratio    = weekday_rev / max(weekend_rev, 1)
    if wk_ratio > 1.5:
        insights.append(Insight(
            type="info",
            title=f"Weekday revenue {wk_ratio:.1f}× higher than weekends",
            detail=(
                f"Average weekday revenue is ${weekday_rev:,.0f} vs "
                f"${weekend_rev:,.0f} on weekends. "
                "Campaign scheduling should prioritise weekdays."
            ),
            metric="weekday_weekend_ratio",
            value=round(wk_ratio, 2),
        ))

    # 5. Q4 seasonality warning / opportunity
    from datetime import date as date_type
    today_month = last_date.month
    if today_month in [8, 9]:
        insights.append(Insight(
            type="opportunity",
            title="Q4 seasonal uplift approaching",
            detail=(
                "Historical data shows consistent Q4 uplift (Oct–Dec). "
                "Consider increasing inventory and marketing spend ahead of the peak period."
            ),
            metric=None,
            value=None,
        ))

    return insights


@router.get("/insights", response_model=InsightsResponse, summary="Business insights")
def get_insights(registry: ModelRegistry = Depends(get_registry)):
    """Auto-generated business insights derived from actuals + forecast trends."""
    df  = registry.daily_features
    ens = registry.ensemble_forecast

    if df is None:
        raise HTTPException(503, "Feature data not loaded")

    insights = _generate_insights(df.copy(), ens)

    return InsightsResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        insights=insights,
    )


# ─────────────────────────────────────────────
# /models
# ─────────────────────────────────────────────
@router.get("/models", response_model=ModelComparisonResponse, summary="Model comparison")
def get_models(registry: ModelRegistry = Depends(get_registry)):
    """Returns RMSE / MAE / MAPE metrics and feature importance for all trained models."""
    if registry.model_comparison is None:
        raise HTTPException(503, "Model comparison data not loaded")

    comp = registry.model_comparison
    models_out = {}
    for name, m in comp["models"].items():
        models_out[name] = ModelMetrics(
            holdout_rmse  = m.get("holdout_rmse"),
            holdout_mae   = m.get("holdout_mae"),
            holdout_mape  = m.get("holdout_mape"),
            cv_rmse_mean  = m.get("cv_rmse_mean"),
            cv_rmse_std   = m.get("cv_rmse_std"),
            cv_mae_mean   = m.get("cv_mae_mean"),
            weight        = m.get("weight"),
            top_features  = m.get("top_features"),
        )

    return ModelComparisonResponse(
        winner=comp["winner"],
        forecast_horizon_days=comp["forecast_horizon_days"],
        models=models_out,
    )
