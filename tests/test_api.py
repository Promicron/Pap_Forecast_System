"""
test_api.py
Integration tests for all SalesCast API endpoints.
Uses FastAPI's TestClient — no server process needed.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Health ───────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self):
        r = client.get("/health")
        body = r.json()
        assert "status" in body
        assert "version" in body
        assert "models_loaded" in body
        assert "data_rows" in body
        assert body["data_rows"] > 0

    def test_health_status_ok(self):
        r = client.get("/health")
        assert r.json()["status"] in ("ok", "degraded")


# ── Forecast ─────────────────────────────────────────────────────────────────
class TestForecast:
    def test_get_forecast_default(self):
        r = client.get("/api/v1/forecast")
        assert r.status_code == 200

    def test_forecast_response_schema(self):
        r = client.get("/api/v1/forecast?model=ensemble&horizon=30")
        body = r.json()
        assert "forecast" in body
        assert "model" in body
        assert "horizon" in body
        assert body["horizon"] == 30
        assert len(body["forecast"]) == 30

    def test_forecast_point_schema(self):
        r = client.get("/api/v1/forecast?model=ensemble&horizon=7")
        point = r.json()["forecast"][0]
        assert "date" in point
        assert "yhat" in point
        assert "model" in point
        assert point["yhat"] >= 0

    def test_forecast_xgboost(self):
        r = client.get("/api/v1/forecast?model=xgboost&horizon=14")
        assert r.status_code == 200
        assert r.json()["model"] == "xgboost"
        assert len(r.json()["forecast"]) == 14

    def test_forecast_prophet(self):
        r = client.get("/api/v1/forecast?model=prophet&horizon=14")
        assert r.status_code == 200
        body = r.json()
        # Prophet returns uncertainty bounds
        point = body["forecast"][0]
        assert point["yhat_lower"] is not None
        assert point["yhat_upper"] is not None
        assert point["yhat_upper"] >= point["yhat"] >= point["yhat_lower"]

    def test_forecast_post_endpoint(self):
        r = client.post("/api/v1/forecast", json={"model": "ensemble", "horizon": 10})
        assert r.status_code == 200
        assert len(r.json()["forecast"]) == 10

    def test_forecast_horizon_validation(self):
        r = client.get("/api/v1/forecast?horizon=91")
        assert r.status_code == 422  # exceeds max 90

    def test_forecast_invalid_model(self):
        r = client.get("/api/v1/forecast?model=gpt5")
        assert r.status_code == 422

    def test_forecast_dates_are_sequential(self):
        r = client.get("/api/v1/forecast?horizon=30")
        dates = [p["date"] for p in r.json()["forecast"]]
        assert dates == sorted(dates)
        assert len(set(dates)) == 30  # no duplicates

    def test_forecast_no_negative_yhat(self):
        r = client.get("/api/v1/forecast?horizon=90")
        for pt in r.json()["forecast"]:
            assert pt["yhat"] >= 0


# ── Actuals ───────────────────────────────────────────────────────────────────
class TestActuals:
    def test_actuals_default(self):
        r = client.get("/api/v1/actuals")
        assert r.status_code == 200

    def test_actuals_schema(self):
        r = client.get("/api/v1/actuals")
        body = r.json()
        for key in ("start_date", "end_date", "granularity", "points", "total", "mean", "std"):
            assert key in body

    def test_actuals_points_non_empty(self):
        r = client.get("/api/v1/actuals")
        assert len(r.json()["points"]) > 0

    def test_actuals_granularity_monthly(self):
        r = client.get("/api/v1/actuals?granularity=monthly")
        assert r.status_code == 200
        # Monthly should have fewer points than daily
        r_daily = client.get("/api/v1/actuals?granularity=daily")
        assert len(r.json()["points"]) <= len(r_daily.json()["points"])

    def test_actuals_granularity_weekly(self):
        r = client.get("/api/v1/actuals?granularity=weekly")
        assert r.status_code == 200

    def test_actuals_invalid_granularity(self):
        r = client.get("/api/v1/actuals?granularity=quarterly")
        assert r.status_code == 422

    def test_actuals_revenue_positive(self):
        r = client.get("/api/v1/actuals")
        for pt in r.json()["points"]:
            assert pt["value"] > 0


# ── KPIs ──────────────────────────────────────────────────────────────────────
class TestKPIs:
    def test_kpis_200(self):
        r = client.get("/api/v1/kpis")
        assert r.status_code == 200

    def test_kpis_schema(self):
        body = client.get("/api/v1/kpis").json()
        required = [
            "total_revenue", "forecast_revenue_90d", "avg_daily_revenue",
            "gp_margin_pct", "recurring_share_pct", "discount_rate_pct",
            "n_customers", "n_products", "n_transactions",
        ]
        for key in required:
            assert key in body

    def test_kpis_sanity(self):
        body = client.get("/api/v1/kpis").json()
        assert body["total_revenue"] > 0
        assert body["n_customers"] > 0
        assert body["n_transactions"] > 0
        assert 0 <= body["gp_margin_pct"] <= 100
        assert 0 <= body["discount_rate_pct"] <= 100


# ── Segments ──────────────────────────────────────────────────────────────────
class TestSegments:
    def test_segments_200(self):
        r = client.get("/api/v1/segments")
        assert r.status_code == 200

    def test_segments_share_sums_to_100(self):
        body = client.get("/api/v1/segments").json()
        total_share = sum(s["share_pct"] for s in body["segments"])
        assert abs(total_share - 100.0) < 0.1

    def test_known_superstore_segments(self):
        body = client.get("/api/v1/segments").json()
        names = {s["segment"] for s in body["segments"]}
        assert names.issubset({"Consumer", "Corporate", "Home Office"})


# ── Insights ──────────────────────────────────────────────────────────────────
class TestInsights:
    def test_insights_200(self):
        r = client.get("/api/v1/insights")
        assert r.status_code == 200

    def test_insights_schema(self):
        body = client.get("/api/v1/insights").json()
        assert "insights" in body
        assert "generated_at" in body
        assert len(body["insights"]) > 0

    def test_insight_types_valid(self):
        body = client.get("/api/v1/insights").json()
        valid_types = {"trend", "risk", "opportunity", "info"}
        for ins in body["insights"]:
            assert ins["type"] in valid_types
            assert len(ins["title"]) > 0
            assert len(ins["detail"]) > 0


# ── Models ────────────────────────────────────────────────────────────────────
class TestModels:
    def test_models_200(self):
        r = client.get("/api/v1/models")
        assert r.status_code == 200

    def test_models_schema(self):
        body = client.get("/api/v1/models").json()
        assert "winner" in body
        assert "models" in body
        assert set(body["models"].keys()) == {"xgboost", "prophet", "ensemble"}

    def test_models_metrics_present(self):
        body = client.get("/api/v1/models").json()
        for name, m in body["models"].items():
            assert "holdout_rmse" in m
            assert "holdout_mae" in m
            assert m["holdout_rmse"] > 0

    def test_models_winner_valid(self):
        body = client.get("/api/v1/models").json()
        assert body["winner"] in {"xgboost", "prophet", "ensemble"}


# ── Response headers ──────────────────────────────────────────────────────────
class TestHeaders:
    def test_timing_header_present(self):
        r = client.get("/health")
        assert "x-response-time-ms" in r.headers

    def test_cors_header(self):
        r = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in r.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
