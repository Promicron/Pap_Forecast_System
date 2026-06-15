"""
main.py
SalesCast FastAPI application entry point.

Endpoints
─────────
GET  /health                     — liveness + readiness
GET  /api/v1/forecast            — sales forecast (GET)
POST /api/v1/forecast            — sales forecast (POST)
GET  /api/v1/actuals             — historical revenue
GET  /api/v1/kpis                — headline KPI cards
GET  /api/v1/segments            — revenue by segment
GET  /api/v1/insights            — auto-generated insights
GET  /api/v1/models              — model comparison metrics
GET  /docs                       — Swagger UI
GET  /redoc                      — ReDoc
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.model_registry import get_registry
from api.routers import analytics, forecast
from api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_VERSION = "1.0.0"


# ─────────────────────────────────────────────
# Lifespan: load models once at startup
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("SalesCast API starting up...")
    get_registry()          # warm the lru_cache — loads all artefacts
    log.info("SalesCast API ready.")
    yield
    log.info("SalesCast API shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="SalesCast API",
    description=(
        "AI-powered sales forecasting and decision support API.\n\n"
        "Serves XGBoost, Prophet, and ensemble forecasts trained on the "
        "Superstore retail dataset, along with KPIs, segment breakdowns, "
        "and auto-generated business insights."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the React dashboard (dev + prod origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server
        "http://localhost:5173",   # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request timing middleware
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
    return response


# ─────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ─────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness + readiness probe. Returns loaded model names and data row count."""
    registry = get_registry()
    df = registry.daily_features

    status = "ok" if len(registry.models_loaded) >= 3 else "degraded"

    return HealthResponse(
        status=status,
        version=API_VERSION,
        models_loaded=registry.models_loaded,
        data_rows=len(df) if df is not None else 0,
        last_data_date=df["date"].max().date() if df is not None else None,
    )


# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────
app.include_router(forecast.router,  prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")


# ─────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
