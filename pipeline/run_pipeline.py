"""
run_pipeline.py
Orchestrates the full data pipeline end-to-end.

With Kaggle Superstore data (default):
  1. Ingest + map Superstore CSV → pipeline schema
  2. Clean and validate
  3. Feature engineering
  4. EDA summary

With synthetic data (--synthetic flag):
  1. Generate synthetic data
  2–4. Same as above

Run with:
  python run_pipeline.py              # Superstore
  python run_pipeline.py --synthetic  # synthetic
"""

import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from clean    import run as run_clean
from features import run as run_features
from model_xgboost import run as run_xgboost
from model_prophet import run as run_prophet
from model_ensemble import run as run_ensemble
from eda      import run_eda

USE_SYNTHETIC = "--synthetic" in sys.argv


def main():
    t0 = time.time()
    print("\n" + "═"*50)
    print("  SALESCAST — DATA PIPELINE")
    mode = "synthetic" if USE_SYNTHETIC else "Superstore (Kaggle)"
    print(f"  Mode: {mode}")
    print("═"*50 + "\n")

    if USE_SYNTHETIC:
        print("STEP 1/4 — Generating synthetic data")
        print("─"*40)
        from generate_data import generate
        generate(save=True)
    else:
        print("STEP 1/4 — Ingesting Kaggle Superstore data")
        print("─"*40)
        from kaggle_ingest import ingest
        ingest(save=True)

    print("\nSTEP 2/4 — Cleaning & validating")
    print("─"*40)
    run_clean(save=True)

    print("\nSTEP 3/5 — Feature engineering")
    print("─"*40)
    run_features(save=True)

    print("\nSTEP 4/5 — Train forecasting models")
    print("─"*40)
    run_xgboost(save=True)
    run_prophet(save=True)
    run_ensemble(save=True)

    print("\nSTEP 5/5 — EDA summary")
    print("─"*40)
    run_eda(save_report=True)

    elapsed = time.time() - t0
    print(f"\n{'═'*50}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Output: salescast/data/")
    print("═"*50 + "\n")


if __name__ == "__main__":
    main()
