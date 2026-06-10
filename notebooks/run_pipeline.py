"""
Orchestrates the full data pipeline end-to-end:
  1. Generate synthetic data
  2. Clean and validate
  3. Feature engineering
  4. EDA summary
"""

import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from notebooks.generate_data import generate
from clean        import run as run_clean
from notebooks.features     import run as run_features
from eda          import run_eda


def main():
    t0 = time.time()
    print("\n" + "═"*50)
    print("  SALESCAST — DATA PIPELINE")
    print("═"*50 + "\n")

    print("STEP 1/4 — Generating synthetic data")
    print("─"*40)
    generate(save=True)

    print("\nSTEP 2/4 — Cleaning & validating")
    print("─"*40)
    run_clean(save=True)

    print("\nSTEP 3/4 — Feature engineering")
    print("─"*40)
    run_features(save=True)

    print("\nSTEP 4/4 — EDA summary")
    print("─"*40)
    run_eda(save_report=True)

    elapsed = time.time() - t0
    print(f"\n{'═'*50}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Output: salescast/data/")
    print("═"*50 + "\n")


if __name__ == "__main__":
    main()
