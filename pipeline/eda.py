"""
Exploratory Data Analysis for the sales pipeline.
Produces a written summary report + key statistics.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def run_eda(save_report: bool = True) -> dict:
    txn       = pd.read_csv(DATA_DIR / "transactions_clean.csv", parse_dates=["date"])
    customers = pd.read_csv(DATA_DIR / "customers_clean.csv")
    products  = pd.read_csv(DATA_DIR / "products_clean.csv")
    daily     = pd.read_csv(DATA_DIR / "daily_features.csv",     parse_dates=["date"])

    report = {}

    #  Overview 
    report["overview"] = {
        "total_transactions": len(txn),
        "total_revenue":      round(txn["revenue"].sum(), 2),
        "total_customers":    txn["customer_id"].nunique(),
        "total_products":     txn["product_id"].nunique(),
        "date_range":         f"{txn['date'].min().date()} to {txn['date'].max().date()}",
        "avg_daily_revenue":  round(daily["revenue"].mean(), 2),
        "median_daily_revenue": round(daily["revenue"].median(), 2),
        "revenue_std":        round(daily["revenue"].std(), 2),
    }

    #  Revenue by segment 
    report["by_segment"] = (
        txn.groupby("segment")["revenue"]
        .agg(total="sum", avg_order="mean", count="count")
        .round(2)
        .to_dict(orient="index")
    )

    #  Revenue by category 
    report["by_category"] = (
        txn.groupby("category")["revenue"]
        .agg(total="sum", avg_order="mean", count="count")
        .round(2)
        .to_dict(orient="index")
    )

    #  Monthly revenue trend 
    monthly = txn.copy()
    monthly["month"] = txn["date"].dt.to_period("M")
    report["monthly_revenue"] = {
        str(k): round(float(v), 2)
        for k, v in monthly.groupby("month")["revenue"].sum().items()
    }

    #  Seasonality indicators 
    dow_rev = daily.copy()
    dow_rev["dow_name"] = pd.to_datetime(daily["date"]).dt.day_name()
    report["avg_revenue_by_dow"] = (
        dow_rev.groupby("dow_name")["revenue"].mean()
        .round(2)
        .to_dict()
    )

    report["avg_revenue_by_month"] = (
        daily.groupby("month")["revenue"].mean()
        .round(2)
        .to_dict()
    )

    #  Discount analysis 
    report["discount_analysis"] = {
        "avg_discount_pct":    round(txn["discount_pct"].mean() * 100, 2),
        "pct_transactions_discounted": round((txn["discount_pct"] > 0).mean() * 100, 2),
        "revenue_at_zero_discount":    round(txn[txn["discount_pct"] == 0]["revenue"].sum(), 2),
        "revenue_with_discount":       round(txn[txn["discount_pct"] > 0]["revenue"].sum(), 2),
    }

    #  Gross profit margin 
    report["profitability"] = {
        "avg_gp_margin_pct":     round(daily["gp_margin"].mean() * 100, 2),
        "total_gross_profit":    round(txn["gross_profit"].sum(), 2),
        "recurring_rev_share":   round(daily["recurring_share"].mean() * 100, 2),
    }

    #  Top features correlated with revenue ─────────
    feature_cols = [c for c in daily.columns if c not in ["date","revenue"]]
    numeric_cols = daily[feature_cols].select_dtypes(include="number").columns.tolist()
    correlations = daily[numeric_cols + ["revenue"]].corr()["revenue"].drop("revenue")
    top_corr = correlations.abs().sort_values(ascending=False).head(15)
    report["top_correlated_features"] = {
        k: round(float(correlations[k]), 4) for k in top_corr.index
    }

    #  Outlier summary 
    q99 = daily["revenue"].quantile(0.99)
    q01 = daily["revenue"].quantile(0.01)
    report["outliers"] = {
        "p1_daily_revenue":  round(q01, 2),
        "p99_daily_revenue": round(q99, 2),
        "days_above_p99":    int((daily["revenue"] > q99).sum()),
        "days_below_p01":    int((daily["revenue"] < q01).sum()),
    }

    #  Print readable summary 
    ov = report["overview"]
    print(f"""
╔══════════════════════════════════════════╗
║         SALESCAST — EDA SUMMARY          ║
╠══════════════════════════════════════════╣
  Transactions   : {ov['total_transactions']:>10,}
  Revenue (total): £{ov['total_revenue']:>12,.2f}
  Unique customers: {ov['total_customers']:>9,}
  Date range     : {ov['date_range']}
  Avg daily rev  : £{ov['avg_daily_revenue']:>10,.2f}
  Revenue std    : £{ov['revenue_std']:>10,.2f}

  Top correlated features with revenue:""")
    for feat, corr in list(report["top_correlated_features"].items())[:6]:
        bar = "█" * int(abs(corr) * 20)
        direction = "+" if corr > 0 else "–"
        print(f"    {direction} {feat:<28} {corr:+.4f}  {bar}")

    print(f"""
  GP Margin (avg): {report['profitability']['avg_gp_margin_pct']}%
  Recurring rev  : {report['profitability']['recurring_rev_share']}% of total
  Discounted txns: {report['discount_analysis']['pct_transactions_discounted']}%
╚══════════════════════════════════════════╝""")

    if save_report:
        import json
        out = DATA_DIR / "eda_report.json"
        # Convert Period keys to strings for JSON serialisation
        def default(obj):
            if hasattr(obj, "isoformat"):
                return str(obj)
            return str(obj)
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=default)
        log.info(f"EDA report saved to {out}")

    return report


if __name__ == "__main__":
    run_eda()
