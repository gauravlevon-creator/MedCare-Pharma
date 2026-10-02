"""
Demand-data validation (Nishit's checks, returning a report instead of printing).
"""

from __future__ import annotations

import pandas as pd

import config


def validate_demand_data(df: pd.DataFrame) -> dict:
    """
    Run Nishit's checks: missing values, negative demand, invalid dates,
    binary promotion/season flags, and chronological order per series.
    Also checks for date gaps and series that do not end on SIMULATION_DATE.

    Returns {"ok": bool, "issues": [...], "summary": {...}}.
    """
    issues: list[str] = []
    df = df.copy()

    missing = int(df.isnull().sum().sum())
    if missing:
        issues.append(f"{missing} missing values")

    negative = int((df["demand_qty"] < 0).sum())
    if negative:
        issues.append(f"{negative} negative demand rows")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(df["date"].isnull().sum())
    if invalid_dates:
        issues.append(f"{invalid_dates} invalid dates")

    for flag in config.COVARIATE_COLUMNS:
        bad = set(df[flag].dropna().unique()) - {0, 1}
        if bad:
            issues.append(f"{flag} has non-binary values {sorted(bad)}")

    df = df.dropna(subset=["date"]).sort_values(["sku_id", "warehouse_id", "date"])
    grouped = df.groupby(["sku_id", "warehouse_id"])["date"]
    spans = grouped.agg(["min", "max", "count"])
    spans["expected"] = (spans["max"] - spans["min"]).dt.days + 1
    gaps = spans[spans["expected"] != spans["count"]]
    if len(gaps):
        issues.append(f"{len(gaps)} series have missing or duplicate days")

    stale = spans[spans["max"] != config.simulation_timestamp()]
    if len(stale):
        issues.append(f"{len(stale)} series do not end on SIMULATION_DATE {config.SIMULATION_DATE}")

    summary = {
        "rows": int(len(df)),
        "series": int(len(spans)),
        "skus": int(df["sku_id"].nunique()),
        "warehouses": int(df["warehouse_id"].nunique()),
        "start_date": str(df["date"].min().date()) if len(df) else None,
        "end_date": str(df["date"].max().date()) if len(df) else None,
        "zero_demand_days": int((df["demand_qty"] == 0).sum()),
    }
    return {"ok": not issues, "issues": issues, "summary": summary}


if __name__ == "__main__":
    from ml.data_loader import load_demand_data

    report = validate_demand_data(load_demand_data())
    print("Validation OK:", report["ok"])
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    for issue in report["issues"]:
        print("  ISSUE:", issue)
