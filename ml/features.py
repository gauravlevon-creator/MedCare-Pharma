"""
Feature engineering (Nishit's original, kept for analysis and the dashboard).

NOTE: Chronos-2 does NOT consume these features. The production forecast in
ml/predict.py passes only demand_qty plus promotion_flag and season_flag.
Do not describe these features as model inputs.
"""

from __future__ import annotations

import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """lag_1/7/14, rolling_mean_7/14 (using only past values), day_of_week, month."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "warehouse_id", "date"])

    group = df.groupby(["sku_id", "warehouse_id"], group_keys=False)
    df["lag_1"] = group["demand_qty"].shift(1)
    df["lag_7"] = group["demand_qty"].shift(7)
    df["lag_14"] = group["demand_qty"].shift(14)
    df["rolling_mean_7"] = group["demand_qty"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_mean_14"] = group["demand_qty"].transform(lambda x: x.shift(1).rolling(14).mean())
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df.dropna().reset_index(drop=True)
