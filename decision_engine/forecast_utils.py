"""
Shared forecast checks for the decision engine.

Every business module uses the same rule: a forecast is accepted only if it
belongs to the requested item, starts the day after SIMULATION_DATE, is a
consecutive daily series, and contains finite, non-negative values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


class ForecastInputError(Exception):
    """The forecast passed to a business calculation is missing or invalid."""


def validated_forecast_values(forecast: pd.DataFrame, sku_id: str, warehouse_id: str) -> np.ndarray:
    """Return forecast_demand as a float array (day 1 = SIMULATION_DATE + 1), or raise."""
    label = f"{sku_id} / {warehouse_id}"
    if forecast is None or len(forecast) == 0:
        raise ForecastInputError(f"Forecast for {label} is missing or empty")
    missing = [c for c in ("date", "sku_id", "warehouse_id", "forecast_demand") if c not in forecast.columns]
    if missing:
        raise ForecastInputError(f"Forecast for {label} is missing columns {missing}")

    fc = forecast.copy()
    if not ((fc["sku_id"] == sku_id) & (fc["warehouse_id"] == warehouse_id)).all():
        raise ForecastInputError(f"Forecast contains rows for items other than {label}")
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    if fc["date"].isna().any():
        raise ForecastInputError(f"Forecast for {label} has invalid dates")
    fc = fc.sort_values("date").reset_index(drop=True)

    start = config.simulation_timestamp() + pd.Timedelta(days=1)
    if fc["date"].iloc[0] != start:
        raise ForecastInputError(
            f"Forecast for {label} starts {fc['date'].iloc[0].date()}, expected {start.date()}"
        )
    if not fc["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ForecastInputError(f"Forecast for {label} is not a consecutive daily series")

    values = pd.to_numeric(fc["forecast_demand"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ForecastInputError(f"Forecast for {label} has missing or non-numeric values")
    if (values < 0).any():
        raise ForecastInputError(f"Forecast for {label} has negative demand")
    return values
