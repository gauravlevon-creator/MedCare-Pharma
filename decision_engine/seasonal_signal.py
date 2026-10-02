"""
Seasonal demand signal (P1 demand sensing).

This is NOT an influenza or epidemiological model. The data has no
flu-case or health-surveillance information, so no flu label is produced.
The signal uses only the historical season_flag and demand_qty in the
demand history, to show when the forecast window matches a period that was
seasonal, with elevated demand, last year.

For each SKU + warehouse and the forecast window W (SIMULATION_DATE+1 ..
+FORECAST_HORIZON_DAYS):
    analog window         = W shifted back one year (365 days)
    analog_season_share   = share of analog days with season_flag = 1
    analog_demand_index   = mean demand in the analog window / the item's mean demand
    seasonal_uplift       = mean demand on season days / mean demand on non-season days
    last_season_flag      = season_flag on SIMULATION_DATE. This is the value
                            Chronos-2 carries forward as its future covariate.

Labels (thresholds in config.py):
    ELEVATED  analog_season_share >= SEASON_ACTIVE_SHARE and
              analog_demand_index >= 1 + SEASON_ELEVATED_UPLIFT
    SEASONAL  analog_season_share >= SEASON_PRESENT_SHARE
    NORMAL    otherwise

covariate_gap = the analog window was seasonal (>= SEASON_ACTIVE_SHARE) but
last_season_flag = 0. Chronos-2's carry-forward fallback then assumes no
season, so its forecast may understate seasonal demand. The signal only
flags this; it does not alter the forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from database import database as db


def classify_seasonal_signal(analog_season_share: float, analog_demand_index: float) -> str:
    if (analog_season_share >= config.SEASON_ACTIVE_SHARE
            and analog_demand_index >= 1 + config.SEASON_ELEVATED_UPLIFT):
        return config.SEASON_SIGNAL_ELEVATED
    if analog_season_share >= config.SEASON_PRESENT_SHARE:
        return config.SEASON_SIGNAL_SEASONAL
    return config.SEASON_SIGNAL_NORMAL


def signal_for_series(history: pd.DataFrame, sku_id: str, warehouse_id: str,
                      forecast_values: np.ndarray | None = None) -> dict:
    """Seasonal signal for one item's demand history (date, demand_qty, season_flag)."""
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["date"] <= config.simulation_timestamp()].sort_values("date")
    start = config.simulation_timestamp() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS - 1)
    a_start, a_end = start - pd.Timedelta(days=365), end - pd.Timedelta(days=365)
    analog = h[(h["date"] >= a_start) & (h["date"] <= a_end)]

    mean_all = float(h["demand_qty"].mean()) if len(h) else 0.0
    share = float(analog["season_flag"].mean()) if len(analog) else 0.0
    index = float(analog["demand_qty"].mean() / mean_all) if len(analog) and mean_all > 0 else 1.0
    on, off = h[h["season_flag"] == 1]["demand_qty"], h[h["season_flag"] == 0]["demand_qty"]
    uplift = float(on.mean() / off.mean()) if len(on) and len(off) and off.mean() > 0 else None
    last_flag = int(h["season_flag"].iloc[-1]) if len(h) else 0
    label = classify_seasonal_signal(share, index) if len(analog) else config.SEASON_SIGNAL_NORMAL
    gap = share >= config.SEASON_ACTIVE_SHARE and last_flag == 0
    fc_index = (float(np.mean(forecast_values)) / mean_all
                if forecast_values is not None and len(forecast_values) and mean_all > 0 else None)

    if not len(analog):
        reason = "No history for the same period last year; signal defaults to NORMAL"
    else:
        reason = (f"Historical seasonal demand pattern: in {a_start.date()}..{a_end.date()} "
                  f"(same period last year) season_flag was active on {share:.0%} of days and demand "
                  f"was {index:.2f}x the item's average")
        if uplift is not None:
            reason += f"; season days average {uplift:.2f}x non-season demand"
    if gap:
        reason += ("; NOTE: season_flag is 0 on the simulation date, so the Chronos-2 forecast "
                   "assumes no season and may understate seasonal demand")
    return {
        "sku_id": sku_id, "warehouse_id": warehouse_id,
        "window_start": start.strftime("%Y-%m-%d"), "window_end": end.strftime("%Y-%m-%d"),
        "analog_start": a_start.strftime("%Y-%m-%d"), "analog_end": a_end.strftime("%Y-%m-%d"),
        "analog_season_share": round(share, 3), "analog_demand_index": round(index, 3),
        "seasonal_uplift": round(uplift, 3) if uplift is not None else None,
        "last_season_flag": last_flag, "covariate_gap": bool(gap),
        "forecast_demand_index": round(fc_index, 3) if fc_index is not None else None,
        "seasonal_signal": label, "seasonal_reason": reason,
    }


def seasonal_signal_for_item(sku_id: str, warehouse_id: str, forecast_values=None, db_path=None) -> dict:
    return signal_for_series(db.get_demand_history(sku_id, warehouse_id, db_path=db_path),
                             sku_id, warehouse_id, forecast_values)


def compute_seasonal_signals(forecasts: pd.DataFrame | None = None, db_path=None) -> pd.DataFrame:
    """Signals for every item. `forecasts` (optional) adds forecast_demand_index."""
    demand = db.get_all_demand(db_path=db_path)
    fc = ({k: g.sort_values("date")["forecast_demand"].to_numpy()
           for k, g in forecasts.groupby(["sku_id", "warehouse_id"])} if forecasts is not None else {})
    rows = [signal_for_series(g, sku, wh, fc.get((sku, wh)))
            for (sku, wh), g in demand.groupby(["sku_id", "warehouse_id"])]
    return pd.DataFrame(rows).sort_values(["sku_id", "warehouse_id"]).reset_index(drop=True)
