"""
ML service (owner: Ruthie): the single entry point the rest of the system
uses to get forecasts.

Chronos-2 on CPU is slow, so the 180-series forecast is saved to
outputs/forecast_output.csv with a metadata file. The cache is reused only
when the simulation date, horizon, model, covariate strategy and demand
data all match; otherwise it is regenerated. Streamlit should read
forecasts through this module, never by calling Chronos-2 directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

import config
from ml import data_loader
from ml.predict import ForecastError, generate_forecast, generate_forecasts_batch

logger = logging.getLogger(__name__)


def _demand_fingerprint(demand: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(
        demand[["date", "sku_id", "warehouse_id", "demand_qty", *config.COVARIATE_COLUMNS]]
        .sort_values(["sku_id", "warehouse_id", "date"]).reset_index(drop=True),
        index=False,
    )
    return f"{len(demand)}-{int(hashed.sum()) & 0xFFFFFFFFFFFF:x}"


def _expected_metadata(demand: pd.DataFrame) -> dict:
    return {
        "model_name": config.CHRONOS_MODEL_NAME,
        "simulation_date": config.SIMULATION_DATE,
        "forecast_horizon_days": config.FORECAST_HORIZON_DAYS,
        "future_covariate_strategy": config.FUTURE_COVARIATE_STRATEGY,
        "point_forecast_quantile": config.POINT_FORECAST_QUANTILE,
        "demand_fingerprint": _demand_fingerprint(demand),
    }


def _load_cache(expected: dict) -> tuple[pd.DataFrame, dict] | None:
    if not (config.FORECAST_OUTPUT_CSV.exists() and config.FORECAST_METADATA_JSON.exists()):
        return None
    try:
        meta = json.loads(config.FORECAST_METADATA_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Forecast metadata unreadable; regenerating")
        return None
    if any(meta.get(k) != v for k, v in expected.items()):
        logger.info("Forecast cache is stale (config or data changed); regenerating")
        return None
    df = pd.read_csv(config.FORECAST_OUTPUT_CSV, parse_dates=["date"],
                     dtype={"sku_id": str, "warehouse_id": str})
    return df, meta


def get_all_forecasts(refresh: bool = False, pipeline=None) -> tuple[pd.DataFrame, dict]:
    """
    Forecasts for every SKU + warehouse with demand history.

    Returns (forecasts, metadata). metadata["errors"] lists series that
    could not be forecast; they are absent from the DataFrame.
    """
    demand = data_loader.load_demand_data()
    expected = _expected_metadata(demand)
    if not refresh:
        cached = _load_cache(expected)
        if cached is not None:
            logger.info("Using cached forecasts from %s", config.FORECAST_OUTPUT_CSV)
            return cached

    forecasts, errors = generate_forecasts_batch(demand=demand, pipeline=pipeline)
    meta = {
        **expected,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series_forecast": int(forecasts[["sku_id", "warehouse_id"]].drop_duplicates().shape[0]),
        "rows": int(len(forecasts)),
        "errors": {f"{s}/{w}": msg for (s, w), msg in errors.items()},
        # Which predictor produced these numbers ("Chronos2Pipeline" for the real model;
        # anything else is a test stand-in and must not be reported as a real result).
        "pipeline": type(pipeline).__name__ if pipeline is not None else "Chronos2Pipeline",
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = forecasts.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(config.FORECAST_OUTPUT_CSV, index=False)
    config.FORECAST_METADATA_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved %d forecast rows (%d errors)", len(forecasts), len(errors))
    return forecasts, meta


def get_forecast(sku_id: str, warehouse_id: str, refresh: bool = False,
                 pipeline=None) -> pd.DataFrame:
    """
    Forecast for one SKU + warehouse. Uses the saved 180-series forecast
    when it is valid; otherwise runs Chronos-2 for this series only.
    Raises ForecastError if no forecast can be produced.
    """
    if not refresh:
        demand = data_loader.load_demand_data()
        cached = _load_cache(_expected_metadata(demand))
        if cached is not None:
            df, meta = cached
            key = f"{sku_id}/{warehouse_id}"
            if key in meta.get("errors", {}):
                raise ForecastError(f"{key}: {meta['errors'][key]}")
            rows = df[(df["sku_id"] == sku_id) & (df["warehouse_id"] == warehouse_id)]
            if len(rows) == config.FORECAST_HORIZON_DAYS:
                return rows.sort_values("date").reset_index(drop=True)
    return generate_forecast(sku_id, warehouse_id, pipeline=pipeline)
