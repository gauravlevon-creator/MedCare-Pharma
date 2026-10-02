"""
Demand forecasting with Chronos-2 (Nishit's implementation, integrated).

What is preserved from Nishit's predict.py
------------------------------------------
* Model: amazon/chronos-2 via Chronos2Pipeline.
* Unit: one series per SKU + warehouse.
* Target: daily demand_qty.
* Covariates: promotion_flag and season_flag as past covariates, and the
  last known value carried forward as future covariates (fallback,
  because no future promotion/season calendar exists in the data).
* Point forecast: the median (0.5 quantile) of Chronos-2's output.
* Output columns: date, sku_id, warehouse_id, forecast_demand.

What changed
------------
* History comes from the database and stops at config.SIMULATION_DATE.
* Horizon comes from config.FORECAST_HORIZON_DAYS (30).
* The model loads lazily (see ml/model.py), not at import time.
* The 0.5 quantile is selected explicitly. Chronos-2 returns
  (n_variates, n_quantiles, horizon); Nishit's code took the median across
  the 21 quantile levels, which is the same 0.5 quantile value.
* Negative forecasts are clipped to 0 (demand cannot be negative).
* Series are checked for gaps, short history and stale end dates.
* generate_forecasts_batch() sends many series per predict() call.

Chronos-2 does NOT use the engineered lag/rolling features in
ml/features.py. It receives only the target and the two flag covariates.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

import config
from ml import data_loader
from ml.model import ModelUnavailableError, get_pipeline

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["date", "sku_id", "warehouse_id", "forecast_demand"]


class ForecastError(Exception):
    """A forecast could not be produced for a series."""


# ------------------------------------------------------------------
# Input preparation
# ------------------------------------------------------------------

def _check_horizon(horizon: int) -> int:
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ForecastError(f"Forecast horizon must be a positive integer, got {horizon!r}")
    return int(horizon)


def prepare_series(history: pd.DataFrame, sku_id: str, warehouse_id: str) -> pd.DataFrame:
    """
    Validate and sort one series' history.

    Requirements: non-empty, at least MIN_HISTORY_DAYS rows, daily with no
    gaps or duplicate dates, and ending exactly on SIMULATION_DATE so the
    forecast starts on the day after the simulation date.
    """
    label = f"{sku_id} / {warehouse_id}"
    if history is None or history.empty:
        raise ForecastError(f"No demand history found for {label}")

    missing = [c for c in ["date", "demand_qty", *config.COVARIATE_COLUMNS] if c not in history.columns]
    if missing:
        raise ForecastError(f"{label}: history missing columns {missing}")

    series = history.copy()
    series["date"] = pd.to_datetime(series["date"], errors="coerce")
    if series["date"].isna().any():
        raise ForecastError(f"{label}: history contains invalid dates")

    series = series[series["date"] <= config.simulation_timestamp()]
    series = series.sort_values("date").reset_index(drop=True)

    if len(series) < config.MIN_HISTORY_DAYS:
        raise ForecastError(
            f"Not enough demand history for {label}: {len(series)} days "
            f"(minimum {config.MIN_HISTORY_DAYS})"
        )
    if series["date"].duplicated().any():
        raise ForecastError(f"{label}: duplicate dates in history")

    expected_days = (series["date"].iloc[-1] - series["date"].iloc[0]).days + 1
    if expected_days != len(series):
        raise ForecastError(
            f"{label}: history has {expected_days - len(series)} missing days; "
            "Chronos-2 input must be a regular daily series"
        )
    if series["date"].iloc[-1] != config.simulation_timestamp():
        raise ForecastError(
            f"{label}: history ends {series['date'].iloc[-1].date()}, "
            f"expected SIMULATION_DATE {config.SIMULATION_DATE}"
        )
    if series[["demand_qty", *config.COVARIATE_COLUMNS]].isna().any().any():
        raise ForecastError(f"{label}: missing demand or flag values")
    return series


def build_chronos_input(series: pd.DataFrame, horizon: int) -> dict:
    """
    Build one Chronos-2 input dict: target, past covariates, and future
    covariates using Nishit's carry-forward fallback.
    """
    if config.FUTURE_COVARIATE_STRATEGY != "carry_forward":
        raise ForecastError(
            f"Unsupported FUTURE_COVARIATE_STRATEGY {config.FUTURE_COVARIATE_STRATEGY!r}"
        )
    past = {c: series[c].to_numpy(dtype=np.float32) for c in config.COVARIATE_COLUMNS}
    future = {c: np.full(horizon, float(series[c].iloc[-1]), dtype=np.float32)
              for c in config.COVARIATE_COLUMNS}
    return {
        "target": series["demand_qty"].to_numpy(dtype=np.float32),
        "past_covariates": past,
        "future_covariates": future,
    }


# ------------------------------------------------------------------
# Output handling
# ------------------------------------------------------------------

def _to_numpy(prediction) -> np.ndarray:
    if hasattr(prediction, "detach"):  # torch.Tensor
        prediction = prediction.detach().cpu().float().numpy()
    return np.asarray(prediction, dtype=np.float64)


def extract_point_forecast(prediction, quantile_levels, horizon: int) -> np.ndarray:
    """
    Turn one Chronos-2 prediction of shape (1, n_quantiles, horizon) into a
    1-D median forecast of length horizon, clipped at 0.
    """
    arr = _to_numpy(prediction)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ForecastError(f"Expected a univariate prediction, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] != horizon:
        raise ForecastError(f"Unexpected prediction shape {arr.shape} for horizon {horizon}")

    levels = list(quantile_levels) if quantile_levels is not None else []
    q = config.POINT_FORECAST_QUANTILE
    matches = [i for i, level in enumerate(levels) if abs(float(level) - q) < 1e-9]
    if matches and len(levels) == arr.shape[0]:
        values = arr[matches[0]]
    else:
        values = np.median(arr, axis=0)  # Nishit's original approach

    if not np.all(np.isfinite(values)):
        raise ForecastError("Chronos-2 returned non-finite forecast values")
    return np.clip(values, 0.0, None)


def _forecast_frame(sku_id: str, warehouse_id: str, values: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range(config.simulation_timestamp() + pd.Timedelta(days=1),
                          periods=len(values), freq="D")
    return pd.DataFrame({
        "date": dates,
        "sku_id": sku_id,
        "warehouse_id": warehouse_id,
        "forecast_demand": values.astype(float),
    })[OUTPUT_COLUMNS]


def _predict(pipeline, inputs: list[dict], horizon: int) -> list:
    try:
        predictions = pipeline.predict(inputs, prediction_length=horizon)
    except Exception as exc:
        raise ForecastError(f"Chronos-2 prediction failed: {exc}") from exc
    if len(predictions) != len(inputs):
        raise ForecastError(
            f"Chronos-2 returned {len(predictions)} predictions for {len(inputs)} series"
        )
    return predictions


# ------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------

def generate_forecast(sku_id: str, warehouse_id: str,
                      history: pd.DataFrame | None = None,
                      pipeline=None,
                      horizon: int | None = None,
                      db_path=None) -> pd.DataFrame:
    """
    30-day demand forecast for one SKU + warehouse.

    Returns a DataFrame with columns date, sku_id, warehouse_id,
    forecast_demand; dates start the day after SIMULATION_DATE.

    `history` and `pipeline` can be injected (tests, batch runs); by default
    history comes from the database and the shared Chronos-2 model is used.
    Raises ForecastError or ModelUnavailableError.
    """
    horizon = _check_horizon(horizon if horizon is not None else config.FORECAST_HORIZON_DAYS)
    if history is None:
        history = data_loader.load_series(sku_id, warehouse_id, db_path=db_path)
    series = prepare_series(history, sku_id, warehouse_id)
    pipe = pipeline if pipeline is not None else get_pipeline()

    prediction = _predict(pipe, [build_chronos_input(series, horizon)], horizon)[0]
    values = extract_point_forecast(prediction, getattr(pipe, "quantiles", None), horizon)
    return _forecast_frame(sku_id, warehouse_id, values)


def generate_forecasts_batch(pairs: Iterable[tuple[str, str]] | None = None,
                             demand: pd.DataFrame | None = None,
                             pipeline=None,
                             horizon: int | None = None,
                             batch_size: int | None = None,
                             db_path=None) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    """
    Forecast many SKU + warehouse series, several per Chronos-2 call.

    Each series is forecast independently (cross_learning is off), so the
    results match calling generate_forecast() one series at a time.

    Returns (forecasts, errors): the combined forecast DataFrame and a dict
    mapping (sku_id, warehouse_id) -> error message for series that failed.
    A failed series never gets substitute values.
    """
    horizon = _check_horizon(horizon if horizon is not None else config.FORECAST_HORIZON_DAYS)
    size = batch_size or config.FORECAST_BATCH_SIZE
    if demand is None:
        demand = data_loader.load_demand_data(db_path=db_path)
    if pairs is None:
        pairs = (demand[["sku_id", "warehouse_id"]].drop_duplicates()
                 .sort_values(["sku_id", "warehouse_id"]).itertuples(index=False, name=None))
    pairs = list(pairs)

    grouped = {key: grp for key, grp in demand.groupby(["sku_id", "warehouse_id"])}
    errors: dict[tuple[str, str], str] = {}
    ready: list[tuple[tuple[str, str], dict]] = []
    for key in pairs:
        try:
            series = prepare_series(grouped.get(key), *key)
            ready.append((key, build_chronos_input(series, horizon)))
        except ForecastError as exc:
            errors[key] = str(exc)
            logger.error("Forecast skipped for %s/%s: %s", *key, exc)

    pipe = pipeline if pipeline is not None else get_pipeline()
    quantiles = getattr(pipe, "quantiles", None)
    frames = []
    for start in range(0, len(ready), size):
        chunk = ready[start:start + size]
        logger.info("Forecasting series %d-%d of %d", start + 1, start + len(chunk), len(ready))
        try:
            predictions = _predict(pipe, [inp for _, inp in chunk], horizon)
        except ForecastError as exc:
            for key, _ in chunk:
                errors[key] = str(exc)
            continue
        for (key, _), prediction in zip(chunk, predictions):
            try:
                values = extract_point_forecast(prediction, quantiles, horizon)
                frames.append(_forecast_frame(*key, values))
            except ForecastError as exc:
                errors[key] = str(exc)

    forecasts = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=OUTPUT_COLUMNS))
    forecasts = forecasts.sort_values(["sku_id", "warehouse_id", "date"]).reset_index(drop=True)
    return forecasts, errors


__all__ = ["generate_forecast", "generate_forecasts_batch", "ForecastError",
           "ModelUnavailableError"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = generate_forecast("M001", "W002")
    print("\n30-day demand forecast (M001 / W002):")
    print(result.to_string(index=False))
    print("\nTotal forecast demand:", round(result["forecast_demand"].sum(), 2), "units")
