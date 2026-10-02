"""
Holdout evaluation: 7-day moving-average baseline vs Chronos-2.

Combines Nishit's compare_baseline.py and evaluate_chronos.py:
* time-based split (default 80/20, no shuffling)
* evaluate the first `horizon` days of the test period
* baseline  = mean of the last 7 training days, repeated (Nishit's baseline)
* Chronos-2 = same inputs as production; future covariates are the real
  held-out flags, because they are known for a historical window
* metrics  = MAE and RMSE

Results apply ONLY to the series and window evaluated. They are not
overall model accuracy.

Usage:
    python -m ml.evaluate --sku M001 --warehouse W002 --horizon 30
    python -m ml.evaluate --sku M001 --warehouse W002 --horizon 7 --no-covariates
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config
from ml import data_loader
from ml.model import get_pipeline
from ml.predict import ForecastError, extract_point_forecast


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def evaluate_series(sku_id: str, warehouse_id: str, horizon: int = 30,
                    train_ratio: float = 0.8, use_covariates: bool = True,
                    pipeline=None, history: pd.DataFrame | None = None) -> dict:
    """Evaluate baseline and Chronos-2 on one series' holdout window."""
    if history is None:
        history = data_loader.load_series(sku_id, warehouse_id)
    series = history.sort_values("date").reset_index(drop=True)
    split = int(len(series) * train_ratio)
    train, test = series.iloc[:split], series.iloc[split:split + horizon]
    if len(train) < config.MIN_HISTORY_DAYS or len(test) < horizon:
        raise ForecastError(
            f"Not enough data to evaluate {sku_id}/{warehouse_id}: "
            f"{len(train)} train days, {len(test)} of {horizon} test days"
        )

    actual = test["demand_qty"].to_numpy(dtype=float)
    baseline = np.repeat(train["demand_qty"].tail(7).mean(), horizon)

    inp = {"target": train["demand_qty"].to_numpy(dtype=np.float32)}
    if use_covariates:
        inp["past_covariates"] = {c: train[c].to_numpy(dtype=np.float32)
                                  for c in config.COVARIATE_COLUMNS}
        inp["future_covariates"] = {c: test[c].to_numpy(dtype=np.float32)
                                    for c in config.COVARIATE_COLUMNS}
    pipe = pipeline if pipeline is not None else get_pipeline()
    prediction = pipe.predict([inp], prediction_length=horizon)[0]
    chronos = extract_point_forecast(prediction, getattr(pipe, "quantiles", None), horizon)

    return {
        "sku_id": sku_id,
        "warehouse_id": warehouse_id,
        "scope": "single series, single holdout window",
        "train_days": len(train),
        "evaluation_period": f"{test['date'].iloc[0].date()} to {test['date'].iloc[-1].date()}",
        "horizon_days": horizon,
        "chronos_covariates": use_covariates,
        "baseline_mae": round(mae(actual, baseline), 2),
        "baseline_rmse": round(rmse(actual, baseline), 2),
        "chronos_mae": round(mae(actual, chronos), 2),
        "chronos_rmse": round(rmse(actual, chronos), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chronos-2 vs 7-day moving average")
    parser.add_argument("--sku", default="M001")
    parser.add_argument("--warehouse", default="W002")
    parser.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON_DAYS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--no-covariates", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = evaluate_series(args.sku, args.warehouse, args.horizon, args.train_ratio,
                             use_covariates=not args.no_covariates)
    for key, value in result.items():
        print(f"{key:>20}: {value}")


if __name__ == "__main__":
    main()
