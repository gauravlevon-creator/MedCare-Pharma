"""Shared test helpers: a temporary database and a deterministic fake Chronos-2."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import config
from database import database as db

QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


class FakeChronosPipeline:
    """
    Mimics Chronos2Pipeline.predict output: one array per input of shape
    (1, n_quantiles, horizon). The 0.5 quantile equals the mean of the
    last 7 target values plus `offset`; other quantiles spread around it.
    Records every call so tests can inspect the inputs.
    """

    quantiles = QUANTILES

    def __init__(self, offset: float = 0.0, fail: bool = False):
        self.offset = offset
        self.fail = fail
        self.calls: list[tuple[list[dict], int]] = []

    def predict(self, inputs, prediction_length):
        self.calls.append((inputs, prediction_length))
        if self.fail:
            raise RuntimeError("simulated model failure")
        out = []
        for inp in inputs:
            median = float(np.mean(inp["target"][-7:])) + self.offset
            levels = np.array([median + (q - 0.5) * 20 for q in QUANTILES])
            out.append(np.tile(levels[:, None], (1, prediction_length))[None, :, :])
        return out


def make_temp_db() -> Path:
    """Build a database from the real Ayush CSVs in a temp folder."""
    tmp = Path(tempfile.mkdtemp(prefix="medcare_test_"))
    path = tmp / "test.db"
    db.init_database(db_path=path)
    return path


def point_forecast_expected(history_demand: np.ndarray, offset: float = 0.0) -> float:
    return max(0.0, float(np.mean(history_demand[-7:])) + offset)


__all__ = ["FakeChronosPipeline", "make_temp_db", "point_forecast_expected", "config"]
