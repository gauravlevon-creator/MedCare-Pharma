"""
Phase 4 golden test: REAL Chronos-2 forecast for M001 / W002.

Skipped automatically when chronos-forecasting/torch are not installed or
MEDCARE_SKIP_MODEL_TESTS=1. First run downloads amazon/chronos-2.

Checks structure and sanity only. It does not assert specific forecast
numbers, because those come from the model, not from us.
"""

import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

import config
from database import database as db

MODEL_AVAILABLE = (importlib.util.find_spec("chronos") is not None
                   and importlib.util.find_spec("torch") is not None
                   and os.environ.get("MEDCARE_SKIP_MODEL_TESTS") != "1")


@unittest.skipUnless(MODEL_AVAILABLE, "Chronos-2 not installed (or MEDCARE_SKIP_MODEL_TESTS=1)")
class RealChronosM001W002Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ml.predict import generate_forecast
        from tests.helpers import make_temp_db
        cls.db_path = make_temp_db()
        cls.history = db.get_demand_history("M001", "W002", db_path=cls.db_path)
        cls.forecast = generate_forecast("M001", "W002", db_path=cls.db_path)

    def test_structure(self):
        fc = self.forecast
        self.assertEqual(list(fc.columns), ["date", "sku_id", "warehouse_id", "forecast_demand"])
        self.assertEqual(len(fc), config.FORECAST_HORIZON_DAYS)
        self.assertEqual(fc["date"].iloc[0], config.simulation_timestamp() + pd.Timedelta(days=1))
        self.assertTrue(fc["date"].diff().dropna().eq(pd.Timedelta(days=1)).all())

    def test_values_finite_and_non_negative(self):
        values = self.forecast["forecast_demand"].to_numpy()
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values >= 0))

    def test_sanity_against_recent_history(self):
        # Loose guard against unit or scaling bugs: the mean forecast should be
        # the same order of magnitude as the last 90 days of actual demand.
        recent = self.history["demand_qty"].tail(90).mean()
        mean_fc = self.forecast["forecast_demand"].mean()
        self.assertGreater(mean_fc, 0.25 * recent)
        self.assertLess(mean_fc, 4.0 * recent)
        print(f"\nM001/W002: recent 90-day mean {recent:.2f}, forecast mean {mean_fc:.2f}, "
              f"30-day total {self.forecast['forecast_demand'].sum():.2f}")


if __name__ == "__main__":
    unittest.main()
