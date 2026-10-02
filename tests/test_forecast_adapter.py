"""
Forecast adapter tests using a fake Chronos-2 pipeline, so they run
without torch or the model download. The real-model test is in
test_m001_w002_forecast.py.
"""

import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from ml.predict import (ForecastError, OUTPUT_COLUMNS, build_chronos_input,
                        extract_point_forecast, generate_forecast, generate_forecasts_batch)
from tests.helpers import QUANTILES, FakeChronosPipeline, make_temp_db, point_forecast_expected


class ForecastAdapterTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.history = db.get_demand_history("M001", "W002", db_path=cls.db_path)

    def test_m001_w002_forecast_shape_and_dates(self):
        fake = FakeChronosPipeline()
        fc = generate_forecast("M001", "W002", pipeline=fake, db_path=self.db_path)
        self.assertEqual(list(fc.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(fc), config.FORECAST_HORIZON_DAYS)
        first = config.simulation_timestamp() + pd.Timedelta(days=1)
        self.assertEqual(fc["date"].iloc[0], first)
        self.assertEqual(fc["date"].iloc[-1], first + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS - 1))
        self.assertTrue((fc["sku_id"] == "M001").all())
        self.assertTrue((fc["warehouse_id"] == "W002").all())

    def test_uses_median_quantile(self):
        fake = FakeChronosPipeline()
        fc = generate_forecast("M001", "W002", pipeline=fake, db_path=self.db_path)
        expected = point_forecast_expected(self.history["demand_qty"].to_numpy())
        np.testing.assert_allclose(fc["forecast_demand"].to_numpy(), expected)

    def test_chronos_receives_full_history_and_carry_forward_covariates(self):
        fake = FakeChronosPipeline()
        generate_forecast("M001", "W002", pipeline=fake, db_path=self.db_path)
        inputs, horizon = fake.calls[0]
        self.assertEqual(horizon, config.FORECAST_HORIZON_DAYS)
        inp = inputs[0]
        np.testing.assert_array_equal(inp["target"], self.history["demand_qty"].to_numpy())
        for col in config.COVARIATE_COLUMNS:
            self.assertEqual(len(inp["past_covariates"][col]), len(self.history))
            self.assertEqual(len(inp["future_covariates"][col]), horizon)
            self.assertTrue((inp["future_covariates"][col] == self.history[col].iloc[-1]).all())

    def test_negative_forecasts_clipped(self):
        fake = FakeChronosPipeline(offset=-10_000)
        fc = generate_forecast("M001", "W002", pipeline=fake, db_path=self.db_path)
        self.assertTrue((fc["forecast_demand"] >= 0).all())

    def test_median_fallback_without_quantile_labels(self):
        levels = np.array([1.0, 2.0, 7.0])
        pred = np.tile(levels[:, None], (1, 5))[None]
        np.testing.assert_allclose(extract_point_forecast(pred, None, 5), 2.0)

    def test_quantile_selection_matches_nishit_median(self):
        rng = np.random.default_rng(0)
        pred = np.sort(rng.normal(50, 10, size=(len(QUANTILES), 30)), axis=0)[None]
        np.testing.assert_allclose(extract_point_forecast(pred, QUANTILES, 30),
                                   np.clip(np.median(pred[0], axis=0), 0, None))

    def test_custom_horizon(self):
        fc = generate_forecast("M001", "W002", pipeline=FakeChronosPipeline(), horizon=7,
                               db_path=self.db_path)
        self.assertEqual(len(fc), 7)

    def test_invalid_horizon(self):
        for bad in (0, -3, 2.5):
            with self.assertRaises(ForecastError):
                generate_forecast("M001", "W002", pipeline=FakeChronosPipeline(), horizon=bad,
                                  db_path=self.db_path)

    def test_missing_series(self):
        with self.assertRaises(ForecastError):
            generate_forecast("M999", "W002", pipeline=FakeChronosPipeline(), db_path=self.db_path)

    def test_short_history(self):
        with self.assertRaises(ForecastError):
            generate_forecast("M001", "W002", history=self.history.tail(5),
                              pipeline=FakeChronosPipeline())

    def test_gap_in_history(self):
        gapped = self.history.drop(index=[100, 101])
        with self.assertRaises(ForecastError):
            generate_forecast("M001", "W002", history=gapped, pipeline=FakeChronosPipeline())

    def test_stale_history(self):
        stale = self.history[self.history["date"] <= config.simulation_timestamp() - pd.Timedelta(days=30)]
        with self.assertRaises(ForecastError):
            generate_forecast("M001", "W002", history=stale, pipeline=FakeChronosPipeline())

    def test_future_rows_are_ignored(self):
        extra = self.history.tail(3).copy()
        extra["date"] = extra["date"] + pd.Timedelta(days=3)
        extra["demand_qty"] = 99_999
        fake = FakeChronosPipeline()
        generate_forecast("M001", "W002", history=pd.concat([self.history, extra]), pipeline=fake)
        self.assertEqual(len(fake.calls[0][0][0]["target"]), len(self.history))
        self.assertLess(fake.calls[0][0][0]["target"].max(), 99_999)

    def test_model_failure_raises_forecast_error(self):
        with self.assertRaises(ForecastError):
            generate_forecast("M001", "W002", pipeline=FakeChronosPipeline(fail=True),
                              db_path=self.db_path)

    def test_build_input_types(self):
        inp = build_chronos_input(self.history, 30)
        self.assertEqual(inp["target"].dtype, np.float32)

    def test_batch_all_180_series(self):
        demand = db.get_all_demand(db_path=self.db_path)
        fake = FakeChronosPipeline()
        fc, errors = generate_forecasts_batch(demand=demand, pipeline=fake, batch_size=50)
        self.assertEqual(errors, {})
        self.assertEqual(fc[["sku_id", "warehouse_id"]].drop_duplicates().shape[0], 180)
        self.assertEqual(len(fc), 180 * config.FORECAST_HORIZON_DAYS)
        self.assertEqual(len(fake.calls), 4)  # 180 series / 50 per call
        single = generate_forecast("M001", "W002", pipeline=FakeChronosPipeline(),
                                   db_path=self.db_path)
        batch_row = fc[(fc.sku_id == "M001") & (fc.warehouse_id == "W002")].reset_index(drop=True)
        pd.testing.assert_frame_equal(batch_row, single)

    def test_batch_reports_bad_series_without_fake_values(self):
        demand = db.get_all_demand(db_path=self.db_path)
        demand = demand[~((demand.sku_id == "M002") & (demand.warehouse_id == "W003"))]
        fc, errors = generate_forecasts_batch(
            pairs=[("M001", "W002"), ("M002", "W003")], demand=demand,
            pipeline=FakeChronosPipeline())
        self.assertIn(("M002", "W003"), errors)
        self.assertFalse(((fc.sku_id == "M002") & (fc.warehouse_id == "W003")).any())
        self.assertEqual(len(fc), config.FORECAST_HORIZON_DAYS)


if __name__ == "__main__":
    unittest.main()
