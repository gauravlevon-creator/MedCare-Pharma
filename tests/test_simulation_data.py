"""
Tests for the date-shifted simulation demand copy and the simulation timeline.
Ayush's source demand file is compared row by row with the shifted copy.
"""

import json
import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from database.simulation_data import VALUE_COLUMNS, build_simulation_demand
from ml.predict import build_chronos_input, generate_forecasts_batch, prepare_series
from tests.helpers import FakeChronosPipeline, make_temp_db


class SimulationDataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.meta = build_simulation_demand()
        cls.db_path = make_temp_db()
        cls.source = pd.read_csv(config.SOURCE_DEMAND_CSV, parse_dates=["date"])
        cls.sim = pd.read_csv(config.SIMULATION_DEMAND_CSV, parse_dates=["date"])

    def test_source_file_untouched_and_distinct(self):
        self.assertNotEqual(config.SOURCE_DEMAND_CSV, config.SIMULATION_DEMAND_CSV)
        # The source file keeps its own dates; the simulation copy ends on the simulation date.
        self.assertEqual(self.source["date"].nunique(), 365)
        self.assertEqual(self.sim["date"].max(), config.simulation_timestamp())

    def test_same_rows_series_and_values(self):
        self.assertEqual(len(self.sim), len(self.source))
        key = ["sku_id", "warehouse_id", "date"]
        a = self.source.sort_values(key).reset_index(drop=True)
        b = self.sim.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[VALUE_COLUMNS], b[VALUE_COLUMNS])
        shift = (b["date"] - a["date"]).dt.days
        self.assertEqual(shift.nunique(), 1)
        self.assertEqual(int(shift.iloc[0]), self.meta["shift_days"])

    def test_simulation_period(self):
        self.assertEqual(self.sim["date"].max(), config.simulation_timestamp())
        per_series = self.sim.groupby(["sku_id", "warehouse_id"])["date"].agg(["min", "max", "count"])
        self.assertEqual(len(per_series), 180)
        self.assertTrue((per_series["max"] == config.simulation_timestamp()).all())
        self.assertTrue(((per_series["max"] - per_series["min"]).dt.days + 1 == per_series["count"]).all())

    def test_metadata(self):
        on_disk = json.loads(config.SIMULATION_DEMAND_METADATA.read_text())
        self.assertEqual(on_disk["simulation_end"], config.SIMULATION_DATE)
        self.assertEqual(on_disk["rows"], len(self.source))
        self.assertEqual(on_disk["series"], 180)

    def test_database_matches_simulation_copy(self):
        demand = db.get_all_demand(db_path=self.db_path)
        self.assertEqual(len(demand), len(self.sim))
        self.assertEqual(demand[["sku_id", "warehouse_id"]].drop_duplicates().shape[0], 180)
        self.assertLessEqual(demand["date"].max(), config.simulation_timestamp())

    def test_no_dates_after_simulation_date_in_database(self):
        everything = db.get_all_demand(end_date="2099-12-31", db_path=self.db_path)
        self.assertFalse((everything["date"] > config.simulation_timestamp()).any())

    def test_forecast_period_follows_simulation_date(self):
        demand = db.get_all_demand(db_path=self.db_path)
        fc, errors = generate_forecasts_batch(demand=demand, pipeline=FakeChronosPipeline())
        self.assertEqual(errors, {})
        start = config.simulation_timestamp() + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=config.FORECAST_HORIZON_DAYS - 1)
        self.assertEqual(fc["date"].min(), start)
        self.assertEqual(fc["date"].max(), end)
        self.assertEqual(len(fc), 180 * config.FORECAST_HORIZON_DAYS)

    def test_chronos_inputs_identical_to_source_series(self):
        # Chronos-2 gets arrays without timestamps, so the shift must not change its inputs.
        hist = db.get_demand_history("M001", "W002", db_path=self.db_path)
        inp = build_chronos_input(prepare_series(hist, "M001", "W002"), config.FORECAST_HORIZON_DAYS)
        src = self.source[(self.source.sku_id == "M001") & (self.source.warehouse_id == "W002")].sort_values("date")
        np.testing.assert_array_equal(inp["target"], src["demand_qty"].to_numpy(dtype=np.float32))
        for col in config.COVARIATE_COLUMNS:
            np.testing.assert_array_equal(inp["past_covariates"][col], src[col].to_numpy(dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
