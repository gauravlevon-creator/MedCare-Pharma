"""Derived sales dataset: +8-day shift, source untouched, values identical, loaded into SQLite."""

import json
import unittest

import pandas as pd

import config
from database import database as db
from database.simulation_data import SALES_VALUE_COLUMNS, build_simulation_sales, file_sha256
from tests.helpers import make_temp_db


class SalesDerivationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source_hash_before = file_sha256(config.SOURCE_SALES_CSV)
        cls.meta = build_simulation_sales()
        cls.source = pd.read_csv(config.SOURCE_SALES_CSV, parse_dates=["date"])
        cls.derived = pd.read_csv(config.SIMULATION_SALES_CSV, parse_dates=["date"])

    def test_source_file_unchanged(self):
        self.assertEqual(file_sha256(config.SOURCE_SALES_CSV), self.source_hash_before)
        self.assertEqual(self.meta["source_sha256"], self.source_hash_before)
        self.assertNotEqual(config.SOURCE_SALES_CSV, config.SIMULATION_SALES_CSV)

    def test_source_and_derived_ranges(self):
        self.assertEqual((self.source.date.min(), self.source.date.max()),
                         (pd.Timestamp("2025-09-23"), pd.Timestamp("2026-09-22")))
        self.assertEqual((self.derived.date.min(), self.derived.date.max()),
                         (pd.Timestamp("2025-10-01"), config.simulation_timestamp()))
        self.assertEqual(self.meta["shift_days"], 8)

    def test_only_dates_changed(self):
        key = ["sku_id", "warehouse_id", "date"]
        a = self.source.sort_values(key).reset_index(drop=True)
        b = self.derived.sort_values(key).reset_index(drop=True)
        self.assertEqual(len(a), len(b))
        pd.testing.assert_frame_equal(a[SALES_VALUE_COLUMNS], b[SALES_VALUE_COLUMNS], check_dtype=False)
        self.assertTrue(((b.date - a.date).dt.days == 8).all())
        self.assertEqual(set(a.sku_id), set(b.sku_id))
        self.assertEqual(set(a.warehouse_id), set(b.warehouse_id))
        self.assertEqual(int(a.units_sold.sum()), int(b.units_sold.sum()))
        self.assertAlmostEqual(float(a.revenue.sum()), float(b.revenue.sum()), places=2)

    def test_metadata_file(self):
        on_disk = json.loads(config.SIMULATION_SALES_METADATA.read_text())
        self.assertEqual(on_disk["rows"], len(self.source))
        self.assertEqual(on_disk["series"], 180)
        self.assertIn("NOT for forecasting", on_disk["description"])

    def test_loaded_into_database(self):
        path = make_temp_db()
        sales = db.get_all_sales(db_path=path)
        self.assertEqual(len(sales), len(self.source))
        self.assertEqual(sales.date.max(), config.simulation_timestamp())
        one = db.get_sales_history("M001", "W002", start_date="2026-09-30", db_path=path)
        src = self.source[(self.source.sku_id == "M001") & (self.source.warehouse_id == "W002")
                          & (self.source.date == pd.Timestamp("2026-09-22"))]
        self.assertEqual(int(one.units_sold.iloc[0]), int(src.units_sold.iloc[0]))
        # demand is still the demand dataset (sales never replaces it)
        self.assertEqual(len(db.get_all_demand(db_path=path)), 65700)
        self.assertEqual(file_sha256(config.SOURCE_SALES_CSV), self.source_hash_before)


if __name__ == "__main__":
    unittest.main()
