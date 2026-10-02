"""Database layer tests against Ayush's real CSVs (values compared to the CSVs, not hard-coded)."""

import unittest
from pathlib import Path

import pandas as pd

import config
from database import database as db
from tests.helpers import make_temp_db


class DatabaseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.demand_csv = pd.read_csv(config.DEMAND_CSV)
        cls.inventory_csv = pd.read_csv(config.INVENTORY_CSV)
        cls.batches_csv = pd.read_csv(config.BATCHES_CSV)

    def test_row_counts_match_csvs(self):
        self.assertEqual(len(db.get_all_demand(db_path=self.db_path)), len(self.demand_csv))
        self.assertEqual(len(db.get_all_inventory(db_path=self.db_path)), len(self.inventory_csv))
        self.assertEqual(len(db.get_all_batches(db_path=self.db_path)), len(self.batches_csv))

    def test_all_pairs_present(self):
        pairs = db.list_sku_warehouse_pairs(db_path=self.db_path)
        expected = set(map(tuple, self.inventory_csv[["sku_id", "warehouse_id"]].values))
        self.assertEqual(set(pairs), expected)
        self.assertEqual(len(pairs), len(expected))

    def test_inventory_item_matches_csv(self):
        for sku, wh in [("M001", "W002"), ("M001", "W004"), ("M030", "W006")]:
            item = db.get_inventory_item(sku, wh, db_path=self.db_path)
            row = self.inventory_csv[(self.inventory_csv.sku_id == sku)
                                     & (self.inventory_csv.warehouse_id == wh)].iloc[0]
            for col in db.INVENTORY_COLUMNS[2:]:
                self.assertEqual(item[col], int(row[col]), f"{sku}/{wh} {col}")

    def test_demand_history_stops_at_simulation_date(self):
        hist = db.get_demand_history("M001", "W002", db_path=self.db_path)
        self.assertEqual(hist["date"].max(), config.simulation_timestamp())
        self.assertTrue(hist["date"].is_monotonic_increasing)
        csv_rows = self.demand_csv[(self.demand_csv.sku_id == "M001")
                                   & (self.demand_csv.warehouse_id == "W002")
                                   & (self.demand_csv.date <= config.SIMULATION_DATE)]
        self.assertEqual(len(hist), len(csv_rows))
        self.assertEqual(int(hist["demand_qty"].sum()), int(csv_rows["demand_qty"].sum()))

    def test_demand_history_respects_earlier_end_date(self):
        hist = db.get_demand_history("M001", "W002", end_date="2026-06-30", db_path=self.db_path)
        self.assertEqual(hist["date"].max(), pd.Timestamp("2026-06-30"))

    def test_batches_sum_to_current_stock(self):
        batches = db.get_batches_for_item("M001", "W004", db_path=self.db_path)
        item = db.get_inventory_item("M001", "W004", db_path=self.db_path)
        self.assertEqual(int(batches["quantity"].sum()), item["current_stock"])
        self.assertTrue(batches["expiry_date"].is_monotonic_increasing)

    def test_missing_records(self):
        with self.assertRaises(db.RecordNotFoundError):
            db.get_inventory_item("M999", "W002", db_path=self.db_path)
        self.assertTrue(db.get_demand_history("M001", "W999", db_path=self.db_path).empty)
        self.assertTrue(db.get_batches_for_item("M999", "W001", db_path=self.db_path).empty)

    def test_parameterised_sql_resists_injection(self):
        hist = db.get_demand_history("M001' OR '1'='1", "W002", db_path=self.db_path)
        self.assertTrue(hist.empty)

    def test_missing_database_raises_clear_error(self):
        with self.assertRaises(db.DatabaseNotInitializedError):
            db.get_all_inventory(db_path=Path(self.db_path).parent / "does_not_exist.db")

    def test_validation_rejects_bad_data(self):
        bad = self.inventory_csv.copy()
        bad.loc[0, "current_stock"] = -5
        with self.assertRaises(db.DataValidationError):
            db.validate_source_data(self.demand_csv, bad, self.batches_csv)
        dup = pd.concat([self.inventory_csv, self.inventory_csv.head(1)])
        with self.assertRaises(db.DataValidationError):
            db.validate_source_data(self.demand_csv, dup, self.batches_csv)


if __name__ == "__main__":
    unittest.main()
