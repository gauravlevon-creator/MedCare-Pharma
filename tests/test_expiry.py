"""
Phase 7 tests: batch status boundaries, usable stock, forecast-based expected
consumption (FEFO), potential expiry excess, expiry risk, transfer
eligibility, and M001/W002 + M001/W004 on Ayush's real batches (expected
values recomputed independently from batches.csv).
"""

import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine import expiry_engine as ee
from tests.helpers import make_temp_db

SIM = config.simulation_timestamp()
H = config.FORECAST_HORIZON_DAYS


def batch(batch_id, qty, days):
    return {"batch_id": batch_id, "quantity": qty,
            "expiry_date": (SIM + pd.Timedelta(days=days)).strftime("%Y-%m-%d")}


def analyse(rows, daily=10.0, values=None):
    fc = np.full(H, daily) if values is None else np.asarray(values, dtype=float)
    return ee.analyse_batches(pd.DataFrame(rows), "MX", "WX", fc)


def make_forecast(sku, wh, values):
    return pd.DataFrame({"date": pd.date_range(SIM + pd.Timedelta(days=1), periods=len(values)),
                         "sku_id": sku, "warehouse_id": wh, "forecast_demand": values})


class StatusTests(unittest.TestCase):

    def test_expired_batch(self):
        r = analyse([batch("B1", 100, -1)])
        d = r.batch_details[0]
        self.assertEqual(d["expiry_status"], config.BATCH_EXPIRED)
        self.assertFalse(d["usable"])
        self.assertEqual((r.expired_quantity, r.usable_quantity), (100, 0))
        self.assertEqual(d["potential_expiry_excess"], 0.0)  # expired stock is reported, not "excess"

    def test_expiry_exactly_on_simulation_date_is_unavailable(self):
        # SIMULATION_DATE is the end of the day, so a batch expiring on it cannot serve future demand.
        r = analyse([batch("B0", 40, 0), batch("B1", 25, 1)], daily=10.0)
        d0, d1 = r.batch_details
        self.assertEqual((d0["days_to_expiry"], d0["expiry_status"]), (0, config.BATCH_NEAR_EXPIRY))
        self.assertFalse(d0["usable"])
        self.assertFalse(d0["transfer_eligible"])
        self.assertIsNone(d0["forecast_demand_through_expiry"])
        self.assertEqual((d0["expected_consumption"], d0["potential_expiry_excess"]), (0.0, 0.0))
        self.assertEqual((r.expired_quantity, r.usable_quantity), (40, 25))
        self.assertEqual(r.near_expiry_quantity, 25)  # usable near-expiry only
        # The day-0 batch consumed no demand: day 1's forecast goes to B1.
        self.assertEqual(d1["expected_consumption"], 10.0)
        self.assertTrue(d1["usable"])
        self.assertEqual(ee.split_stock(pd.DataFrame([batch("B0", 40, 0)])),
                         {"recorded": 40, "expired": 40, "usable": 0})

    def test_boundaries(self):
        cases = {30: config.BATCH_NEAR_EXPIRY, 31: config.BATCH_EXPIRING_SOON,
                 60: config.BATCH_EXPIRING_SOON, 61: config.BATCH_NORMAL, 200: config.BATCH_NORMAL}
        for days, status in cases.items():
            with self.subTest(days=days):
                self.assertEqual(ee.classify_batch(days), status)
                self.assertEqual(analyse([batch("B", 10, days)]).batch_details[0]["expiry_status"], status)

    def test_mixed_batches(self):
        r = analyse([batch("B1", 10, -5), batch("B2", 20, 0), batch("B3", 30, 30),
                     batch("B4", 40, 31), batch("B5", 50, 60), batch("B6", 60, 90)])
        self.assertEqual(r.total_recorded_quantity, 210)
        self.assertEqual(r.expired_quantity, 30)      # B1 (past) + B2 (expires on simulation date)
        self.assertEqual(r.usable_quantity, 180)
        self.assertEqual(r.near_expiry_quantity, 30)  # B3 only; B2 is unavailable
        self.assertEqual(r.expiring_soon_quantity, 90)
        self.assertEqual(r.normal_quantity, 60)
        self.assertEqual([d["batch_id"] for d in r.batch_details], ["B1", "B2", "B3", "B4", "B5", "B6"])

    def test_zero_batches(self):
        r = ee.analyse_batches(pd.DataFrame(columns=["batch_id", "quantity", "expiry_date"]),
                               "MX", "WX", np.full(H, 5.0))
        self.assertEqual((r.total_recorded_quantity, r.usable_quantity, r.expired_quantity), (0, 0, 0))
        self.assertEqual((r.batch_count, r.batch_details, r.potential_expiry_excess), (0, [], 0.0))
        self.assertEqual(r.expiry_risk, ee.EXPIRY_RISK_LOW)

    def test_invalid_inputs(self):
        with self.assertRaises(ee.ExpiryCalculationError):
            analyse([{"batch_id": "B1", "quantity": 5, "expiry_date": "not-a-date"}])
        with self.assertRaises(ee.ExpiryCalculationError):
            analyse([batch("B1", 5, 5)], values=[])
        with self.assertRaises(ee.ExpiryCalculationError):
            analyse([batch("B1", 5, 5)], values=[-1.0] * H)


class ConsumptionTests(unittest.TestCase):

    def test_uses_actual_forecast_days_not_average(self):
        values = [1.0] * 5 + [100.0] * (H - 5)
        d = analyse([batch("B1", 50, 5)], values=values).batch_details[0]
        self.assertEqual(d["forecast_demand_through_expiry"], 5.0)
        self.assertEqual(d["expected_consumption"], 5.0)
        self.assertEqual(d["potential_expiry_excess"], 45.0)

    def test_fully_consumed_near_expiry_is_low_risk(self):
        r = analyse([batch("B1", 100, 20)], daily=10.0)  # 200 demand before expiry
        self.assertEqual(r.potential_expiry_excess, 0.0)
        self.assertEqual(r.expiry_risk, ee.EXPIRY_RISK_LOW)

    def test_unconsumed_near_expiry_is_high_risk(self):
        r = analyse([batch("B1", 300, 20)], daily=10.0)
        self.assertEqual(r.near_expiry_excess, 100.0)
        self.assertEqual(r.expiry_risk, ee.EXPIRY_RISK_HIGH)

    def test_unconsumed_expiring_soon_is_medium_risk(self):
        # 45 days: 30 forecast days (300) + 15 extended at mean 10 (150) = 450 demand.
        r = analyse([batch("B1", 1000, 45)], daily=10.0)
        d = r.batch_details[0]
        self.assertEqual(d["forecast_demand_through_expiry"], 450.0)
        self.assertEqual(d["consumption_basis"], ee.BASIS_EXTENDED)
        self.assertEqual(r.expiring_soon_excess, 550.0)
        self.assertEqual(r.expiry_risk, ee.EXPIRY_RISK_MEDIUM)

    def test_same_days_to_expiry_is_not_decisive(self):
        # Same batch and expiry date; only demand differs -> different risk.
        low = analyse([batch("B1", 100, 10)], daily=20.0)
        high = analyse([batch("B1", 100, 10)], daily=2.0)
        self.assertEqual(low.expiry_risk, ee.EXPIRY_RISK_LOW)
        self.assertEqual(high.expiry_risk, ee.EXPIRY_RISK_HIGH)

    def test_fefo_does_not_double_count_demand(self):
        # 10/day. B1 (10 d): demand 100 -> uses 100 of 150, excess 50.
        # B2 (30 d): demand 300 - 100 already used = 200 -> uses 200 of 250, excess 50.
        r = analyse([batch("B2", 250, 30), batch("B1", 150, 10)], daily=10.0)
        by_id = {d["batch_id"]: d for d in r.batch_details}
        self.assertEqual((by_id["B1"]["expected_consumption"], by_id["B1"]["potential_expiry_excess"]),
                         (100.0, 50.0))
        self.assertEqual((by_id["B2"]["expected_consumption"], by_id["B2"]["potential_expiry_excess"]),
                         (200.0, 50.0))
        self.assertEqual(r.potential_expiry_excess, 100.0)

    def test_expired_batches_do_not_consume_demand(self):
        r = analyse([batch("B0", 500, -3), batch("B1", 50, 10)], daily=10.0)
        self.assertEqual(r.batch_details[1]["expected_consumption"], 50.0)
        self.assertEqual(r.potential_expiry_excess, 0.0)
        self.assertIn("already expired", r.expiry_reason)

    def test_transfer_eligibility_uses_transit_assumption(self):
        t = config.TRANSFER_TRANSIT_DAYS
        r = analyse([batch("A", 10, t), batch("B", 10, t + 1), batch("C", 10, -1)])
        flags = {d["batch_id"]: d["transfer_eligible"] for d in r.batch_details}
        self.assertEqual(flags, {"A": False, "B": True, "C": False})
        self.assertEqual(r.transfer_eligible_quantity, 10)


class RealDataExpiryTests(unittest.TestCase):
    """M001/W002 and M001/W004: statuses/quantities from batches.csv; consumption from a given forecast."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.batches = pd.read_csv(config.BATCHES_CSV, parse_dates=["expiry_date"])
        cls.inventory = pd.read_csv(config.INVENTORY_CSV)

    def check_item(self, sku, wh):
        values = np.linspace(15.0, 45.0, H)
        r = ee.evaluate_expiry(sku, wh, make_forecast(sku, wh, values), db_path=self.db_path)
        b = self.batches[(self.batches.sku_id == sku) & (self.batches.warehouse_id == wh)]
        days = (b.expiry_date - SIM).dt.days
        inv = self.inventory[(self.inventory.sku_id == sku) & (self.inventory.warehouse_id == wh)].iloc[0]

        self.assertEqual(r.total_recorded_quantity, int(inv.current_stock))
        self.assertEqual(r.expired_quantity, int(b.quantity[days <= 0].sum()))
        self.assertEqual(r.usable_quantity, int(b.quantity[days > 0].sum()))
        self.assertEqual(r.near_expiry_quantity, int(b.quantity[(days > 0) & (days <= 30)].sum()))

        # Recompute FEFO consumption independently.
        consumed, total_excess = 0.0, 0.0
        for _, row in b.assign(d=days).sort_values(["expiry_date", "batch_id"]).iterrows():
            det = next(x for x in r.batch_details if x["batch_id"] == row.batch_id)
            self.assertEqual(det["days_to_expiry"], row.d)
            if row.d <= 0:
                continue
            dd = int(row.d)
            demand = values[:min(dd, H)].sum() + max(0, dd - H) * values.mean()
            used = min(row.quantity, max(0.0, demand - consumed))
            consumed += used
            total_excess += row.quantity - used
            self.assertAlmostEqual(det["expected_consumption"], round(used, 2), places=2)
        self.assertAlmostEqual(r.potential_expiry_excess, round(total_excess, 2), places=2)
        return r

    def test_m001_w002(self):
        self.check_item("M001", "W002")

    def test_m001_w004(self):
        self.check_item("M001", "W004")

    def test_same_day_batches_unavailable_in_real_data(self):
        same_day = self.batches[self.batches.expiry_date == SIM]
        table = ee.usable_stock_by_item(db_path=self.db_path).set_index(["sku_id", "warehouse_id"])
        for _, row in same_day.iterrows():
            b = self.batches[(self.batches.sku_id == row.sku_id) & (self.batches.warehouse_id == row.warehouse_id)]
            expected_usable = int(b.quantity[b.expiry_date > SIM].sum())
            self.assertEqual(table.loc[(row.sku_id, row.warehouse_id), "usable_quantity"], expected_usable)

    def test_usable_stock_table(self):
        table = ee.usable_stock_by_item(db_path=self.db_path)
        self.assertEqual(len(table), 180)
        self.assertTrue((table.usable_quantity + table.expired_quantity == table.total_recorded_quantity).all())
        self.assertTrue((table.total_recorded_quantity == table.current_stock).all())

    def test_bad_forecast_rejected(self):
        with self.assertRaises(ee.ExpiryCalculationError):
            ee.evaluate_expiry("M001", "W002", make_forecast("M002", "W002", [1.0] * H),
                               db_path=self.db_path)

    def test_missing_item(self):
        with self.assertRaises(db.RecordNotFoundError):
            ee.evaluate_expiry("M999", "W002", make_forecast("M999", "W002", [1.0] * H),
                               db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
