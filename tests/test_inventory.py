"""
Phase 5 tests: inventory formulas, plus the DB-backed calculator on Ayush's
real data. Expected values are computed independently from the CSVs.
"""

import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine import inventory_calculator as ic
from tests.helpers import make_temp_db


def make_forecast(sku, wh, values):
    start = config.simulation_timestamp() + pd.Timedelta(days=1)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(values), freq="D"),
        "sku_id": sku, "warehouse_id": wh, "forecast_demand": values,
    })


class FormulaTests(unittest.TestCase):

    def test_average_daily_demand(self):
        self.assertAlmostEqual(ic.average_daily_demand([10, 20, 30, 40]), 25.0)

    def test_average_daily_demand_all_zero(self):
        self.assertEqual(ic.average_daily_demand([0, 0, 0]), 0.0)

    def test_average_daily_demand_empty_raises(self):
        with self.assertRaises(ic.InventoryCalculationError):
            ic.average_daily_demand([])

    def test_lead_time_demand_sums_first_n_days(self):
        values = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        self.assertAlmostEqual(ic.lead_time_demand(values, 3), 30.0)
        self.assertAlmostEqual(ic.lead_time_demand(values, 5), 75.0)
        self.assertAlmostEqual(ic.lead_time_demand(values, 6), sum(values))

    def test_lead_time_zero(self):
        self.assertEqual(ic.lead_time_demand([5.0, 6.0], 0), 0.0)

    def test_lead_time_longer_than_forecast_raises(self):
        with self.assertRaises(ic.InventoryCalculationError):
            ic.lead_time_demand([1.0] * 30, 31)

    def test_negative_lead_time_raises(self):
        with self.assertRaises(ic.InventoryCalculationError):
            ic.lead_time_demand([1.0] * 30, -1)

    def test_required_stock(self):
        self.assertAlmostEqual(ic.required_stock(270.5, 151), 421.5)

    def test_projected_inventory(self):
        self.assertAlmostEqual(ic.projected_inventory(115, 270.5), -155.5)
        self.assertAlmostEqual(ic.projected_inventory(500, 100.0), 400.0)

    def test_days_of_stock(self):
        self.assertAlmostEqual(ic.days_of_stock(100, 25.0), 4.0)

    def test_days_of_stock_zero_demand(self):
        self.assertIsNone(ic.days_of_stock(100, 0.0))
        self.assertIsNone(ic.days_of_stock(0, 0.0))

    def test_excess_inventory(self):
        self.assertAlmostEqual(ic.excess_inventory(1000, 400.0), 600.0)
        self.assertAlmostEqual(ic.excess_inventory(100, 400.0), -300.0)


class CalculatorOnRealDataTests(unittest.TestCase):
    """Real Ayush inventory + demand; a known forecast so expected values are exact."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.inv = pd.read_csv(config.INVENTORY_CSV)
        cls.dem = pd.read_csv(config.DEMAND_CSV)
        cls.bat = pd.read_csv(config.BATCHES_CSV, parse_dates=["expiry_date"])

    def expected(self, sku, wh, fc_values):
        row = self.inv[(self.inv.sku_id == sku) & (self.inv.warehouse_id == wh)].iloc[0]
        hist = self.dem[(self.dem.sku_id == sku) & (self.dem.warehouse_id == wh)
                        & (self.dem.date <= config.SIMULATION_DATE)]
        avg = hist.demand_qty.sum() / len(hist)
        ltd = sum(fc_values[: int(row.lead_time_days)])
        req = ltd + row.safety_stock
        b = self.bat[(self.bat.sku_id == sku) & (self.bat.warehouse_id == wh)]
        usable = int(b.quantity[b.expiry_date > config.simulation_timestamp()].sum())
        return row, avg, ltd, req, usable

    def test_every_field_for_several_items(self):
        fc_values = [float(v) for v in np.linspace(20, 80, config.FORECAST_HORIZON_DAYS)]
        for sku, wh in [("M001", "W002"), ("M001", "W004"), ("M017", "W002"), ("M030", "W005")]:
            calc = ic.calculate_inventory(sku, wh, make_forecast(sku, wh, fc_values),
                                          db_path=self.db_path)
            row, avg, ltd, req, usable = self.expected(sku, wh, fc_values)
            self.assertEqual(calc.current_stock, row.current_stock)
            self.assertEqual(calc.usable_stock, usable)
            self.assertEqual(calc.expired_stock, row.current_stock - usable)
            self.assertTrue(calc.batch_total_matches_inventory)
            self.assertEqual(calc.lead_time_days, row.lead_time_days)
            self.assertAlmostEqual(calc.average_daily_demand, avg)
            self.assertAlmostEqual(calc.lead_time_demand, ltd)
            self.assertAlmostEqual(calc.required_stock, req)
            # Operational figures use usable stock, never recorded stock.
            self.assertAlmostEqual(calc.projected_inventory, usable - ltd)
            self.assertAlmostEqual(calc.days_of_stock, usable / avg)
            self.assertAlmostEqual(calc.excess_inventory, usable - req)

    def test_uses_forecast_not_average(self):
        # Two forecasts that differ only after the lead time must give the same lead-time demand.
        sku, wh = "M001", "W002"
        lt = db.get_inventory_item(sku, wh, db_path=self.db_path)["lead_time_days"]
        a = [10.0] * config.FORECAST_HORIZON_DAYS
        b = [10.0] * lt + [999.0] * (config.FORECAST_HORIZON_DAYS - lt)
        ca = ic.calculate_inventory(sku, wh, make_forecast(sku, wh, a), db_path=self.db_path)
        cb = ic.calculate_inventory(sku, wh, make_forecast(sku, wh, b), db_path=self.db_path)
        self.assertAlmostEqual(ca.lead_time_demand, 10.0 * lt)
        self.assertAlmostEqual(ca.lead_time_demand, cb.lead_time_demand)

    def test_unsorted_forecast_is_sorted_before_summing(self):
        sku, wh = "M001", "W002"
        values = list(range(1, config.FORECAST_HORIZON_DAYS + 1))
        fc = make_forecast(sku, wh, values).iloc[::-1]
        calc = ic.calculate_inventory(sku, wh, fc, db_path=self.db_path)
        lt = calc.lead_time_days
        self.assertAlmostEqual(calc.lead_time_demand, sum(values[:lt]))

    def test_bad_forecasts_rejected(self):
        sku, wh = "M001", "W002"
        good = make_forecast(sku, wh, [10.0] * 30)
        cases = {
            "empty": good.iloc[0:0],
            "other item": make_forecast("M002", wh, [10.0] * 30),
            "wrong start": good.assign(date=good["date"] + pd.Timedelta(days=1)),
            "gap": good.drop(index=2),
            "nan": good.assign(forecast_demand=[np.nan] + [10.0] * 29),
            "negative": good.assign(forecast_demand=[-1.0] + [10.0] * 29),
            "missing column": good.drop(columns=["forecast_demand"]),
        }
        for name, fc in cases.items():
            with self.subTest(name):
                with self.assertRaises(ic.InventoryCalculationError):
                    ic.calculate_inventory(sku, wh, fc, db_path=self.db_path)
        with self.assertRaises(ic.InventoryCalculationError):
            ic.calculate_inventory(sku, wh, None, db_path=self.db_path)

    def test_missing_item(self):
        with self.assertRaises(db.RecordNotFoundError):
            ic.calculate_inventory("M999", "W002", make_forecast("M999", "W002", [1.0] * 30),
                                   db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
