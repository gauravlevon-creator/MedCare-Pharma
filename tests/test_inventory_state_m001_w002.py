"""
Phase 5-6 checkpoint: M001 / W002 with the REAL Chronos-2 forecast and the
real Ayush database (config.DATABASE_PATH).

Uses integration.ml_service.get_forecast (saved forecast if valid,
otherwise Chronos-2). Skipped when the model is unavailable or the database
has not been initialised. Expected values are recomputed from the CSVs and
the forecast; no business result is hard-coded.
"""

import unittest

import pandas as pd

import config
from database.database import DatabaseNotInitializedError
from ml.model import ModelUnavailableError


class M001W002InventoryStateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from decision_engine.decision_engine import evaluate_inventory_state
        from integration.ml_service import get_forecast
        try:
            cls.forecast = get_forecast("M001", "W002")
        except (ModelUnavailableError, DatabaseNotInitializedError) as exc:
            raise unittest.SkipTest(f"Real forecast unavailable: {exc}")
        cls.state = evaluate_inventory_state("M001", "W002", cls.forecast)

    def test_values_match_independent_calculation(self):
        inv = pd.read_csv(config.INVENTORY_CSV)
        dem = pd.read_csv(config.DEMAND_CSV)
        row = inv[(inv.sku_id == "M001") & (inv.warehouse_id == "W002")].iloc[0]
        hist = dem[(dem.sku_id == "M001") & (dem.warehouse_id == "W002")
                   & (dem.date <= config.SIMULATION_DATE)]
        fc = self.forecast.sort_values("date")["forecast_demand"].tolist()
        bat = pd.read_csv(config.BATCHES_CSV, parse_dates=["expiry_date"])
        b = bat[(bat.sku_id == "M001") & (bat.warehouse_id == "W002")]
        usable = int(b.quantity[b.expiry_date > config.simulation_timestamp()].sum())

        avg = hist.demand_qty.sum() / len(hist)
        ltd = sum(fc[: int(row.lead_time_days)])
        s = self.state
        self.assertEqual(s["current_stock"], row.current_stock)
        self.assertEqual(s["usable_stock"], usable)
        self.assertEqual(s["lead_time_days"], row.lead_time_days)
        self.assertAlmostEqual(s["average_daily_demand"], avg, places=6)
        self.assertAlmostEqual(s["lead_time_demand"], ltd, places=6)
        self.assertAlmostEqual(s["required_stock"], ltd + row.safety_stock, places=6)
        self.assertAlmostEqual(s["projected_inventory"], usable - ltd, places=6)
        self.assertAlmostEqual(s["days_of_stock"], usable / avg, places=6)
        self.assertAlmostEqual(s["excess_inventory"], usable - ltd - row.safety_stock, places=6)

    def test_risk_follows_rules(self):
        s = self.state
        self.assertIn(s["risk_status"], config.RISK_LEVELS)
        if s["projected_inventory"] <= 0:
            self.assertEqual(s["risk_status"], config.RISK_HIGH)
        elif s["projected_inventory"] <= s["safety_stock"]:
            self.assertEqual(s["risk_status"], config.RISK_MEDIUM)
        else:
            self.assertEqual(s["risk_status"], config.RISK_LOW)
        self.assertTrue(s["risk_reason"])
        self.assertEqual(s["e1_threshold_alert"], s["usable_stock"] <= s["min_threshold"])


if __name__ == "__main__":
    unittest.main()
