"""
Final acceptance check for the E1 + P1 use case, run end to end on Ayush's
real data with a stand-in forecast (the real Chronos-2 run is covered by
the model tests on a machine that has it). Each test maps to one checklist line.
"""

import unittest

import pandas as pd

import config
from database import database as db
from decision_engine.decision_engine import evaluate_all
from decision_engine.seasonal_signal import compute_seasonal_signals
from ml.predict import generate_forecasts_batch
from notification_engine import NotificationEngine
from operations.sales_simulation import e1_alerts_from_events, run_simulation
from tests.helpers import FakeChronosPipeline, make_temp_db
from tests.test_allocation import flat_forecast_provider


class AcceptanceE1P1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.decisions, cls.transfers, cls.alerts = evaluate_all(flat_forecast_provider(cls.db_path),
                                                                db_path=cls.db_path)
        cls.sim = run_simulation(days=config.E1_SIMULATION_DAYS, decisions=cls.decisions,
                                 transfers=cls.transfers, db_path=cls.db_path, persist=True)
        cls.notify = NotificationEngine(mode="outbox", env={}, db_path=cls.db_path).notify(
            [a for a in cls.alerts.to_dict("records")] + e1_alerts_from_events(cls.sim.threshold_events,
                                                                               cls.decisions))
        cls.tx = cls.sim.transactions

    # ---------------- E1 ----------------
    def test_e1_inventory_and_thresholds_exist(self):
        inv = db.get_all_inventory(db_path=self.db_path)
        self.assertEqual(len(inv), 180)
        self.assertTrue((inv.min_threshold > 0).all())

    def test_e1_sales_consume_inventory(self):
        sales = self.tx[self.tx.txn_type == "SALE"]
        self.assertGreater(int(-sales.quantity.sum()), 0)

    def test_e1_receipts_increase_inventory(self):
        rec = self.tx[self.tx.txn_type == "RECEIPT"]
        self.assertTrue((rec.quantity > 0).all())
        self.assertEqual(set(rec.txn_subtype), {"TRANSFER_RECEIPT", "SUPPLIER_RECEIPT"})

    def test_e1_adjustments_change_inventory(self):
        adj = self.tx[self.tx.txn_type == "ADJUSTMENT"]
        self.assertGreater(len(adj), 0)
        self.assertTrue((adj.reason.str.len() > 0).all())

    def test_e1_inventory_updated_consistently(self):
        self.assertEqual(self.sim.reconciliation_problems, [])
        self.assertTrue((self.tx.on_hand_after >= 0).all())
        self.assertEqual(len(db.get_transactions(self.sim.run_id, db_path=self.db_path)), len(self.tx))

    def test_e1_threshold_alerts_generated(self):
        self.assertTrue((self.alerts.alert_type == "E1_THRESHOLD").any())
        self.assertTrue((self.sim.threshold_events.event == "BREACH").any())

    def test_e1_email_capability_and_alert_log(self):
        self.assertGreater(self.notify.notified, 0)
        log = db.get_notifications(db_path=self.db_path)
        self.assertEqual(len(log), self.notify.notified)
        self.assertTrue((log.status == "OUTBOX").all())  # no credentials in tests: recorded, not sent
        self.assertTrue(hasattr(NotificationEngine, "_send"))

    def test_e1_reorder_recommendations(self):
        # Whether any item needs a supplier reorder depends on the data and forecast
        # (transfers may cover every shortage). Every REORDER must carry a quantity,
        # and every shortage must be handled by a TRANSFER or a REORDER.
        d = self.decisions
        self.assertTrue((d[d.final_action == "REORDER"].reorder_quantity > 0).all())
        short = d[d.shortage_before > 0]
        self.assertTrue(short.final_action.isin(["TRANSFER", "REORDER"]).all())

    # ---------------- P1 ----------------
    def test_p1_historical_demand(self):
        self.assertEqual(len(db.get_all_demand(db_path=self.db_path)), 65700)

    def test_p1_seasonal_signal(self):
        sig = compute_seasonal_signals(db_path=self.db_path)
        self.assertEqual(len(sig), 180)
        self.assertTrue(self.decisions.seasonal_signal.notna().all())

    def test_p1_30_day_forecast_interface(self):
        demand = db.get_all_demand(db_path=self.db_path)
        fc, errors = generate_forecasts_batch(pairs=[("M001", "W002")], demand=demand,
                                              pipeline=FakeChronosPipeline())
        self.assertEqual((len(fc), errors), (config.FORECAST_HORIZON_DAYS, {}))
        self.assertEqual(fc.date.min(), config.simulation_timestamp() + pd.Timedelta(days=1))

    def test_p1_inventory_batches_expiry_capacity_lead_time(self):
        d = self.decisions
        for col in ("usable_stock_before", "expired_stock", "expiry_risk", "capacity", "lead_time_days",
                    "lead_time_demand"):
            self.assertIn(col, d.columns)
        self.assertEqual(len(db.get_all_batches(db_path=self.db_path)), len(pd.read_csv(config.BATCHES_CSV)))

    def test_p1_expiry_aware_cross_warehouse_transfers(self):
        self.assertGreater(len(self.transfers), 0)
        self.assertTrue((self.transfers.days_to_expiry > config.TRANSFER_TRANSIT_DAYS).all())
        self.assertTrue((self.transfers.source_warehouse != self.transfers.destination_warehouse).all())

    def test_p1_stockout_risk_replenishment_final_action(self):
        d = self.decisions
        self.assertTrue(d.risk_status.isin(config.RISK_LEVELS).all())
        self.assertTrue(d.final_action.isin(["TRANSFER", "REORDER", "NO_ACTION"]).all())
        self.assertEqual(len(d), 180)


if __name__ == "__main__":
    unittest.main()
