"""
E1 simulation on Ayush's real stock + derived sales: calibrated sales pattern,
transfer receipts after the transit days, reorder receipts after the lead
time, expiry write-offs, no negative stock, reconciliation, persistence,
and that the Phase 9 golden inputs are untouched.
"""

import math
import unittest

import pandas as pd

import config
from database import database as db
from decision_engine.decision_engine import evaluate_sku
from operations.sales_simulation import (BREACH, e1_alerts_from_events, requested_sales,
                                         run_simulation)
from tests.helpers import make_temp_db
from tests.test_allocation import flat_forecast_provider

D = config.simulation_timestamp()


def day(n):
    return (D + pd.Timedelta(days=n)).strftime("%Y-%m-%d")


class E1SimulationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.transfers = pd.DataFrame([{"sku_id": "M001", "source_warehouse": "W004",
                                       "destination_warehouse": "W002", "batch_id": "B00006",
                                       "transfer_quantity": 10}])
        cls.decisions = pd.DataFrame([{"sku_id": "M001", "warehouse_id": "W002", "final_action": "REORDER",
                                       "reorder_quantity": 24},
                                      {"sku_id": "M001", "warehouse_id": "W004", "final_action": "NO_ACTION",
                                       "reorder_quantity": 0}])
        # plan-only isolates the P1 plan's own receipts; the default policy is tested below.
        cls.result = run_simulation(days=10, decisions=cls.decisions, transfers=cls.transfers,
                                    db_path=cls.db_path, run_id="t-run", replenishment_policy="plan-only")
        cls.tx = cls.result.transactions

    def test_requested_sales_follow_calibrated_pattern(self):
        req = requested_sales(5, "demand_calibrated", db_path=self.db_path)
        sales = db.get_all_sales(db_path=self.db_path)
        demand = db.get_all_demand(db_path=self.db_path)
        s = sales[(sales.sku_id == "M001") & (sales.warehouse_id == "W002")]
        d = demand[(demand.sku_id == "M001") & (demand.warehouse_id == "W002")]
        scale = d.demand_qty.mean() / s.units_sold.mean()
        for i in range(5):
            target = D + pd.Timedelta(days=i + 1)
            pattern = int(s[s.date == target - pd.Timedelta(days=365)].units_sold.iloc[0])
            row = req[(req.date == target) & (req.sku_id == "M001") & (req.warehouse_id == "W002")].iloc[0]
            self.assertEqual(row.units_sold, pattern)
            self.assertEqual(row.requested, math.floor(pattern * scale + 0.5))
        raw = requested_sales(5, "raw", db_path=self.db_path)
        self.assertTrue((raw.requested == raw.units_sold).all())
        self.assertFalse(req.pattern_missing.any())

    def test_every_item_day_has_a_sale_row(self):
        sales = self.tx[self.tx.txn_type == "SALE"]
        self.assertEqual(len(sales), 180 * 10)
        self.assertTrue((sales.quantity <= 0).all())
        self.assertTrue((-sales.quantity + sales.unfulfilled_quantity == sales.requested_quantity).all())

    def test_no_negative_inventory(self):
        self.assertTrue((self.tx.on_hand_after >= 0).all())
        self.assertTrue((self.tx.usable_after >= 0).all())
        self.assertTrue((self.result.daily.on_hand >= 0).all())
        self.assertEqual(self.result.reconciliation_problems, [])

    def test_opening_expiry_write_off(self):
        wo = self.tx[(self.tx.txn_subtype == "EXPIRY_WRITE_OFF") & (self.tx.txn_date == day(0))]
        batches = db.get_all_batches(db_path=self.db_path)
        expired = batches[batches.expiry_date <= D]
        self.assertEqual(set(wo.batch_id), set(expired.batch_id))
        self.assertEqual(int(-wo.quantity.sum()), int(expired.quantity.sum()))
        later = self.tx[(self.tx.txn_subtype == "EXPIRY_WRITE_OFF") & (self.tx.txn_date > day(0))]
        for r in later.itertuples():
            exp = batches[batches.batch_id == r.batch_id].expiry_date.iloc[0]
            self.assertEqual(pd.Timestamp(r.txn_date), exp)  # written off at end of its expiry day

    def test_transfer_receipt_after_transit(self):
        out = self.tx[self.tx.txn_type == "TRANSFER_OUT"]
        rec = self.tx[self.tx.txn_subtype == "TRANSFER_RECEIPT"]
        self.assertEqual(len(out), 1)
        self.assertEqual((out.iloc[0].txn_date, out.iloc[0].warehouse_id, out.iloc[0].quantity), (day(1), "W004", -10))
        self.assertEqual((rec.iloc[0].txn_date, rec.iloc[0].warehouse_id, rec.iloc[0].quantity),
                         (day(1 + config.TRANSFER_TRANSIT_DAYS), "W002", 10))
        self.assertEqual(rec.iloc[0].batch_id, "B00006")
        self.assertEqual(out.iloc[0].reference, rec.iloc[0].reference)

    def test_reorder_receipt_after_lead_time(self):
        lt = db.get_inventory_item("M001", "W002", db_path=self.db_path)["lead_time_days"]
        rec = self.tx[self.tx.txn_subtype == "SUPPLIER_RECEIPT"]
        self.assertEqual(len(rec), 1)
        self.assertEqual((rec.iloc[0].txn_date, rec.iloc[0].warehouse_id, rec.iloc[0].quantity),
                         (day(1 + lt), "W002", 24))
        po = self.result.purchase_orders.iloc[0]
        self.assertEqual((po.order_date, po.due_date, po.quantity), (day(1), day(1 + lt), 24))
        self.assertEqual(po.origin, "P1_PLAN")
        self.assertTrue(rec.iloc[0].reference.startswith("P1-PO:"))
        self.assertIn("P1 plan reorder", rec.iloc[0].reason)

    def test_threshold_events_and_e1_alerts(self):
        ev = self.result.threshold_events
        opening = ev[(ev.date == day(0)) & (ev.event == BREACH)]
        stock = db.get_all_batches(db_path=self.db_path)
        usable = stock[stock.expiry_date > D].groupby(["sku_id", "warehouse_id"]).quantity.sum()
        inv = db.get_all_inventory(db_path=self.db_path).set_index(["sku_id", "warehouse_id"])
        expected = sum(1 for k in inv.index if usable.get(k, 0) <= inv.loc[k, "min_threshold"])
        self.assertEqual(len(opening), expected)
        alerts = e1_alerts_from_events(ev, self.decisions)
        self.assertEqual(len(alerts), int((ev.event == BREACH).sum()))
        w002 = [a for a in alerts if (a["sku_id"], a["warehouse_id"]) == ("M001", "W002")][0]
        self.assertEqual((w002["alert_type"], w002["recommended_action"]), ("E1_THRESHOLD", "REORDER"))

    def test_reorder_point_policy_orders_after_breach(self):
        r = run_simulation(days=10, db_path=self.db_path, replenishment_policy="plan+reorder_point")
        self.assertEqual(r.reconciliation_problems, [])
        self.assertGreater(len(r.purchase_orders), 0)
        for po in r.purchase_orders.head(20).itertuples():
            rec = r.transactions[(r.transactions.reference == po.po_reference)]
            if pd.Timestamp(po.due_date) <= pd.Timestamp(r.end_date):
                self.assertEqual(rec.iloc[0].txn_date, po.due_date)
                self.assertEqual(int(rec.iloc[0].quantity), po.quantity)

    def test_persist_and_phase9_inputs_untouched(self):
        before_inv = db.get_all_inventory(db_path=self.db_path)
        before_batches = db.get_all_batches(db_path=self.db_path)
        provider = flat_forecast_provider(self.db_path)
        d_before, _, _ = evaluate_sku("M001", provider, db_path=self.db_path)
        r = run_simulation(days=5, db_path=self.db_path, run_id="persist-run", persist=True)
        stored = db.get_transactions("persist-run", db_path=self.db_path)
        self.assertEqual(len(stored), len(r.transactions))
        pd.testing.assert_frame_equal(before_inv, db.get_all_inventory(db_path=self.db_path))
        pd.testing.assert_frame_equal(before_batches, db.get_all_batches(db_path=self.db_path))
        d_after, _, _ = evaluate_sku("M001", provider, db_path=self.db_path)
        self.assertEqual({w: x.to_dict() for w, x in d_before.items()},
                         {w: x.to_dict() for w, x in d_after.items()})


if __name__ == "__main__":
    unittest.main()
