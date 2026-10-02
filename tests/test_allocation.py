"""
Phase 8 tests: allocation rules on small hand-built warehouses, constraint
checks across all 30 SKUs on Ayush's real data, and the M001 golden scenario
with the real Chronos-2 forecast (skipped when the model is unavailable).
"""

import math
import unittest

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine import allocation_engine as ae
from decision_engine.allocation_engine import WarehouseSnapshot, allocate
from decision_engine.expiry_engine import cumulative_forecast_demand
from tests.helpers import make_temp_db

H = config.FORECAST_HORIZON_DAYS
T = config.TRANSFER_TRANSIT_DAYS
SIM = config.simulation_timestamp()


def b(batch_id, qty, days, excess=0.0, used=None):
    return {"batch_id": batch_id, "quantity": qty, "days_to_expiry": days,
            "expiry_date": (SIM + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "usable": days > 0, "transfer_eligible": days > T,
            "potential_expiry_excess": excess,
            "expected_consumption": (qty - excess) if used is None else used}


def wh(w, usable, required, batches, daily=10.0, capacity=10_000, recorded=None, risk=None,
       safety=10, sku="MX"):
    if risk is None:
        proj = usable - (required - safety)
        risk = config.RISK_HIGH if proj <= 0 else config.RISK_MEDIUM if proj <= safety else config.RISK_LOW
    return WarehouseSnapshot(sku_id=sku, warehouse_id=w,
                             recorded_stock=usable if recorded is None else recorded,
                             usable_stock=usable, required_stock=float(required), safety_stock=safety,
                             capacity=capacity, risk_status=risk,
                             forecast=np.full(H, float(daily)), batches=batches)


def check_invariants(tc, result, snapshots):
    """Hard constraints that must hold for every allocation."""
    snaps = {s.warehouse_id: s for s in snapshots}
    batch_index = {(s.warehouse_id, x["batch_id"]): x for s in snapshots for x in s.batches}
    sent = {}
    inbound = {w: 0 for w in snaps}
    for t in result.transfers:
        tc.assertIsInstance(t.transfer_quantity, int)
        tc.assertGreaterEqual(t.transfer_quantity, 1)
        tc.assertNotEqual(t.source_warehouse, t.destination_warehouse)
        src_batch = batch_index[(t.source_warehouse, t.batch_id)]
        tc.assertTrue(src_batch["usable"])
        tc.assertGreater(t.days_to_expiry, T, "batch must outlive transit")
        tc.assertLessEqual(t.transfer_quantity, t.destination_consumption_before_expiry + 1e-6)
        key = (t.source_warehouse, t.batch_id)
        sent[key] = sent.get(key, 0) + t.transfer_quantity
        tc.assertLessEqual(sent[key], src_batch["quantity"])
        inbound[t.destination_warehouse] += t.transfer_quantity
        tc.assertTrue(t.reason)
    for w, s in snaps.items():
        tc.assertLessEqual(s.recorded_stock + inbound[w], max(s.capacity, s.recorded_stock),
                           f"{w} capacity exceeded")
    tc.assertEqual(set(result.decisions), set(snaps))
    for w, d in result.decisions.items():
        tc.assertIn(d.recommended_action, ae.ACTIONS)
        tc.assertTrue(d.reason)
        tc.assertGreaterEqual(d.reorder_quantity, 0)
        tc.assertGreaterEqual(d.usable_stock_after, 0)
        if d.recommended_action == ae.ACTION_REORDER:
            tc.assertEqual(d.transferred_in, 0)
        if any(t.source_warehouse == w and t.transfer_type == ae.TYPE_EXCESS for t in result.transfers):
            # Independent check: replay the source's remaining batches earliest-expiry-first
            # against its own forecast and count what would still expire unused.
            consumed, at_risk = 0.0, 0.0
            for x in sorted(snaps[w].batches, key=lambda x: (x["days_to_expiry"], x["batch_id"])):
                left = x["quantity"] - sent.get((w, x["batch_id"]), 0)
                demand = cumulative_forecast_demand(snaps[w].forecast, x["days_to_expiry"])[0]
                used = min(left, max(0.0, demand - consumed))
                consumed += used
                at_risk += left - used
            tc.assertGreaterEqual(d.usable_stock_after - at_risk, snaps[w].required_stock - 1e-6,
                                  f"{w} dropped below required stock after a normal transfer")


class AllocationRuleTests(unittest.TestCase):

    def test_shortage_filled_from_source_excess(self):
        snaps = [wh("W1", 20, 100, [b("A", 20, 100)]),
                 wh("W2", 300, 120, [b("B", 300, 100)])]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        (t,) = r.transfers
        self.assertEqual((t.source_warehouse, t.destination_warehouse, t.transfer_quantity),
                         ("W2", "W1", 80))
        self.assertEqual(t.transfer_type, ae.TYPE_EXCESS)
        self.assertEqual((t.source_usable_before, t.source_usable_after), (300, 220))
        self.assertEqual((t.destination_usable_before, t.destination_usable_after), (20, 100))
        self.assertEqual(r.decision_for("W1").recommended_action, ae.ACTION_TRANSFER)
        self.assertEqual(r.decision_for("W2").recommended_action, ae.ACTION_NO_ACTION)

    def test_transfer_limited_to_source_excess_then_supplementary_reorder(self):
        snaps = [wh("W1", 0, 100, []), wh("W2", 150, 120, [b("B", 150, 100)])]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        self.assertEqual(r.transfers[0].transfer_quantity, 30)
        d = r.decision_for("W1")
        self.assertEqual(d.recommended_action, ae.ACTION_TRANSFER)
        self.assertEqual(d.supplementary_reorder_quantity, 70)
        self.assertEqual(r.decision_for("W2").usable_stock_after, 120)  # exactly required

    def test_reorder_when_no_source(self):
        snaps = [wh("W1", 10, 100.4, [b("A", 10, 100)]), wh("W2", 50, 120, [b("B", 50, 100)])]
        r = allocate(snaps)
        self.assertEqual(r.transfers, [])
        d = r.decision_for("W1")
        self.assertEqual(d.recommended_action, ae.ACTION_REORDER)
        self.assertEqual(d.reorder_quantity, math.ceil(100.4 - 10))

    def test_no_action_when_covered(self):
        snaps = [wh("W1", 200, 100, [b("A", 200, 100)]), wh("W2", 200, 100, [b("B", 200, 100)])]
        r = allocate(snaps)
        self.assertEqual(r.transfers, [])
        self.assertTrue(all(d.recommended_action == ae.ACTION_NO_ACTION for d in r.decisions.values()))

    def test_batch_inside_transit_window_not_transferred(self):
        snaps = [wh("W1", 0, 100, []), wh("W2", 500, 100, [b("B", 500, T)])]  # expires on arrival day
        r = allocate(snaps)
        self.assertEqual(r.transfers, [])
        self.assertEqual(r.decision_for("W1").recommended_action, ae.ACTION_REORDER)

    def test_unusable_batch_rejected(self):
        with self.assertRaises(ae.AllocationError):
            allocate([wh("W1", 0, 10, [b("X", 5, 0)])])

    def test_capacity_caps_transfer_and_reorder(self):
        snaps = [wh("W1", 0, 100, [], capacity=40, recorded=10),
                 wh("W2", 500, 100, [b("B", 500, 100)])]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        self.assertEqual(r.transfers[0].transfer_quantity, 30)  # 40 capacity - 10 recorded
        self.assertEqual(r.decision_for("W1").supplementary_reorder_quantity, 0)
        r2 = allocate([wh("W1", 0, 100, [], capacity=25, recorded=5)])
        self.assertEqual(r2.decision_for("W1").reorder_quantity, 20)

    def test_expiry_prevention_limited_by_destination_consumption(self):
        # Source batch expires in 5 days with 100 at risk; destination sells 10/day.
        # Arrival after T days -> destination can use days T+1..5.
        snaps = [wh("W1", 0, 200, [], daily=10.0),
                 wh("W2", 300, 50, [b("E", 150, 5, excess=100), b("L", 150, 200)])]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        expiry = [t for t in r.transfers if t.transfer_type == ae.TYPE_EXPIRY]
        self.assertEqual(len(expiry), 1)
        self.assertEqual(expiry[0].batch_id, "E")
        self.assertEqual(expiry[0].transfer_quantity, (5 - T) * 10)
        self.assertAlmostEqual(expiry[0].destination_consumption_before_expiry, (5 - T) * 10)

    def test_destination_own_earlier_stock_reduces_window(self):
        snaps = [wh("W1", 25, 200, [b("OWN", 25, 4)], daily=10.0),
                 wh("W2", 300, 50, [b("E", 150, 5, excess=100), b("L", 150, 200)])]
        r = allocate(snaps)
        expiry = [t for t in r.transfers if t.transfer_type == ae.TYPE_EXPIRY]
        # window = demand days T+1..5 (30) - own stock expiring by day 5 (25) = 5
        self.assertEqual(expiry[0].transfer_quantity, 5)

    def test_expiry_prevention_even_to_destination_without_shortage(self):
        snaps = [wh("W1", 500, 100, [b("L1", 500, 200)], daily=20.0),
                 wh("W2", 300, 50, [b("E", 200, 6, excess=150), b("L", 100, 200)], daily=5.0)]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        (t,) = r.transfers
        self.assertEqual((t.source_warehouse, t.destination_warehouse, t.transfer_type),
                         ("W2", "W1", ae.TYPE_EXPIRY))
        self.assertEqual(t.transfer_quantity, (6 - T) * 20)
        self.assertEqual(r.decision_for("W1").recommended_action, ae.ACTION_TRANSFER)

    def test_at_risk_units_do_not_count_as_normal_excess(self):
        # W2: usable 300, required 100, but 150 units will expire unused and are
        # ineligible (expire before arrival), so true surplus is only 50.
        snaps = [wh("W1", 0, 500, []),
                 wh("W2", 300, 100, [b("E", 160, T, excess=150), b("L", 140, 200)])]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        self.assertEqual(sum(t.transfer_quantity for t in r.transfers), 50)

    def test_fefo_prefers_earlier_expiring_source_batch(self):
        snaps = [wh("W1", 0, 50, []),
                 wh("W2", 500, 100, [b("LATE", 500, 150)]),
                 wh("W3", 500, 100, [b("EARLY", 500, 40)])]
        r = allocate(snaps)
        self.assertEqual(r.transfers[0].batch_id, "EARLY")

    def test_high_risk_destination_served_first(self):
        snaps = [wh("W1", 90, 100, [b("A", 90, 100)], risk=config.RISK_MEDIUM),
                 wh("W2", 0, 100, [], risk=config.RISK_HIGH),
                 wh("W3", 150, 100, [b("S", 150, 100)])]
        r = allocate(snaps)
        self.assertEqual(r.transfers[0].destination_warehouse, "W2")
        self.assertEqual(r.decision_for("W1").recommended_action, ae.ACTION_REORDER)

    def test_no_cross_shipping_in_expiry_pass(self):
        snaps = [wh("W1", 0, 60, [], daily=10.0),
                 wh("W2", 400, 100, [b("S", 400, 200)], daily=10.0),
                 wh("W3", 300, 50, [b("E", 200, 8, excess=150), b("L", 100, 200)], daily=2.0)]
        r = allocate(snaps)
        check_invariants(self, r, snaps)
        senders = {t.source_warehouse for t in r.transfers}
        for t in r.transfers:
            if t.transfer_type == ae.TYPE_EXPIRY and t.destination_warehouse in senders:
                self.fail(f"{t.destination_warehouse} both sends and receives expiry stock")

    def test_mixed_skus_rejected(self):
        with self.assertRaises(ae.AllocationError):
            allocate([wh("W1", 0, 10, [], sku="A"), wh("W2", 0, 10, [], sku="B")])


def flat_forecast_provider(db_path):
    """Stand-in forecast (flat historical mean) used ONLY to exercise constraints on real stock data."""
    def provider(sku, w):
        mean = db.get_demand_history(sku, w, db_path=db_path)["demand_qty"].mean()
        return pd.DataFrame({"date": pd.date_range(SIM + pd.Timedelta(days=1), periods=H),
                             "sku_id": sku, "warehouse_id": w, "forecast_demand": mean})
    return provider


class RealDataConstraintTests(unittest.TestCase):
    """Ayush's real inventory + batches for all 30 SKUs; constraints only, no expected actions."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.provider = staticmethod(flat_forecast_provider(cls.db_path))

    def test_all_skus_respect_constraints(self):
        skus = sorted({s for s, _ in db.list_sku_warehouse_pairs(db_path=self.db_path)})
        self.assertEqual(len(skus), 30)
        decisions = 0
        for sku in skus:
            snaps = [ae.build_snapshot(sku, w, self.provider(sku, w), db_path=self.db_path)
                     for w in db.get_inventory_for_sku(sku, db_path=self.db_path)["warehouse_id"]]
            r = allocate(snaps)
            with self.subTest(sku=sku):
                check_invariants(self, r, snaps)
            decisions += len(r.decisions)
        self.assertEqual(decisions, 180)

    def test_expired_stock_never_transferred(self):
        snaps = [ae.build_snapshot("M001", w, self.provider("M001", w), db_path=self.db_path)
                 for w in db.get_inventory_for_sku("M001", db_path=self.db_path)["warehouse_id"]]
        r = allocate(snaps)
        batches = pd.read_csv(config.BATCHES_CSV, parse_dates=["expiry_date"]).set_index("batch_id")
        for t in r.transfers:
            self.assertGreater(batches.loc[t.batch_id, "expiry_date"], SIM + pd.Timedelta(days=T))


class M001GoldenScenarioTest(unittest.TestCase):
    """Real Chronos-2 forecast (saved or generated) and the real database. No expected result is hard-coded."""

    @classmethod
    def setUpClass(cls):
        from database.database import DatabaseNotInitializedError
        from ml.model import ModelUnavailableError
        try:
            cls.result = ae.allocate_sku("M001")
        except (ModelUnavailableError, DatabaseNotInitializedError) as exc:
            raise unittest.SkipTest(f"Real forecast unavailable: {exc}")

    def test_w002_decision_is_consistent(self):
        d = self.result.decision_for("W002")
        self.assertIn(d.recommended_action, ae.ACTIONS)
        if d.shortage_before > 0:
            self.assertNotEqual(d.recommended_action, ae.ACTION_NO_ACTION)
        for t in self.result.transfers_to("W002"):
            self.assertGreater(t.days_to_expiry, T)
            self.assertLessEqual(t.transfer_quantity, t.destination_consumption_before_expiry + 1e-6)
        print("\nM001/W002:", d.recommended_action, "|", d.reason)
        for t in self.result.transfers_to("W002"):
            print(f"  {t.source_warehouse} {t.batch_id} x{t.transfer_quantity} ({t.transfer_type})")


if __name__ == "__main__":
    unittest.main()
