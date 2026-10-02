"""
Phase 9 tests: final action rules, alert generation and ordering, a
constraint sweep over all 180 items on real stock data, and the M001 golden
scenario with the real Chronos-2 forecast (skipped when unavailable).
"""

import math
import unittest

import pandas as pd

import config
from database import database as db
from decision_engine import alert_engine as al
from decision_engine.allocation_engine import (ACTION_NO_ACTION, ACTION_REORDER, ACTION_TRANSFER,
                                               TYPE_EXPIRY, allocate)
from decision_engine.decision_engine import evaluate_all, evaluate_sku
from decision_engine.final_decision import ROLE_DESTINATION, ROLE_SOURCE, build_final_decisions
from tests.helpers import make_temp_db
from tests.test_allocation import b, flat_forecast_provider, wh


def state(snap, risk=None, e1=False, expiry_risk="LOW", excess=0.0, min_threshold=0):
    """Minimal pre-allocation state matching a WarehouseSnapshot."""
    return {
        "sku_id": snap.sku_id, "warehouse_id": snap.warehouse_id,
        "risk_status": risk or snap.risk_status, "risk_reason": "r",
        "e1_threshold_alert": e1, "e1_reason": "e1",
        "expiry_risk": expiry_risk, "expiry_reason": "x",
        "potential_expiry_excess": excess, "near_expiry_quantity": 0,
        "current_stock": snap.recorded_stock, "expired_stock": 0, "usable_stock": snap.usable_stock,
        "min_threshold": min_threshold, "safety_stock": snap.safety_stock, "capacity": snap.capacity,
        "lead_time_days": 5, "average_daily_demand": 10.0, "lead_time_demand": 50.0,
        "required_stock": snap.required_stock, "projected_inventory": snap.usable_stock - 50.0,
        "days_of_stock": 1.0,
    }


def decide(snaps, **per_wh):
    states = {s.warehouse_id: state(s, **per_wh.get(s.warehouse_id, {})) for s in snaps}
    result = allocate(snaps)
    return build_final_decisions(states, result), result


class FinalActionTests(unittest.TestCase):

    def test_full_transfer_coverage_is_transfer(self):
        snaps = [wh("W1", 20, 100, [b("A", 20, 100)]), wh("W2", 300, 120, [b("B", 300, 100)])]
        d, _ = decide(snaps)
        self.assertEqual(d["W1"].final_action, ACTION_TRANSFER)
        self.assertTrue(d["W1"].transfer_recommended)
        self.assertEqual(d["W1"].transfer_role, ROLE_DESTINATION)
        self.assertEqual(d["W1"].reorder_quantity, 0)
        self.assertEqual(d["W1"].remaining_shortage, 0)
        # Surplus-only source: its shipment is the destination's action.
        self.assertEqual(d["W2"].final_action, ACTION_NO_ACTION)
        self.assertEqual(d["W2"].transfer_role, ROLE_SOURCE)
        self.assertFalse(d["W2"].transfer_recommended)

    def test_partial_transfer_then_reorder(self):
        snaps = [wh("W1", 0, 100.4, []), wh("W2", 150, 120, [b("B", 150, 100)])]
        d, r = decide(snaps)
        x = d["W1"]
        self.assertEqual(x.final_action, ACTION_REORDER)
        self.assertTrue(x.transfer_recommended)
        self.assertEqual(x.transfer_in_quantity, 30)
        self.assertAlmostEqual(x.remaining_shortage, 70.4)
        self.assertEqual(x.reorder_quantity, math.ceil(70.4))

    def test_reorder_without_transfer(self):
        d, _ = decide([wh("W1", 10, 60, [b("A", 10, 100)])])
        self.assertEqual(d["W1"].final_action, ACTION_REORDER)
        self.assertFalse(d["W1"].transfer_recommended)
        self.assertEqual(d["W1"].reorder_quantity, 50)

    def test_no_action(self):
        d, r = decide([wh("W1", 200, 100, [b("A", 200, 100)]), wh("W2", 200, 100, [b("B", 200, 100)])])
        self.assertEqual(r.transfers, [])
        for x in d.values():
            self.assertEqual(x.final_action, ACTION_NO_ACTION)
            self.assertEqual((x.reorder_quantity, x.transfer_recommended), (0, False))
            self.assertTrue(x.reason)

    def test_expiry_prevention_source_is_transfer(self):
        snaps = [wh("W1", 500, 100, [b("L1", 500, 200)], daily=20.0),
                 wh("W2", 300, 50, [b("E", 200, 6, excess=150), b("L", 100, 200)], daily=5.0)]
        d, r = decide(snaps, W2={"expiry_risk": "HIGH", "excess": 150})
        self.assertTrue(any(t.transfer_type == TYPE_EXPIRY for t in r.transfers))
        self.assertEqual(d["W2"].final_action, ACTION_TRANSFER)
        self.assertTrue(d["W2"].transfer_recommended)
        self.assertEqual(d["W1"].final_action, ACTION_TRANSFER)  # receives stock that would expire

    def test_exactly_one_action_each(self):
        snaps = [wh("W1", 0, 100, []), wh("W2", 150, 120, [b("B", 150, 100)]),
                 wh("W3", 500, 100, [b("C", 500, 100)])]
        d, _ = decide(snaps)
        self.assertEqual(set(d), {"W1", "W2", "W3"})
        for x in d.values():
            self.assertIn(x.final_action, (ACTION_TRANSFER, ACTION_REORDER, ACTION_NO_ACTION))


class AlertTests(unittest.TestCase):

    def test_e1_independent_of_p1(self):
        snaps = [wh("W1", 200, 100, [b("A", 200, 100)])]
        d, _ = decide(snaps, W1={"risk": config.RISK_LOW, "e1": True, "min_threshold": 250})
        types = [a.alert_type for a in al.alerts_for_decision(d["W1"])]
        self.assertEqual(types, [al.E1_THRESHOLD])
        self.assertEqual(al.alerts_for_decision(d["W1"])[0].value, 50)

    def test_p1_without_e1(self):
        snaps = [wh("W1", 10, 60, [b("A", 10, 100)])]
        d, _ = decide(snaps, W1={"risk": config.RISK_HIGH, "e1": False})
        types = {a.alert_type for a in al.alerts_for_decision(d["W1"])}
        self.assertIn(al.STOCKOUT_RISK_HIGH, types)
        self.assertNotIn(al.E1_THRESHOLD, types)
        self.assertIn(al.REORDER_RECOMMENDED, types)

    def test_expiry_alert_levels(self):
        snaps = [wh("W1", 200, 100, [b("A", 200, 100)]), wh("W2", 200, 100, [b("B", 200, 100)])]
        d, _ = decide(snaps, W1={"expiry_risk": "HIGH", "excess": 40.0},
                      W2={"expiry_risk": "MEDIUM", "excess": 25.0})
        a1 = [a for a in al.alerts_for_decision(d["W1"]) if a.alert_type.startswith("EXPIRY")]
        a2 = [a for a in al.alerts_for_decision(d["W2"]) if a.alert_type.startswith("EXPIRY")]
        self.assertEqual((a1[0].alert_type, a1[0].severity, a1[0].value), (al.EXPIRY_RISK_HIGH, "HIGH", 40.0))
        self.assertEqual((a2[0].alert_type, a2[0].severity, a2[0].value), (al.EXPIRY_RISK_MEDIUM, "MEDIUM", 25.0))
        self.assertIn("cannot be moved", a1[0].message)

    def test_low_expiry_risk_no_alert(self):
        d, _ = decide([wh("W1", 200, 100, [b("A", 200, 100)])])
        self.assertEqual(al.alerts_for_decision(d["W1"]), [])

    def test_alert_ordering(self):
        def mk(t, v, sku="S", w="W"):
            return al.Alert(sku, w, al.SEVERITY[t], t, "m", v, "u", "X", al.DISPLAY_RANK[t])
        alerts = [mk(al.REORDER_RECOMMENDED, 999), mk(al.E1_THRESHOLD, 500),
                  mk(al.EXPIRY_RISK_MEDIUM, 10), mk(al.STOCKOUT_RISK_MEDIUM, 10),
                  mk(al.EXPIRY_RISK_HIGH, 5), mk(al.STOCKOUT_RISK_HIGH, 1),
                  mk(al.STOCKOUT_RISK_HIGH, 50), mk(al.TRANSFER_RECOMMENDED, 3)]
        ordered = [(a.alert_type, a.value) for a in al.sort_alerts(alerts)]
        self.assertEqual(ordered[:6], [
            (al.STOCKOUT_RISK_HIGH, 50), (al.STOCKOUT_RISK_HIGH, 1), (al.EXPIRY_RISK_HIGH, 5),
            (al.STOCKOUT_RISK_MEDIUM, 10), (al.EXPIRY_RISK_MEDIUM, 10), (al.E1_THRESHOLD, 500)])
        self.assertEqual({t for t, _ in ordered[6:]}, {al.REORDER_RECOMMENDED, al.TRANSFER_RECOMMENDED})

    def test_every_alert_has_required_fields(self):
        snaps = [wh("W1", 0, 100.4, []), wh("W2", 150, 120, [b("B", 150, 100)])]
        d, _ = decide(snaps, W1={"risk": config.RISK_HIGH, "e1": True, "min_threshold": 300})
        for a in al.generate_alerts(d.values()):
            for f in ("sku_id", "warehouse_id", "severity", "alert_type", "message", "recommended_action"):
                self.assertTrue(getattr(a, f))
            self.assertEqual(a.recommended_action, d[a.warehouse_id].final_action)


class RealDataDecisionTests(unittest.TestCase):
    """All 180 items on Ayush's data with a stand-in forecast; checks structure, not specific actions."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.decisions, cls.transfers, cls.alerts = evaluate_all(
            flat_forecast_provider(cls.db_path), db_path=cls.db_path)

    def test_every_item_has_exactly_one_decision(self):
        d = self.decisions
        self.assertEqual(len(d), 180)
        self.assertFalse(d.duplicated(["sku_id", "warehouse_id"]).any())
        self.assertTrue(d.final_action.isin([ACTION_TRANSFER, ACTION_REORDER, ACTION_NO_ACTION]).all())
        self.assertTrue((d.reason.str.len() > 0).all())

    def test_action_consistency(self):
        d = self.decisions
        reorder = d[d.final_action == ACTION_REORDER]
        self.assertTrue((reorder.remaining_shortage > 0).all())
        cap_room = reorder.capacity - reorder.current_stock - reorder.transfer_in_quantity
        expected = [min(math.ceil(r - 1e-9), c) for r, c in zip(reorder.remaining_shortage, cap_room)]
        self.assertEqual(reorder.reorder_quantity.tolist(), expected)
        no_action = d[d.final_action == ACTION_NO_ACTION]
        self.assertTrue((no_action.remaining_shortage == 0).all())
        self.assertTrue((no_action.transfer_in_quantity == 0).all())
        self.assertTrue((no_action.expiry_transfer_out_quantity == 0).all())
        transfer = d[d.final_action == ACTION_TRANSFER]
        self.assertTrue(((transfer.transfer_in_quantity > 0) | (transfer.expiry_transfer_out_quantity > 0)).all())
        self.assertTrue((transfer.remaining_shortage == 0).all())

    def test_alerts_match_decisions(self):
        d = self.decisions.set_index(["sku_id", "warehouse_id"])
        a = self.alerts
        self.assertTrue(a.alert_type.isin(list(al.DISPLAY_RANK)).all())
        self.assertTrue(a.display_rank.is_monotonic_increasing)
        e1 = a[a.alert_type == al.E1_THRESHOLD]
        self.assertEqual(len(e1), int(d.e1_threshold_alert.sum()))
        high = a[a.alert_type == al.STOCKOUT_RISK_HIGH]
        self.assertEqual(len(high), int((d.risk_status == config.RISK_HIGH).sum()))
        reorder = a[a.alert_type == al.REORDER_RECOMMENDED]
        self.assertEqual(len(reorder), int((d.final_action == ACTION_REORDER).sum()))
        for row in a.itertuples():
            self.assertEqual(row.recommended_action, d.loc[(row.sku_id, row.warehouse_id), "final_action"])


class M001GoldenDecisionTest(unittest.TestCase):
    """Real Chronos-2 forecast + real database. Checks the golden scenario's structure honestly."""

    @classmethod
    def setUpClass(cls):
        from database.database import DatabaseNotInitializedError
        from ml.model import ModelUnavailableError
        try:
            cls.decisions, cls.transfers, cls.alerts = evaluate_sku("M001")
        except (ModelUnavailableError, DatabaseNotInitializedError) as exc:
            raise unittest.SkipTest(f"Real forecast unavailable: {exc}")

    def test_w002(self):
        d = self.decisions["W002"]
        self.assertEqual(d.risk_status, config.RISK_HIGH)
        self.assertTrue(d.transfer_recommended)
        self.assertGreater(d.transfer_in_quantity, 0)
        if d.remaining_shortage > 0:
            self.assertEqual(d.final_action, ACTION_REORDER)
            self.assertEqual(d.reorder_quantity, math.ceil(d.remaining_shortage - 1e-9))
        types = {a.alert_type for a in self.alerts if a.warehouse_id == "W002"}
        self.assertTrue({al.STOCKOUT_RISK_HIGH, al.E1_THRESHOLD, al.TRANSFER_RECOMMENDED} <= types)
        print(f"\nM001/W002: {d.final_action}, transfer_in {d.transfer_in_quantity}, "
              f"remaining {d.remaining_shortage}, reorder {d.reorder_quantity}")

    def test_w004_expiry_alert(self):
        d = self.decisions["W004"]
        types = {a.alert_type for a in self.alerts if a.warehouse_id == "W004"}
        if d.expiry_risk in ("HIGH", "MEDIUM"):
            self.assertTrue(types & {al.EXPIRY_RISK_HIGH, al.EXPIRY_RISK_MEDIUM})
        if d.expiry_transfer_out_quantity:
            self.assertIn(al.TRANSFER_RECOMMENDED, types)
            self.assertEqual(d.final_action, ACTION_TRANSFER)


if __name__ == "__main__":
    unittest.main()
