"""
E1 simulation policy tests: plan+reorder_point is the default; operational
reorders are labelled separately from the 2026-09-30 P1 plan and follow
the documented rule; the summary is saved for the dashboard; the CLI refuses
to report a stand-in forecast as a real result.
"""

import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import config
import main
from database import database as db
from decision_engine.decision_engine import evaluate_sku
from integration import ml_service
from operations import sales_simulation as sim
from tests.helpers import FakeChronosPipeline, make_temp_db
from tests.test_allocation import flat_forecast_provider

D = config.simulation_timestamp()


class PolicyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_path = make_temp_db()
        cls.decisions = pd.DataFrame([{"sku_id": "M001", "warehouse_id": "W002", "final_action": "REORDER",
                                       "reorder_quantity": 24}])
        cls.transfers = pd.DataFrame([{"sku_id": "M001", "source_warehouse": "W004",
                                       "destination_warehouse": "W002", "batch_id": "B00015",
                                       "transfer_quantity": 10}])
        cls.default = sim.run_simulation(days=30, decisions=cls.decisions, transfers=cls.transfers,
                                         db_path=cls.db_path)
        cls.plan_only = sim.run_simulation(days=30, decisions=cls.decisions, transfers=cls.transfers,
                                           db_path=cls.db_path, replenishment_policy="plan-only")

    def test_default_policy(self):
        self.assertEqual(config.E1_REPLENISHMENT_POLICY, "plan+reorder_point")
        self.assertEqual(sim.resolve_policy(None), "plan+reorder_point")
        self.assertEqual(self.default.policy, "plan+reorder_point")
        args = main.build_parser().parse_args(["simulate"])
        self.assertEqual(args.policy, "plan+reorder_point")

    def test_plan_only_and_alias(self):
        self.assertEqual(sim.resolve_policy("plan-only"), "plan-only")
        self.assertEqual(sim.resolve_policy("plan"), "plan-only")
        with self.assertRaises(sim.SimulationError):
            sim.resolve_policy("optimise")
        self.assertEqual(set(self.plan_only.purchase_orders.origin), {sim.ORIGIN_P1})

    def test_p1_plan_identical_under_both_policies(self):
        p1_default = self.default.purchase_orders[self.default.purchase_orders.origin == sim.ORIGIN_P1]
        pd.testing.assert_frame_equal(p1_default.reset_index(drop=True),
                                      self.plan_only.purchase_orders.reset_index(drop=True))
        out = lambda r: r.transactions[r.transactions.txn_type == "TRANSFER_OUT"].drop(columns=["run_id", "sequence"]).reset_index(drop=True)  # noqa: E731
        pd.testing.assert_frame_equal(out(self.default), out(self.plan_only))

    def test_operational_reorders_labelled_and_after_simulation_date(self):
        e1 = self.default.purchase_orders[self.default.purchase_orders.origin == sim.ORIGIN_E1]
        self.assertGreater(len(e1), 0)
        self.assertTrue(e1.po_reference.str.startswith("E1-OPS-PO:").all())
        self.assertTrue((pd.to_datetime(e1.order_date) > D).all())  # never on the P1 decision date
        rec = self.default.transactions[self.default.transactions.reference.fillna("").str.startswith("E1-OPS-PO:")]
        self.assertTrue(rec.reason.str.contains("not part of the P1 plan").all())

    def test_operational_reorder_quantity_follows_rule(self):
        demand = db.get_all_demand(db_path=self.db_path).groupby(["sku_id", "warehouse_id"]).demand_qty.mean()
        inv = db.get_all_inventory(db_path=self.db_path).set_index(["sku_id", "warehouse_id"])
        e1 = self.default.purchase_orders[self.default.purchase_orders.origin == sim.ORIGIN_E1]
        for po in e1.head(50).itertuples():
            usable, on_order, thr, target = map(int, re.findall(r"-?\d+", po.trigger)[:4])
            key = (po.sku_id, po.warehouse_id)
            self.assertEqual(thr, int(inv.loc[key, "min_threshold"]))
            self.assertLessEqual(usable + on_order, thr)
            lt = int(inv.loc[key, "lead_time_days"])
            self.assertEqual(po.lead_time_days, lt)
            self.assertEqual(target, math.ceil(thr + demand[key] * lt))
            self.assertEqual(po.quantity, target - (usable + on_order))
            self.assertEqual(pd.Timestamp(po.due_date), pd.Timestamp(po.order_date) + pd.Timedelta(days=lt))
            daily = self.default.daily
            row = daily[(daily.date == po.order_date) & (daily.sku_id == po.sku_id) & (daily.warehouse_id == po.warehouse_id)]
            self.assertEqual(int(row.usable.iloc[0]), usable)

    def test_reorder_point_improves_service_without_negative_stock(self):
        self.assertEqual(self.default.reconciliation_problems, [])
        self.assertTrue((self.default.transactions.on_hand_after >= 0).all())
        self.assertGreaterEqual(self.default.summary()["sales"]["fill_rate"],
                                self.plan_only.summary()["sales"]["fill_rate"])

    def test_summary_metrics_consistent_and_saved(self):
        s = self.default.summary()
        tx = self.default.transactions
        self.assertEqual(s["sales"]["units_requested"], s["sales"]["units_fulfilled"] + s["sales"]["units_lost"])
        self.assertEqual(s["e1_operational_reorders"]["orders_placed"],
                         int((self.default.purchase_orders.origin == sim.ORIGIN_E1).sum()))
        self.assertEqual(s["receipts"]["supplier_receipts_total"],
                         s["receipts"]["supplier_receipts_p1_plan"] + s["receipts"]["supplier_receipts_e1_operational"])
        self.assertEqual(s["receipts"]["transfer_receipts"], int((tx.txn_subtype == "TRANSFER_RECEIPT").sum()))
        self.assertEqual(s["adjustments"]["expired_units_written_off"],
                         s["adjustments"]["expired_units_written_off_opening"]
                         + s["adjustments"]["expired_units_written_off_during_simulation"])
        self.assertEqual(s["inventory"]["opening_on_hand_units"] + int(tx.quantity.sum()),
                         s["inventory"]["ending_on_hand_units"])
        with tempfile.TemporaryDirectory() as d:
            out = sim.write_summary(self.default, {"pipeline": "X"}, path=Path(d) / "s.json",
                                    po_path=Path(d) / "po.csv")
            saved = json.loads((Path(d) / "s.json").read_text())
            self.assertEqual(saved["sales"], out["sales"])
            self.assertEqual(saved["forecast_source"], {"pipeline": "X"})
            self.assertEqual(len(pd.read_csv(Path(d) / "po.csv")), len(self.default.purchase_orders))

    def test_phase9_decisions_unchanged_by_default_simulation(self):
        provider = flat_forecast_provider(self.db_path)
        before, _, _ = evaluate_sku("M001", provider, db_path=self.db_path)
        sim.run_simulation(days=30, db_path=self.db_path, persist=True, run_id="default-persist")
        after, _, _ = evaluate_sku("M001", provider, db_path=self.db_path)
        self.assertEqual({w: x.to_dict() for w, x in before.items()},
                         {w: x.to_dict() for w, x in after.items()})


class CliStandInGuardTests(unittest.TestCase):
    """`python main.py simulate` must not present a stand-in forecast as a real result."""

    def test_refuses_stand_in_then_labels_it_when_allowed(self):
        tmp = Path(tempfile.mkdtemp(prefix="medcare_cli_"))
        db_file = tmp / "cli.db"
        db.init_database(db_path=db_file)
        patches = dict(DATABASE_PATH=db_file, OUTPUT_DIR=tmp, FORECAST_OUTPUT_CSV=tmp / "f.csv",
                       FORECAST_METADATA_JSON=tmp / "f.json", TRANSACTION_LOG_CSV=tmp / "t.csv",
                       SIMULATED_INVENTORY_CSV=tmp / "d.csv", SIMULATION_SUMMARY_JSON=tmp / "s.json",
                       SIMULATION_PURCHASE_ORDERS_CSV=tmp / "po.csv", NOTIFICATION_LOG_CSV=tmp / "n.csv")
        with mock.patch.multiple(config, **patches):
            _, meta = ml_service.get_all_forecasts(refresh=True, pipeline=FakeChronosPipeline())
            self.assertEqual(meta["pipeline"], "FakeChronosPipeline")
            self.assertEqual(main.main(["simulate", "--days", "3"]), 2)
            self.assertFalse((tmp / "s.json").exists())
            self.assertEqual(main.main(["simulate", "--days", "3", "--allow-stand-in"]), 0)
            saved = json.loads((tmp / "s.json").read_text())
            self.assertFalse(saved["forecast_source"]["is_real_chronos2"])
            self.assertEqual(saved["replenishment_policy"], "plan+reorder_point")


if __name__ == "__main__":
    unittest.main()
