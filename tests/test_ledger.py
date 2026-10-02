"""E1 ledger: SALE / RECEIPT / ADJUSTMENT / TRANSFER_OUT, FEFO, no negative stock, expiry write-off."""

import unittest

import pandas as pd

from operations.ledger import (ADJUSTMENT, EXPIRY_WRITE_OFF, INVENTORY_CORRECTION, RECEIPT, SALE,
                               SUPPLIER_RECEIPT, TRANSFER_OUT, TRANSFER_RECEIPT, InventoryLedger,
                               LedgerError)


def ledger(*batches):
    df = pd.DataFrame([{"batch_id": b, "sku_id": "S", "warehouse_id": "W", "quantity": q, "expiry_date": e}
                       for b, q, e in batches])
    return InventoryLedger.from_batches(df, run_id="test")


class LedgerTests(unittest.TestCase):

    def test_sale_decreases_stock_fefo(self):
        lg = ledger(("LATE", 50, "2026-12-01"), ("EARLY", 30, "2026-10-10"))
        row = lg.record_sale("2026-10-01", "S", "W", 40, reference="r1")
        self.assertEqual((row["txn_type"], row["quantity"], row["requested_quantity"]), (SALE, -40, 40))
        self.assertEqual(row["batch_id"], "EARLY:30;LATE:10")
        self.assertEqual((lg.batch_quantity("S", "W", "EARLY"), lg.batch_quantity("S", "W", "LATE")), (0, 40))
        self.assertEqual(row["on_hand_after"], 40)
        self.assertEqual(row["reference"], "r1")

    def test_sale_never_negative(self):
        lg = ledger(("A", 10, "2026-12-01"))
        row = lg.record_sale("2026-10-01", "S", "W", 25)
        self.assertEqual((row["quantity"], row["unfulfilled_quantity"]), (-10, 15))
        self.assertEqual(lg.on_hand("S", "W"), 0)
        row2 = lg.record_sale("2026-10-02", "S", "W", 5)
        self.assertEqual((row2["quantity"], row2["unfulfilled_quantity"], row2["on_hand_after"]), (0, 5, 0))
        with self.assertRaises(LedgerError):
            lg.record_sale("2026-10-02", "S", "W", -1)

    def test_expired_batch_not_sold(self):
        lg = ledger(("OLD", 100, "2026-09-30"), ("NEW", 5, "2026-12-01"))
        row = lg.record_sale("2026-10-01", "S", "W", 20)
        self.assertEqual((row["quantity"], row["unfulfilled_quantity"]), (-5, 15))
        self.assertEqual(lg.batch_quantity("S", "W", "OLD"), 100)  # still on hand until written off

    def test_receipt_increases_stock(self):
        lg = ledger(("A", 10, "2026-12-01"))
        row = lg.record_receipt("2026-10-03", "S", "W", 25, "T1", "2027-01-01", TRANSFER_RECEIPT, reference="TRF")
        self.assertEqual((row["txn_type"], row["txn_subtype"], row["quantity"], row["on_hand_after"]),
                         (RECEIPT, TRANSFER_RECEIPT, 25, 35))
        lg.record_receipt("2026-10-04", "S", "W", 5, "T1", "2027-01-01", SUPPLIER_RECEIPT)
        self.assertEqual(lg.batch_quantity("S", "W", "T1"), 30)
        with self.assertRaises(LedgerError):
            lg.record_receipt("2026-10-04", "S", "W", 0, "X", "2027-01-01", SUPPLIER_RECEIPT)
        with self.assertRaises(LedgerError):
            lg.record_receipt("2026-10-04", "S", "W", 5, "T1", "2027-02-01", SUPPLIER_RECEIPT)  # expiry clash

    def test_adjustment_both_directions_with_reason(self):
        lg = ledger(("A", 10, "2026-12-01"))
        up = lg.record_adjustment("2026-10-02", "S", "W", 4, reason="Cycle count found 4 extra", batch_id="A")
        self.assertEqual((up["txn_type"], up["txn_subtype"], up["quantity"], up["on_hand_after"]),
                         (ADJUSTMENT, INVENTORY_CORRECTION, 4, 14))
        down = lg.record_adjustment("2026-10-02", "S", "W", -3, reason="Damaged units")
        self.assertEqual((down["quantity"], down["on_hand_after"], down["reason"]), (-3, 11, "Damaged units"))
        with self.assertRaises(LedgerError):
            lg.record_adjustment("2026-10-02", "S", "W", -50, reason="too much")
        with self.assertRaises(LedgerError):
            lg.record_adjustment("2026-10-02", "S", "W", -1, reason="")
        with self.assertRaises(LedgerError):
            lg.record_adjustment("2026-10-02", "S", "W", 5, reason="no batch")
        self.assertEqual(lg.on_hand("S", "W"), 11)

    def test_expiry_write_off(self):
        lg = ledger(("E1", 7, "2026-09-30"), ("E2", 3, "2026-10-02"), ("OK", 20, "2026-12-01"))
        rows = lg.write_off_expired("2026-09-30")
        self.assertEqual([(r["batch_id"], r["quantity"], r["txn_subtype"]) for r in rows],
                         [("E1", -7, EXPIRY_WRITE_OFF)])
        self.assertEqual(rows[0]["usable_after"], 23)  # end of day: usable = expiry > date
        rows = lg.write_off_expired("2026-10-02")
        self.assertEqual([(r["batch_id"], r["quantity"]) for r in rows], [("E2", -3)])
        self.assertEqual(lg.on_hand("S", "W"), 20)
        self.assertEqual(lg.write_off_expired("2026-10-02"), [])  # idempotent

    def test_transfer_out(self):
        lg = ledger(("B", 30, "2026-12-01"))
        row, shipped, expiry = lg.record_transfer_out("2026-10-01", "S", "W", "B", 12, reference="TRF")
        self.assertEqual((row["txn_type"], row["quantity"], shipped), (TRANSFER_OUT, -12, 12))
        self.assertEqual(expiry, pd.Timestamp("2026-12-01"))
        row, shipped, _ = lg.record_transfer_out("2026-10-01", "S", "W", "B", 40)
        self.assertEqual((shipped, row["unfulfilled_quantity"], lg.on_hand("S", "W")), (18, 22, 0))

    def test_reconciliation(self):
        lg = ledger(("A", 10, "2026-10-01"), ("B", 40, "2026-12-01"))
        lg.write_off_expired("2026-10-01")
        lg.record_sale("2026-10-02", "S", "W", 15)
        lg.record_receipt("2026-10-03", "S", "W", 9, "N", "2027-01-01", SUPPLIER_RECEIPT)
        self.assertEqual(lg.reconcile(), [])
        lg.batches[("S", "W")]["B"].quantity += 1  # tamper outside the ledger
        self.assertTrue(lg.reconcile())


if __name__ == "__main__":
    unittest.main()
