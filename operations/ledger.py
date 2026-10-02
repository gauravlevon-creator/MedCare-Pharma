"""
E1 inventory ledger (owner: team lead; integration: Ruthie).

Batch-level, auditable stock ledger. Every stock change goes through one of
the record_* methods, which append a row to the transaction log. There is
no other way to change stock.

TRANSACTION TYPES
    SALE          stock leaves to customers (quantity < 0), earliest expiry first (FEFO)
    RECEIPT       stock arrives (quantity > 0)
                  subtype TRANSFER_RECEIPT  from another warehouse
                  subtype SUPPLIER_RECEIPT  from a supplier reorder
    ADJUSTMENT    deliberate correction with a reason (quantity +/-)
                  subtype EXPIRY_WRITE_OFF      expired batch removed
                  subtype INVENTORY_CORRECTION  manual count correction
    TRANSFER_OUT  stock dispatched to another warehouse (quantity < 0). This
                  extra type keeps the dispatch visible and pairs with the
                  destination's TRANSFER_RECEIPT.

INVARIANTS
    * On-hand and usable stock never go negative. A sale larger than usable
      stock is only partly fulfilled: quantity = what was available, and
      unfulfilled_quantity records the rest as a lost sale.
    * Stock is sellable on day d only from batches with expiry_date >= d.
      On the simulation date (end of day), usable = expiry_date > date,
      matching Phase 7.
    * A negative adjustment larger than the stock it targets is rejected
      (LedgerError). It is never silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

import config
from database import database as db

SALE = "SALE"
RECEIPT = "RECEIPT"
ADJUSTMENT = "ADJUSTMENT"
TRANSFER_OUT = "TRANSFER_OUT"
TXN_TYPES = (SALE, RECEIPT, ADJUSTMENT, TRANSFER_OUT)

TRANSFER_RECEIPT = "TRANSFER_RECEIPT"
SUPPLIER_RECEIPT = "SUPPLIER_RECEIPT"
EXPIRY_WRITE_OFF = "EXPIRY_WRITE_OFF"
INVENTORY_CORRECTION = "INVENTORY_CORRECTION"


class LedgerError(Exception):
    """A requested transaction would break a ledger invariant."""


@dataclass
class _Batch:
    batch_id: str
    quantity: int
    expiry_date: pd.Timestamp


@dataclass
class InventoryLedger:
    run_id: str
    batches: dict[tuple[str, str], dict[str, _Batch]] = field(default_factory=dict)
    opening_on_hand: dict[tuple[str, str], int] = field(default_factory=dict)
    transactions: list[dict] = field(default_factory=list)

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_batches(cls, batches: pd.DataFrame, run_id: str | None = None) -> "InventoryLedger":
        """Opening stock = recorded batches (expired ones included until written off)."""
        ledger = cls(run_id=run_id or new_run_id())
        for r in batches.itertuples(index=False):
            key = (str(r.sku_id), str(r.warehouse_id))
            qty = int(r.quantity)
            if qty < 0:
                raise LedgerError(f"Negative opening quantity for batch {r.batch_id}")
            ledger.batches.setdefault(key, {})[str(r.batch_id)] = _Batch(
                str(r.batch_id), qty, pd.Timestamp(r.expiry_date).normalize())
        ledger.opening_on_hand = {k: ledger.on_hand(*k) for k in ledger.batches}
        return ledger

    @classmethod
    def from_database(cls, db_path=None, run_id: str | None = None) -> "InventoryLedger":
        ledger = cls.from_batches(db.get_all_batches(db_path=db_path), run_id)
        for sku, wh in db.list_sku_warehouse_pairs(db_path=db_path):
            ledger.batches.setdefault((sku, wh), {})
            ledger.opening_on_hand.setdefault((sku, wh), 0)
        return ledger

    # ---- queries ------------------------------------------------------------

    def items(self) -> list[tuple[str, str]]:
        return sorted(self.batches)

    def on_hand(self, sku: str, wh: str) -> int:
        return sum(b.quantity for b in self.batches.get((sku, wh), {}).values())

    def usable(self, sku: str, wh: str, on_date, end_of_day: bool = False) -> int:
        """Sellable on on_date (expiry >= date); end_of_day=True -> usable for later days (expiry > date)."""
        d = pd.Timestamp(on_date).normalize()
        return sum(b.quantity for b in self.batches.get((sku, wh), {}).values()
                   if (b.expiry_date > d if end_of_day else b.expiry_date >= d))

    def batch_quantity(self, sku: str, wh: str, batch_id: str) -> int:
        b = self.batches.get((sku, wh), {}).get(batch_id)
        return b.quantity if b else 0

    def _fefo(self, sku: str, wh: str, on_date) -> list[_Batch]:
        d = pd.Timestamp(on_date).normalize()
        return sorted((b for b in self.batches.get((sku, wh), {}).values()
                       if b.quantity > 0 and b.expiry_date >= d),
                      key=lambda b: (b.expiry_date, b.batch_id))

    # ---- recording ----------------------------------------------------------

    def _log(self, date, sku, wh, txn_type, subtype, quantity, batch_id=None, requested=None,
             unfulfilled=0, reference=None, reason=None, end_of_day=False) -> dict:
        if txn_type not in TXN_TYPES:
            raise LedgerError(f"Unknown transaction type {txn_type}")
        d = pd.Timestamp(date).normalize()
        row = {
            "run_id": self.run_id, "sequence": len(self.transactions) + 1,
            "txn_date": d.strftime("%Y-%m-%d"), "sku_id": sku, "warehouse_id": wh,
            "batch_id": batch_id, "txn_type": txn_type, "txn_subtype": subtype,
            "quantity": int(quantity), "requested_quantity": requested,
            "unfulfilled_quantity": int(unfulfilled),
            "on_hand_after": self.on_hand(sku, wh),
            "usable_after": self.usable(sku, wh, d, end_of_day=end_of_day),
            "reference": reference, "reason": reason,
        }
        if row["on_hand_after"] < 0 or row["usable_after"] < 0:
            raise LedgerError(f"Negative stock for {sku}/{wh}")  # defensive; should be unreachable
        self.transactions.append(row)
        return row

    def record_sale(self, date, sku: str, wh: str, requested: int, reference: str | None = None) -> dict:
        """Fulfil up to `requested` units, earliest expiry first. Never negative."""
        if requested < 0:
            raise LedgerError("Requested sale quantity cannot be negative")
        remaining, used = int(requested), []
        for b in self._fefo(sku, wh, date):
            if remaining == 0:
                break
            take = min(b.quantity, remaining)
            b.quantity -= take
            remaining -= take
            used.append(f"{b.batch_id}:{take}")
        fulfilled = int(requested) - remaining
        return self._log(date, sku, wh, SALE, None, -fulfilled,
                         batch_id=";".join(used) or None, requested=int(requested),
                         unfulfilled=remaining, reference=reference,
                         reason="Insufficient usable stock: sale partly/not fulfilled" if remaining else None)

    def record_receipt(self, date, sku: str, wh: str, quantity: int, batch_id: str, expiry_date,
                       subtype: str, reference: str | None = None, reason: str | None = None) -> dict:
        if quantity <= 0:
            raise LedgerError("Receipt quantity must be positive")
        if subtype not in (TRANSFER_RECEIPT, SUPPLIER_RECEIPT):
            raise LedgerError(f"Unknown receipt subtype {subtype}")
        exp = pd.Timestamp(expiry_date).normalize()
        batches = self.batches.setdefault((sku, wh), {})
        if batch_id in batches:
            if batches[batch_id].expiry_date != exp:
                raise LedgerError(f"Batch {batch_id} already at {wh} with a different expiry")
            batches[batch_id].quantity += int(quantity)
        else:
            batches[batch_id] = _Batch(batch_id, int(quantity), exp)
        return self._log(date, sku, wh, RECEIPT, subtype, int(quantity), batch_id=batch_id,
                         reference=reference, reason=reason)

    def record_transfer_out(self, date, sku: str, wh: str, batch_id: str, quantity: int,
                            reference: str | None = None,
                            reason: str | None = None) -> tuple[dict, int, pd.Timestamp | None]:
        """Dispatch up to `quantity` of a specific batch. Returns (row, shipped, expiry)."""
        if quantity <= 0:
            raise LedgerError("Transfer quantity must be positive")
        b = self.batches.get((sku, wh), {}).get(batch_id)
        d = pd.Timestamp(date).normalize()
        available = b.quantity if (b and b.expiry_date >= d) else 0
        shipped = min(int(quantity), available)
        if shipped:
            b.quantity -= shipped
        row = self._log(date, sku, wh, TRANSFER_OUT, None, -shipped, batch_id=batch_id,
                        requested=int(quantity), unfulfilled=int(quantity) - shipped, reference=reference,
                        reason=(reason if shipped == quantity else
                                f"{reason + '; ' if reason else ''}batch had less usable stock than planned"))
        return row, shipped, (b.expiry_date if b else None)

    def record_adjustment(self, date, sku: str, wh: str, quantity: int, reason: str,
                          subtype: str = INVENTORY_CORRECTION, batch_id: str | None = None,
                          expiry_date=None, end_of_day: bool = False) -> dict:
        """
        Deliberate stock correction. Negative quantities come from `batch_id`, or
        earliest expiry first if no batch is named. They are rejected if larger than
        the stock available. Positive quantities need a batch_id (plus expiry_date
        if the batch is new).
        """
        if not reason:
            raise LedgerError("Adjustments require a reason")
        if quantity == 0:
            raise LedgerError("Adjustment quantity cannot be zero")
        batches = self.batches.setdefault((sku, wh), {})
        if quantity > 0:
            if not batch_id:
                raise LedgerError("Positive adjustments need a batch_id")
            if batch_id not in batches:
                if expiry_date is None:
                    raise LedgerError("New batch in adjustment needs an expiry_date")
                batches[batch_id] = _Batch(batch_id, 0, pd.Timestamp(expiry_date).normalize())
            batches[batch_id].quantity += int(quantity)
        else:
            need = -int(quantity)
            if batch_id:
                b = batches.get(batch_id)
                if b is None or b.quantity < need:
                    raise LedgerError(f"Adjustment of {quantity} exceeds stock in batch {batch_id}")
                b.quantity -= need
            else:
                if self.on_hand(sku, wh) < need:
                    raise LedgerError(f"Adjustment of {quantity} exceeds on-hand stock at {sku}/{wh}")
                for b in sorted(batches.values(), key=lambda x: (x.expiry_date, x.batch_id)):
                    take = min(b.quantity, need)
                    b.quantity -= take
                    need -= take
                    if need == 0:
                        break
        return self._log(date, sku, wh, ADJUSTMENT, subtype, int(quantity), batch_id=batch_id,
                         reason=reason, end_of_day=end_of_day)

    def write_off_expired(self, date, reference: str | None = None) -> list[dict]:
        """End of `date`: write off every batch with expiry_date <= date (one ADJUSTMENT each)."""
        d = pd.Timestamp(date).normalize()
        rows = []
        for (sku, wh) in self.items():
            for b in sorted(self.batches[(sku, wh)].values(), key=lambda x: (x.expiry_date, x.batch_id)):
                if b.quantity > 0 and b.expiry_date <= d:
                    rows.append(self.record_adjustment(
                        d, sku, wh, -b.quantity,
                        reason=f"Batch expired on {b.expiry_date.date()}; not usable after {d.date()}",
                        subtype=EXPIRY_WRITE_OFF, batch_id=b.batch_id, end_of_day=True))
        return rows

    # ---- outputs / checks ---------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.transactions, columns=db.TRANSACTION_COLUMNS)

    def reconcile(self) -> list[str]:
        """Opening + sum(changes) must equal closing for every item. Returns problems (empty = OK)."""
        problems = []
        tx = self.to_frame()
        change = tx.groupby(["sku_id", "warehouse_id"])["quantity"].sum() if len(tx) else pd.Series(dtype=int)
        for key in self.items():
            expected = self.opening_on_hand.get(key, 0) + int(change.get(key, 0))
            if expected != self.on_hand(*key):
                problems.append(f"{key}: opening+changes={expected}, closing={self.on_hand(*key)}")
            if any(b.quantity < 0 for b in self.batches[key].values()):
                problems.append(f"{key}: negative batch quantity")
        return problems


def new_run_id() -> str:
    return "sim-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


__all__ = ["InventoryLedger", "LedgerError", "SALE", "RECEIPT", "ADJUSTMENT", "TRANSFER_OUT",
           "TRANSFER_RECEIPT", "SUPPLIER_RECEIPT", "EXPIRY_WRITE_OFF", "INVENTORY_CORRECTION", "config"]
