"""
E1 operational simulation: sales, receipts and adjustments over the 30 days
after SIMULATION_DATE.

ARCHITECTURE: P1 PLAN vs E1 OPERATIONS
    The Phase 9 decision engine is the AUTHORITATIVE P1 planning decision as of
    SIMULATION_DATE (2026-09-30). This module never changes it. It only
    simulates what happens operationally afterwards:

        2026-09-30  initial P1 planning decision (Phase 9)
              |     TRANSFER / REORDER / NO_ACTION
              v
        30-day operational simulation (this module)
              |     daily SALES consume inventory
              |     open transfers/reorders arrive after transit / lead time
              |     if usable stock + open orders <= min_threshold
              |         -> the simulation places an additional OPERATIONAL reorder
              v     continue simulation

    Every purchase order and receipt is labelled with its origin:
        P1_PLAN                 the 2026-09-30 Phase 9 recommendation
        E1_OPERATIONAL_REORDER  generated later by the daily reorder-point rule;
                                NOT part of the 2026-09-30 P1 recommendation

POLICIES (E1_REPLENISHMENT_POLICY / --policy)
    plan+reorder_point (default)  P1 plan + daily E1 reorder-point rule
    plan-only                     P1 plan only (comparison/testing; "plan" is an alias)

E1 REORDER-POINT RULE (uses only existing data: thresholds, lead times, demand)
    at the end of each simulated day, per item:
        if usable + on_order <= min_threshold:
            order ceil(min_threshold + avg_daily_demand x lead_time_days) - (usable + on_order)
            received after lead_time_days
    avg_daily_demand is the item's historical mean demand_qty, the same figure
    used to calibrate simulated sales. on_order counts every open transfer and
    reorder, P1 or E1. The rule does not run on SIMULATION_DATE itself; that
    day belongs to the P1 plan.

SALES vs DEMAND
    demand.csv drives forecasting and P1 planning; sales.csv drives E1 stock
    consumption. They are separate datasets and are never merged.

REQUESTED SALES (per item, per simulated day d)
    pattern   = units_sold on d - SALES_LOOKBACK_DAYS in the derived sales
                history (same days last year)
    requested = round(pattern x scale)
    scale     = mean(demand_qty) / mean(units_sold) over the item's history
                when SALES_SIMULATION_SCALING = "demand_calibrated" (default),
                1.0 when "raw".
    Calibration keeps the sales pattern (weekday effects, price/promo dips)
    at the volume that stock levels and thresholds were sized for. Raw sales
    run about 2.8x demand and would empty most warehouses within days.

INVENTORY-AWARE FULFILMENT
    fulfilled = min(requested, usable stock that day); the rest is logged as
    unfulfilled_quantity (lost sale). Stock never goes negative.

DAILY ORDER OF EVENTS (D = SIMULATION_DATE)
    D (end of day)  write off batches with expiry <= D       ADJUSTMENT/EXPIRY_WRITE_OFF
                    record opening E1 state (items already in breach)
    each day d = D+1 .. D+N:
      1. receipts due today                                  RECEIPT
         P1 transfers:  dispatched D+1, received D+1+TRANSFER_TRANSIT_DAYS
                        (same batch_id and expiry as at the source)
         P1 reorders:   placed D+1, received D+1+lead_time_days
         E1 reorders:   placed d, received d+lead_time_days
         (supplier batches: expiry = receipt date + SUPPLIER_RECEIPT_SHELF_LIFE_DAYS,
          a documented simulation assumption)
      2. P1 transfer dispatches (day D+1 only)               TRANSFER_OUT
      3. sales                                               SALE
      4. end of day: write off batches expiring on d         ADJUSTMENT/EXPIRY_WRITE_OFF
      5. E1 check: usable stock (for later days) <= min_threshold
         -> BREACH event when an item enters breach, RECOVERED when it leaves
      6. E1 reorder-point rule (plan+reorder_point only)

Receipts falling after the simulated window are listed as in transit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from database import database as db
from operations.ledger import (SUPPLIER_RECEIPT, TRANSFER_RECEIPT, InventoryLedger, new_run_id)

BREACH = "BREACH"
RECOVERED = "RECOVERED"

POLICY_PLAN_ONLY = "plan-only"
POLICY_REORDER_POINT = "plan+reorder_point"
POLICIES = (POLICY_REORDER_POINT, POLICY_PLAN_ONLY)
_POLICY_ALIASES = {"plan": POLICY_PLAN_ONLY}

ORIGIN_P1 = "P1_PLAN"
ORIGIN_E1 = "E1_OPERATIONAL_REORDER"
PO_COLUMNS = ["po_reference", "origin", "sku_id", "warehouse_id", "order_date", "quantity",
              "lead_time_days", "due_date", "trigger"]


def resolve_policy(policy: str | None) -> str:
    p = policy or config.E1_REPLENISHMENT_POLICY
    p = _POLICY_ALIASES.get(p, p)
    if p not in POLICIES:
        raise SimulationError(f"Unknown replenishment policy {policy!r}; use one of {POLICIES}")
    return p


class SimulationError(Exception):
    """The simulation inputs are inconsistent."""


@dataclass
class SimulationResult:
    run_id: str
    policy: str
    start_date: str
    end_date: str
    scaling: str
    transactions: pd.DataFrame
    daily: pd.DataFrame
    threshold_events: pd.DataFrame
    purchase_orders: pd.DataFrame
    in_transit: pd.DataFrame
    as_of: str = config.SIMULATION_DATE
    opening_on_hand: dict = field(default_factory=dict)
    reconciliation_problems: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Headline metrics for the dashboard (saved to outputs/simulation_summary.json)."""
        tx, daily, po, ev = self.transactions, self.daily, self.purchase_orders, self.threshold_events
        sales = tx[tx.txn_type == "SALE"]
        rec = tx[tx.txn_type == "RECEIPT"]
        adj = tx[tx.txn_type == "ADJUSTMENT"]
        wo = adj[adj.txn_subtype == "EXPIRY_WRITE_OFF"]
        start = daily[daily.date == daily.date.min()]
        end = daily[daily.date == daily.date.max()]
        sim_days = daily[daily.date > daily.date.min()]
        requested = int(sales.requested_quantity.sum())
        fulfilled = int(-sales.quantity.sum())
        lost_by_item = (sales.groupby(["sku_id", "warehouse_id"]).unfulfilled_quantity.sum()
                        .sort_values(ascending=False))
        p1_po, e1_po = po[po.origin == ORIGIN_P1], po[po.origin == ORIGIN_E1]

        def receipts(origin_prefix: str, subtype: str) -> pd.DataFrame:
            return rec[(rec.txn_subtype == subtype) & rec.reference.fillna("").str.startswith(origin_prefix)]

        return {
            "run_id": self.run_id,
            "simulation_date": config.SIMULATION_DATE,
            "period": f"{self.start_date} .. {self.end_date}",
            "days": int(sim_days.date.nunique()),
            "replenishment_policy": self.policy,
            "sales_scaling": self.scaling,
            "sales": {
                "units_requested": requested,
                "units_fulfilled": fulfilled,
                "units_lost": int(sales.unfulfilled_quantity.sum()),
                "fill_rate": round(fulfilled / requested, 4) if requested else None,
            },
            "p1_plan_2026_09_30": {
                "note": "Authoritative Phase 9 decision as of the simulation date; executed, not changed",
                "transfers_dispatched": int((tx.txn_type == "TRANSFER_OUT").sum()),
                "transfer_units_dispatched": int(-tx[tx.txn_type == "TRANSFER_OUT"].quantity.sum()),
                "reorders_placed": int(len(p1_po)),
                "reorder_units": int(p1_po.quantity.sum()),
            },
            "e1_operational_reorders": {
                "note": "Generated during the simulation by the E1 reorder-point rule; NOT part of the P1 plan",
                "orders_placed": int(len(e1_po)),
                "units_ordered": int(e1_po.quantity.sum()),
                "items_reordered": int(e1_po[["sku_id", "warehouse_id"]].drop_duplicates().shape[0]),
            },
            "receipts": {
                "transfer_receipts": int(len(receipts("P1-TRF", "TRANSFER_RECEIPT"))),
                "transfer_units_received": int(receipts("P1-TRF", "TRANSFER_RECEIPT").quantity.sum()),
                "supplier_receipts_p1_plan": int(len(receipts("P1-PO", "SUPPLIER_RECEIPT"))),
                "supplier_receipts_e1_operational": int(len(receipts("E1-OPS-PO", "SUPPLIER_RECEIPT"))),
                "supplier_receipts_total": int((rec.txn_subtype == "SUPPLIER_RECEIPT").sum()),
                "supplier_units_received": int(rec[rec.txn_subtype == "SUPPLIER_RECEIPT"].quantity.sum()),
                "still_in_transit_at_end": int(len(self.in_transit)),
                "units_in_transit_at_end": int(self.in_transit.quantity.sum()) if len(self.in_transit) else 0,
            },
            "adjustments": {
                "count": int(len(adj)),
                "units_net": int(adj.quantity.sum()),
                "expiry_write_offs": int(len(wo)),
                "expired_units_written_off": int(-wo.quantity.sum()),
                "expired_units_written_off_opening": int(-wo[wo.txn_date == self.as_of].quantity.sum()),
                "expired_units_written_off_during_simulation": int(-wo[wo.txn_date != self.as_of].quantity.sum()),
                "inventory_corrections": int((adj.txn_subtype == "INVENTORY_CORRECTION").sum()),
            },
            "inventory": {
                "opening_on_hand_units": int(sum(self.opening_on_hand.values())),
                "usable_units_after_opening_write_off": int(start.usable.sum()),
                "ending_on_hand_units": int(end.on_hand.sum()),
                "ending_usable_units": int(end.usable.sum()),
            },
            "e1_threshold": {
                "items_in_breach_at_start": int(start.below_threshold.sum()),
                "items_in_breach_at_end": int(end.below_threshold.sum()),
                "breach_events": int((ev.event == BREACH).sum()),
                "new_breaches_during_simulation": int(((ev.event == BREACH) & (ev.date != self.as_of)).sum()),
                "recoveries": int((ev.event == RECOVERED).sum()),
                "items_ever_in_breach": int(daily[daily.below_threshold][["sku_id", "warehouse_id"]]
                                            .drop_duplicates().shape[0]),
            },
            "stockouts": {
                "item_days_with_lost_sales": int((sim_days.unfulfilled > 0).sum()),
                "items_with_any_lost_sales": int((lost_by_item > 0).sum()),
                "item_days_ending_with_zero_usable": int((sim_days.usable == 0).sum()),
                "top_items_by_lost_units": [
                    {"sku_id": k[0], "warehouse_id": k[1], "units_lost": int(v)}
                    for k, v in lost_by_item[lost_by_item > 0].head(10).items()],
            },
            "reconciled": not self.reconciliation_problems,
            "transactions": int(len(tx)),
        }


def _scale_factors(scaling: str, db_path=None) -> dict[tuple[str, str], float]:
    if scaling == "raw":
        return {}
    if scaling != "demand_calibrated":
        raise SimulationError(f"Unknown SALES_SIMULATION_SCALING {scaling!r}")
    demand = db.get_all_demand(db_path=db_path).groupby(["sku_id", "warehouse_id"])["demand_qty"].mean()
    sales = db.get_all_sales(db_path=db_path).groupby(["sku_id", "warehouse_id"])["units_sold"].mean()
    factors = {}
    for key in demand.index:
        s = float(sales.get(key, 0.0))
        factors[key] = float(demand[key]) / s if s > 0 else 0.0
    return factors


def requested_sales(days: int, scaling: str | None = None, db_path=None) -> pd.DataFrame:
    """Requested units per item for each simulated day (see module docstring)."""
    scaling = scaling or config.SALES_SIMULATION_SCALING
    start = config.simulation_timestamp() + pd.Timedelta(days=1)
    dates = pd.date_range(start, periods=days, freq="D")
    lookback = pd.Timedelta(days=config.SALES_LOOKBACK_DAYS)
    hist = db.get_all_sales(db_path=db_path)
    hist = hist[(hist.date >= dates[0] - lookback) & (hist.date <= dates[-1] - lookback)].copy()
    hist["date"] = hist["date"] + lookback
    factors = _scale_factors(scaling, db_path)
    hist["scale"] = [factors.get(k, 1.0) if scaling != "raw" else 1.0
                     for k in zip(hist.sku_id, hist.warehouse_id)]
    hist["requested"] = np.floor(hist.units_sold * hist.scale + 0.5).astype(int)
    items = pd.DataFrame(db.list_sku_warehouse_pairs(db_path=db_path), columns=["sku_id", "warehouse_id"])
    full = items.merge(pd.DataFrame({"date": dates}), how="cross")
    out = full.merge(hist[["date", "sku_id", "warehouse_id", "units_sold", "scale", "requested"]],
                     on=["date", "sku_id", "warehouse_id"], how="left")
    out["pattern_missing"] = out["requested"].isna()
    out[["units_sold", "requested"]] = out[["units_sold", "requested"]].fillna(0).astype(int)
    out["scale"] = out["scale"].fillna(0.0)
    return out.sort_values(["date", "sku_id", "warehouse_id"]).reset_index(drop=True)


def run_simulation(days: int | None = None, decisions: pd.DataFrame | None = None,
                   transfers: pd.DataFrame | None = None, scaling: str | None = None,
                   db_path=None, run_id: str | None = None, persist: bool = False,
                   replenishment_policy: str | None = None) -> SimulationResult:
    """
    Simulate `days` days after SIMULATION_DATE.
    decisions / transfers: the Phase 9 outputs (evaluate_all). Leave them as None
    to simulate sales and expiry only, with no planned receipts.
    persist=True writes the ledger rows to the inventory_transactions table.
    """
    days = int(days or config.E1_SIMULATION_DAYS)
    if days < 1:
        raise SimulationError("days must be >= 1")
    scaling = scaling or config.SALES_SIMULATION_SCALING
    policy = resolve_policy(replenishment_policy)
    D = config.simulation_timestamp()
    dates = pd.date_range(D + pd.Timedelta(days=1), periods=days, freq="D")
    ledger = InventoryLedger.from_database(db_path=db_path, run_id=run_id or new_run_id())
    inventory = db.get_all_inventory(db_path=db_path).set_index(["sku_id", "warehouse_id"])
    requested = requested_sales(days, scaling, db_path).set_index(["date", "sku_id", "warehouse_id"])
    avg_demand = (db.get_all_demand(db_path=db_path).groupby(["sku_id", "warehouse_id"])["demand_qty"]
                  .mean().to_dict())

    # ---- schedule the plan ------------------------------------------------
    dispatch_day = dates[0]
    dispatches, receipts, orders = [], [], []
    if transfers is not None and len(transfers):
        for t in transfers.itertuples(index=False):
            ref = f"P1-TRF:{t.sku_id}:{t.source_warehouse}->{t.destination_warehouse}:{t.batch_id}"
            dispatches.append({"sku": t.sku_id, "src": t.source_warehouse, "dst": t.destination_warehouse,
                               "batch": t.batch_id, "qty": int(t.transfer_quantity), "ref": ref})
    if decisions is not None and len(decisions):
        for d in decisions.itertuples(index=False):
            if d.final_action == "REORDER" and int(d.reorder_quantity) > 0:
                lt = int(inventory.loc[(d.sku_id, d.warehouse_id), "lead_time_days"])
                due = dispatch_day + pd.Timedelta(days=lt)
                po = f"P1-PO:{d.sku_id}:{d.warehouse_id}:{D.date()}"
                orders.append({"po_reference": po, "origin": ORIGIN_P1, "sku_id": d.sku_id,
                               "warehouse_id": d.warehouse_id, "order_date": dispatch_day.strftime("%Y-%m-%d"),
                               "quantity": int(d.reorder_quantity), "lead_time_days": lt,
                               "due_date": due.strftime("%Y-%m-%d"),
                               "trigger": f"Phase 9 REORDER decision as of {D.date()}"})
                receipts.append({"date": due, "sku": d.sku_id, "wh": d.warehouse_id,
                                 "qty": int(d.reorder_quantity), "subtype": SUPPLIER_RECEIPT,
                                 "batch": f"SUP-{d.sku_id}-{d.warehouse_id}-{due.strftime('%Y%m%d')}",
                                 "expiry": due + pd.Timedelta(days=config.SUPPLIER_RECEIPT_SHELF_LIFE_DAYS),
                                 "ref": po, "origin": ORIGIN_P1,
                                 "reason": f"P1 plan reorder (Phase 9 decision as of {D.date()})"})

    # ---- day D: opening write-off and E1 state ---------------------------
    ledger.write_off_expired(D)
    daily_rows, events = [], []
    breach = {}
    for key in ledger.items():
        usable = ledger.usable(*key, D, end_of_day=True)
        thr = int(inventory.loc[key, "min_threshold"])
        breach[key] = usable <= thr
        daily_rows.append(_daily_row(D, key, ledger, usable, thr, 0, 0, 0))
        if breach[key]:
            events.append(_event(D, key, BREACH, usable, thr))

    # ---- simulate ----------------------------------------------------------
    for day in dates:
        for r in [r for r in receipts if r["date"] == day]:
            ledger.record_receipt(day, r["sku"], r["wh"], r["qty"], r["batch"], r["expiry"],
                                  r["subtype"], reference=r["ref"], reason=r["reason"])
        if day == dispatch_day:
            for t in dispatches:
                _, shipped, expiry = ledger.record_transfer_out(
                    day, t["sku"], t["src"], t["batch"], t["qty"], reference=t["ref"],
                    reason=f"P1 plan transfer to {t['dst']} (Phase 9 decision as of {D.date()})")
                if shipped:
                    receipts.append({"date": day + pd.Timedelta(days=config.TRANSFER_TRANSIT_DAYS),
                                     "sku": t["sku"], "wh": t["dst"], "qty": shipped,
                                     "subtype": TRANSFER_RECEIPT, "batch": t["batch"], "expiry": expiry,
                                     "ref": t["ref"], "origin": ORIGIN_P1,
                                     "reason": f"P1 plan transfer from {t['src']} (Phase 9 decision as of {D.date()})"})
        sold = {}
        for key in ledger.items():
            req = int(requested.loc[(day, *key), "requested"]) if (day, *key) in requested.index else 0
            row = ledger.record_sale(day, *key, req,
                                     reference=f"sales_pattern:{(day - pd.Timedelta(days=config.SALES_LOOKBACK_DAYS)).date()}")
            sold[key] = (req, -row["quantity"], row["unfulfilled_quantity"])
        ledger.write_off_expired(day)
        for key in ledger.items():
            usable = ledger.usable(*key, day, end_of_day=True)
            thr = int(inventory.loc[key, "min_threshold"])
            now = usable <= thr
            if now != breach[key]:
                events.append(_event(day, key, BREACH if now else RECOVERED, usable, thr))
                breach[key] = now
            daily_rows.append(_daily_row(day, key, ledger, usable, thr, *sold[key]))
            if policy == POLICY_REORDER_POINT:
                on_order = sum(r["qty"] for r in receipts
                               if r["date"] > day and (r["sku"], r["wh"]) == key)
                if usable + on_order <= thr:
                    lt = int(inventory.loc[key, "lead_time_days"])
                    target = math.ceil(thr + avg_demand.get(key, 0.0) * lt)
                    qty = target - (usable + on_order)
                    if qty > 0:
                        due = day + pd.Timedelta(days=lt)
                        po = f"E1-OPS-PO:{key[0]}:{key[1]}:{day.date()}"
                        orders.append({"po_reference": po, "origin": ORIGIN_E1, "sku_id": key[0],
                                       "warehouse_id": key[1], "order_date": day.strftime("%Y-%m-%d"),
                                       "quantity": int(qty), "lead_time_days": lt,
                                       "due_date": due.strftime("%Y-%m-%d"),
                                       "trigger": (f"usable {usable} + on order {on_order} <= min_threshold "
                                                   f"{thr}; order up to {target}")})
                        receipts.append({"date": due, "sku": key[0], "wh": key[1], "qty": int(qty),
                                         "subtype": SUPPLIER_RECEIPT,
                                         "batch": f"SUP-{key[0]}-{key[1]}-{due.strftime('%Y%m%d')}",
                                         "expiry": due + pd.Timedelta(days=config.SUPPLIER_RECEIPT_SHELF_LIFE_DAYS),
                                         "ref": po, "origin": ORIGIN_E1,
                                         "reason": f"E1 operational reorder placed {day.date()} (not part of the P1 plan)"})

    in_transit = pd.DataFrame([{"due_date": r["date"].strftime("%Y-%m-%d"), "sku_id": r["sku"],
                                "warehouse_id": r["wh"], "quantity": r["qty"], "type": r["subtype"],
                                "origin": r["origin"], "reference": r["ref"]}
                               for r in receipts if r["date"] > dates[-1]],
                              columns=["due_date", "sku_id", "warehouse_id", "quantity", "type", "origin",
                                       "reference"])
    result = SimulationResult(
        run_id=ledger.run_id, policy=policy, start_date=dates[0].strftime("%Y-%m-%d"),
        end_date=dates[-1].strftime("%Y-%m-%d"), scaling=scaling,
        transactions=ledger.to_frame(), daily=pd.DataFrame(daily_rows),
        threshold_events=pd.DataFrame(events, columns=["date", "sku_id", "warehouse_id", "event",
                                                       "usable_stock", "min_threshold"]),
        purchase_orders=pd.DataFrame(orders, columns=PO_COLUMNS),
        in_transit=in_transit, as_of=D.strftime("%Y-%m-%d"),
        opening_on_hand=dict(ledger.opening_on_hand), reconciliation_problems=ledger.reconcile(),
    )
    if persist:
        db.save_transactions(result.transactions.to_dict("records"), db_path=db_path)
    return result


def _daily_row(day, key, ledger, usable, thr, requested, sold, unfulfilled) -> dict:
    return {"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "sku_id": key[0], "warehouse_id": key[1],
            "on_hand": ledger.on_hand(*key), "usable": usable, "min_threshold": thr,
            "below_threshold": usable <= thr, "requested": requested, "sold": sold,
            "unfulfilled": unfulfilled}


def _event(day, key, event, usable, thr) -> dict:
    return {"date": pd.Timestamp(day).strftime("%Y-%m-%d"), "sku_id": key[0], "warehouse_id": key[1],
            "event": event, "usable_stock": usable, "min_threshold": thr}


def e1_alerts_from_events(events: pd.DataFrame, decisions: pd.DataFrame | None = None) -> list[dict]:
    """
    E1_THRESHOLD alerts (dicts in the notification engine's input format) for
    every BREACH event. The recommended action comes from the Phase 9
    decision when one is available.
    """
    actions = {}
    if decisions is not None and len(decisions):
        actions = {(r.sku_id, r.warehouse_id): r.final_action for r in decisions.itertuples(index=False)}
    out = []
    for e in events[events.event == BREACH].itertuples(index=False):
        out.append({
            "sku_id": e.sku_id, "warehouse_id": e.warehouse_id, "alert_type": "E1_THRESHOLD",
            "severity": "LOW", "alert_date": e.date,
            "recommended_action": actions.get((e.sku_id, e.warehouse_id), "REVIEW"),
            "message": (f"E1 threshold alert for {e.sku_id} at {e.warehouse_id} on {e.date}: usable stock "
                        f"{e.usable_stock:,} is at or below the minimum threshold {e.min_threshold:,}."),
        })
    return out


__all__ = ["run_simulation", "requested_sales", "e1_alerts_from_events", "SimulationResult",
           "SimulationError", "BREACH", "RECOVERED", "POLICY_PLAN_ONLY", "POLICY_REORDER_POINT",
           "POLICIES", "ORIGIN_P1", "ORIGIN_E1", "resolve_policy", "write_summary"]


def write_summary(result: SimulationResult, forecast_source: dict | None = None,
                  path=None, po_path=None) -> dict:
    """Save the summary JSON and the purchase-order list for the dashboard."""
    import json
    summary = result.summary()
    summary["forecast_source"] = forecast_source or {"note": "not recorded"}
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (path or config.SIMULATION_SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result.purchase_orders.to_csv(po_path or config.SIMULATION_PURCHASE_ORDERS_CSV, index=False)
    return summary
