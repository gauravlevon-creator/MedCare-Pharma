"""
Phase 9: final decision per SKU + warehouse (owner: team lead).

Combines the unchanged outputs of earlier phases:
    inventory state (Phase 5-7): P1 risk, E1 alert, expiry risk, usable stock
    allocation (Phase 8):        transfers, remaining shortage, reorder quantity

Exactly one primary action per warehouse, applied in this order:

    REORDER    usable stock is still below required stock after all valid
               transfers. reorder_quantity = ceil(remaining shortage), as
               computed by the allocation engine (capped at capacity room).
               If transfers were also planned, transfer_recommended = True.
    TRANSFER   no reorder is needed and the warehouse
               - receives stock (the transfer fully covers any shortage), or
               - is the source of an EXPIRY_PREVENTION transfer (the planner
                 must ship the at-risk batch out).
    NO_ACTION  no shortage remains and no inbound or expiry-prevention
               transfer involves this warehouse. A warehouse that only
               supplies normal surplus to another warehouse stays NO_ACTION
               here; that transfer is the destination's action. transfer_role
               still shows it.

P1 risk, E1 alert and expiry risk are reported as they were BEFORE
allocation, so the dashboard shows why the action was needed. The seasonal
demand signal (decision_engine/seasonal_signal.py) is carried through for
visibility; it does not change any quantity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import config
from decision_engine.allocation_engine import (ACTION_NO_ACTION, ACTION_REORDER, ACTION_TRANSFER,
                                               TYPE_EXPIRY, AllocationResult)

ROLE_DESTINATION = "DESTINATION"
ROLE_SOURCE = "SOURCE"
ROLE_BOTH = "SOURCE_AND_DESTINATION"


@dataclass
class FinalDecision:
    sku_id: str
    warehouse_id: str
    final_action: str
    transfer_recommended: bool
    transfer_role: str | None
    transfer_in_quantity: int
    transfer_out_quantity: int
    expiry_transfer_out_quantity: int
    source_warehouses: list[str]
    destination_warehouses: list[str]
    reorder_quantity: int
    # state before allocation
    risk_status: str
    risk_reason: str
    e1_threshold_alert: bool
    e1_reason: str
    expiry_risk: str
    expiry_reason: str
    potential_expiry_excess: float
    near_expiry_quantity: int
    current_stock: int
    expired_stock: int
    usable_stock_before: int
    usable_stock_after: int
    min_threshold: int
    safety_stock: int
    capacity: int
    lead_time_days: int
    average_daily_demand: float
    lead_time_demand: float
    required_stock: float
    projected_inventory: float
    days_of_stock: float | None
    shortage_before: float
    remaining_shortage: float
    reason: str
    seasonal_signal: str = config.SEASON_SIGNAL_NORMAL
    seasonal_reason: str = ""
    seasonal_covariate_gap: bool = False
    transfers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _role(n_in: int, n_out: int) -> str | None:
    if n_in and n_out:
        return ROLE_BOTH
    if n_in:
        return ROLE_DESTINATION
    if n_out:
        return ROLE_SOURCE
    return None


def build_final_decisions(states: dict[str, dict], allocation: AllocationResult) -> dict[str, FinalDecision]:
    """
    states: warehouse_id -> evaluate_inventory_state() dict (pre-allocation).
    allocation: the Phase 8 result for the same SKU.
    """
    if set(states) != set(allocation.decisions):
        raise ValueError("States and allocation cover different warehouses")

    out: dict[str, FinalDecision] = {}
    for w, s in sorted(states.items()):
        alloc = allocation.decisions[w]
        inbound = [t for t in allocation.transfers if t.destination_warehouse == w]
        outbound = [t for t in allocation.transfers if t.source_warehouse == w]
        expiry_out = [t for t in outbound if t.transfer_type == TYPE_EXPIRY]
        q_in = sum(t.transfer_quantity for t in inbound)
        q_out = sum(t.transfer_quantity for t in outbound)
        q_exp_out = sum(t.transfer_quantity for t in expiry_out)
        reorder = alloc.reorder_quantity + alloc.supplementary_reorder_quantity
        srcs = sorted({t.source_warehouse for t in inbound})
        dests = sorted({t.destination_warehouse for t in outbound})

        if alloc.remaining_shortage > 0:
            action = ACTION_REORDER
            reason = (f"After {q_in:,} units of transfers, usable stock ({alloc.usable_stock_after:,}) is "
                      f"still {alloc.remaining_shortage:,.2f} below required stock "
                      f"({alloc.required_stock:,.2f}); reorder {reorder:,} units")
            if q_in:
                reason = f"Transfer {q_in:,} units from {', '.join(srcs)}, then reorder. " + reason
            if reorder < alloc.remaining_shortage:
                reason += " (reorder capped at remaining warehouse capacity)"
        elif q_in:
            action = ACTION_TRANSFER
            reason = f"Receive {q_in:,} units from {', '.join(srcs)}"
            reason += (f"; covers the full shortage of {alloc.shortage_before:,.2f} units"
                       if alloc.shortage_before > 0 else "; uses stock that would otherwise expire")
        elif q_exp_out:
            action = ACTION_TRANSFER
            reason = (f"Send {q_exp_out:,} expiry-risk units to {', '.join(sorted({t.destination_warehouse for t in expiry_out}))} "
                      f"before they expire unused")
        else:
            action = ACTION_NO_ACTION
            reason = (f"Usable stock ({s['usable_stock']:,}) covers required stock "
                      f"({s['required_stock']:,.2f})")
        surplus_out = [t for t in outbound if t.transfer_type != TYPE_EXPIRY]
        if surplus_out:
            reason += (f"; also sends {sum(t.transfer_quantity for t in surplus_out):,} surplus units to "
                       f"{', '.join(sorted({t.destination_warehouse for t in surplus_out}))}")
        if q_exp_out and action != ACTION_TRANSFER:
            reason += f"; sends {q_exp_out:,} expiry-risk units out"

        out[w] = FinalDecision(
            sku_id=s["sku_id"], warehouse_id=w, final_action=action,
            transfer_recommended=bool(q_in or q_exp_out),
            transfer_role=_role(q_in, q_out),
            transfer_in_quantity=q_in, transfer_out_quantity=q_out,
            expiry_transfer_out_quantity=q_exp_out,
            source_warehouses=srcs, destination_warehouses=dests,
            reorder_quantity=int(reorder) if action == ACTION_REORDER else 0,
            risk_status=s["risk_status"], risk_reason=s["risk_reason"],
            e1_threshold_alert=bool(s["e1_threshold_alert"]), e1_reason=s["e1_reason"],
            expiry_risk=s["expiry_risk"], expiry_reason=s["expiry_reason"],
            potential_expiry_excess=float(s["potential_expiry_excess"]),
            near_expiry_quantity=int(s["near_expiry_quantity"]),
            current_stock=int(s["current_stock"]), expired_stock=int(s["expired_stock"]),
            usable_stock_before=int(s["usable_stock"]), usable_stock_after=alloc.usable_stock_after,
            min_threshold=int(s["min_threshold"]), safety_stock=int(s["safety_stock"]),
            capacity=int(s["capacity"]), lead_time_days=int(s["lead_time_days"]),
            average_daily_demand=float(s["average_daily_demand"]),
            lead_time_demand=float(s["lead_time_demand"]),
            required_stock=float(s["required_stock"]),
            projected_inventory=float(s["projected_inventory"]),
            days_of_stock=s["days_of_stock"],
            shortage_before=alloc.shortage_before, remaining_shortage=alloc.remaining_shortage,
            reason=reason,
            seasonal_signal=s.get("seasonal_signal", config.SEASON_SIGNAL_NORMAL),
            seasonal_reason=s.get("seasonal_reason", ""),
            seasonal_covariate_gap=bool(s.get("seasonal_covariate_gap", False)),
            transfers=[t.to_dict() for t in inbound + outbound],
        )
    return out


__all__ = ["FinalDecision", "build_final_decisions", "ROLE_DESTINATION", "ROLE_SOURCE", "ROLE_BOTH",
           "ACTION_TRANSFER", "ACTION_REORDER", "ACTION_NO_ACTION", "config"]
