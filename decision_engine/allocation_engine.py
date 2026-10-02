"""
Phase 8: cross-warehouse allocation engine (owner: team lead).

Transparent, rule-based, deterministic. No optimisation solver and no ML.
The engine runs per SKU across all of that SKU's warehouses, so the same
source stock is never promised to two destinations.

INPUTS PER WAREHOUSE (from Phases 5-7, formulas unchanged)
    usable_stock, required_stock (= lead-time demand + safety stock),
    P1 risk, capacity, recorded stock, Chronos-2 forecast, and usable batches
    with potential_expiry_excess and transfer_eligible.

DEFINITIONS
    shortage       = max(0, required_stock - usable_stock)       (destination)
    source_excess  = max(0, usable_stock - at_risk_units - required_stock)
                     at_risk_units = usable units still forecast to expire unused at
                     the source. They cannot cover the source's own demand, so they
                     are excluded; this keeps the source at or above its required
                     stock even after its expiry-risk units have moved.
    capacity_room  = capacity - recorded_stock - inbound so far
                     (recorded stock includes expired units, which still take
                     up physical space until written off)
    transit        = config.TRANSFER_TRANSIT_DAYS (simulation assumption;
                     supplier lead_time_days is never used for transfers)
    arrival window = a transferred batch with d days to expiry can serve
                     destination demand on forecast days transit+1 .. d
    destination consumption before expiry (for a batch expiring on day d)
                   = destination forecast demand for days transit+1..d
                     - destination's own usable stock expiring on or before day d
                     - units already sent to it that expire on or before day d
                     (conservative: earlier-expiring stock is used first, FEFO)

TRANSFER TYPES
    EXPIRY_PREVENTION  batch has potential_expiry_excess > 0 at its source.
        qty <= min(remaining expiry excess, destination consumption before
                   expiry, capacity_room)
        Allowed even when the source is above its required stock, because
        this stock would otherwise expire unused at the source.
    SHORTAGE_EXCESS    normal surplus from a source above its required stock.
        qty <= min(destination shortage, source_excess, batch remaining,
                   destination consumption before expiry, capacity_room)
        The source never drops below its required stock.

    Every transferred batch must be usable (expiry_date > SIMULATION_DATE) and
    transfer-eligible (days_to_expiry > transit). Quantities are whole units.

ORDER OF DECISIONS
    Pass 1: destinations with a shortage, HIGH risk before MEDIUM, larger
            shortage first. For each one:
              (a) expiry-risk batches from other warehouses, earliest expiry first
              (b) then normal source excess, earliest-expiring batch first
    Pass 2: any expiry excess still left is offered to other warehouses that
            can consume it before expiry (waste prevention), earliest expiry
            first, destination with the most room first. Warehouses that have
            already shipped this SKU out are skipped (no cross-shipping).

ACTIONS (one per warehouse)
    TRANSFER   the warehouse receives at least one transfer. Any shortage
               still left is reported as supplementary_reorder_quantity.
    REORDER    shortage > 0 and no eligible transfer exists.
               reorder_quantity = ceil(shortage), capped at capacity_room.
    NO_ACTION  no shortage and no inbound transfer. Outbound transfers are
               noted in the reason.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

import config
from decision_engine.expiry_engine import cumulative_forecast_demand

ACTION_TRANSFER = "TRANSFER"
ACTION_REORDER = "REORDER"
ACTION_NO_ACTION = "NO_ACTION"
ACTIONS = (ACTION_TRANSFER, ACTION_REORDER, ACTION_NO_ACTION)

TYPE_EXPIRY = "EXPIRY_PREVENTION"
TYPE_EXCESS = "SHORTAGE_EXCESS"

_RISK_ORDER = {config.RISK_HIGH: 0, config.RISK_MEDIUM: 1, config.RISK_LOW: 2}
_EPS = 1e-9


class AllocationError(Exception):
    """Inputs to the allocation engine are inconsistent."""


@dataclass
class WarehouseSnapshot:
    """Everything the allocation engine needs about one SKU at one warehouse."""
    sku_id: str
    warehouse_id: str
    recorded_stock: int
    usable_stock: int
    required_stock: float
    safety_stock: int
    capacity: int
    risk_status: str
    forecast: np.ndarray
    batches: list[dict] = field(default_factory=list)  # usable batches only

    @property
    def shortage(self) -> float:
        return max(0.0, self.required_stock - self.usable_stock)


@dataclass
class Transfer:
    sku_id: str
    source_warehouse: str
    destination_warehouse: str
    batch_id: str
    transfer_quantity: int
    transfer_type: str
    expiry_date: str
    days_to_expiry: int
    destination_consumption_before_expiry: float
    source_usable_before: int
    source_usable_after: int
    destination_usable_before: int
    destination_usable_after: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WarehouseDecision:
    sku_id: str
    warehouse_id: str
    recommended_action: str
    risk_status: str
    usable_stock_before: int
    usable_stock_after: int
    required_stock: float
    shortage_before: float
    transferred_in: int
    transferred_out: int
    remaining_shortage: float
    reorder_quantity: int
    supplementary_reorder_quantity: int
    source_warehouses: list[str]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AllocationResult:
    sku_id: str
    transfers: list[Transfer]
    decisions: dict[str, WarehouseDecision]

    def decision_for(self, warehouse_id: str) -> WarehouseDecision:
        return self.decisions[warehouse_id]

    def transfers_to(self, warehouse_id: str) -> list[Transfer]:
        return [t for t in self.transfers if t.destination_warehouse == warehouse_id]


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class _Allocator:
    """Mutable bookkeeping for one SKU. Every change goes through _record()."""

    def __init__(self, snapshots: list[WarehouseSnapshot]):
        if not snapshots:
            raise AllocationError("No warehouse snapshots supplied")
        skus = {s.sku_id for s in snapshots}
        if len(skus) != 1:
            raise AllocationError(f"Snapshots must share one SKU, got {sorted(skus)}")
        ids = [s.warehouse_id for s in snapshots]
        if len(set(ids)) != len(ids):
            raise AllocationError("Duplicate warehouse snapshots")

        self.sku_id = snapshots[0].sku_id
        self.transit = config.TRANSFER_TRANSIT_DAYS
        self.wh = {s.warehouse_id: s for s in sorted(snapshots, key=lambda s: s.warehouse_id)}
        self.usable = {w: s.usable_stock for w, s in self.wh.items()}
        self.inbound_qty = {w: 0 for w in self.wh}
        self.outbound_qty = {w: 0 for w in self.wh}
        self.inbound_batches: dict[str, list[tuple[int, int]]] = {w: [] for w in self.wh}  # (days, qty)
        self.remaining_shortage = {w: s.shortage for w, s in self.wh.items()}
        self.batch_left: dict[tuple[str, str], int] = {}
        self.expiry_left: dict[tuple[str, str], float] = {}
        self.source_use: dict[tuple[str, str], float] = {}  # expected consumption at its own warehouse
        for w, s in self.wh.items():
            for b in s.batches:
                if not b.get("usable", True) or b["days_to_expiry"] <= 0:
                    raise AllocationError(f"{w}: unusable batch {b['batch_id']} passed to allocation")
                self.batch_left[(w, b["batch_id"])] = int(b["quantity"])
                self.expiry_left[(w, b["batch_id"])] = float(b.get("potential_expiry_excess") or 0.0)
                self.source_use[(w, b["batch_id"])] = float(b.get("expected_consumption") or 0.0)
        self.transfers: list[Transfer] = []

    # ---- constraint helpers -------------------------------------------------

    def capacity_room(self, w: str) -> int:
        s = self.wh[w]
        return max(0, int(s.capacity - s.recorded_stock - self.inbound_qty[w]))

    def at_risk_units(self, w: str) -> float:
        """
        Usable units at w still forecast to expire unused there. A partly
        at-risk unit counts as a whole unit (conservative), so rounding can
        never let a normal transfer eat into the source's required stock.
        """
        return float(sum(math.ceil(v - 1e-6) for (src, _), v in self.expiry_left.items()
                         if src == w and v > 1e-6))

    def source_excess(self, w: str) -> float:
        """
        Normal surplus: usable - required, not counting units still forecast to
        expire unused at w. Those units cannot protect the source's own required
        stock, so counting them would let a transfer leave the source short.
        """
        return max(0.0, self.usable[w] - self.at_risk_units(w) - self.wh[w].required_stock)

    def destination_window(self, dest: str, days: int) -> float:
        """Destination demand on days transit+1..days not already covered by earlier-expiring stock."""
        if days <= self.transit:
            return 0.0
        fc = self.wh[dest].forecast
        demand = (cumulative_forecast_demand(fc, days)[0]
                  - cumulative_forecast_demand(fc, self.transit)[0])
        own = sum(b["quantity"] for b in self.wh[dest].batches if b["days_to_expiry"] <= days)
        inbound = sum(q for d, q in self.inbound_batches[dest] if d <= days)
        return max(0.0, demand - own - inbound)

    @staticmethod
    def _eligible(batch: dict, transit: int) -> bool:
        return bool(batch.get("transfer_eligible")) and batch["days_to_expiry"] > transit

    # ---- recording ----------------------------------------------------------

    def _record(self, src: str, dest: str, batch: dict, qty: int, ttype: str,
                window: float, reason: str) -> None:
        if qty <= 0:
            return
        key = (src, batch["batch_id"])
        s_before, d_before = self.usable[src], self.usable[dest]
        self.usable[src] -= qty
        self.usable[dest] += qty
        self.outbound_qty[src] += qty
        self.inbound_qty[dest] += qty
        self.inbound_batches[dest].append((batch["days_to_expiry"], qty))
        self.batch_left[key] -= qty
        # Units left in the batch beyond what the source itself will use are still at risk.
        self.expiry_left[key] = min(self.expiry_left[key],
                                    max(0.0, self.batch_left[key] - self.source_use[key]))
        self.remaining_shortage[dest] = max(0.0, self.remaining_shortage[dest] - qty)
        self.transfers.append(Transfer(
            sku_id=self.sku_id, source_warehouse=src, destination_warehouse=dest,
            batch_id=batch["batch_id"], transfer_quantity=qty, transfer_type=ttype,
            expiry_date=batch["expiry_date"], days_to_expiry=batch["days_to_expiry"],
            destination_consumption_before_expiry=round(window, config.REPORT_DECIMALS),
            source_usable_before=s_before, source_usable_after=self.usable[src],
            destination_usable_before=d_before, destination_usable_after=self.usable[dest],
            reason=reason,
        ))

    # ---- candidate lists ----------------------------------------------------

    def _expiry_candidates(self, dest: str) -> list[tuple[str, dict]]:
        out = []
        for w, s in self.wh.items():
            if w == dest:
                continue
            for b in s.batches:
                key = (w, b["batch_id"])
                if (self._eligible(b, self.transit) and self.expiry_left[key] >= 1
                        and self.batch_left[key] >= 1):
                    out.append((w, b))
        return sorted(out, key=lambda x: (x[1]["days_to_expiry"], -self.expiry_left[(x[0], x[1]["batch_id"])],
                                          x[0], x[1]["batch_id"]))

    def _excess_candidates(self, dest: str) -> list[tuple[str, dict]]:
        out = []
        for w, s in self.wh.items():
            if w == dest or self.source_excess(w) < 1:
                continue
            for b in s.batches:
                if self._eligible(b, self.transit) and self.batch_left[(w, b["batch_id"])] >= 1:
                    out.append((w, b))
        return sorted(out, key=lambda x: (x[1]["days_to_expiry"], x[0], x[1]["batch_id"]))

    # ---- passes -------------------------------------------------------------

    def _try_expiry_transfer(self, src: str, dest: str, b: dict, why: str) -> None:
        key = (src, b["batch_id"])
        window = self.destination_window(dest, b["days_to_expiry"])
        qty = int(math.floor(min(self.expiry_left[key], self.batch_left[key], window,
                                 self.capacity_room(dest)) + _EPS))
        if qty >= 1:
            reason = (f"{why}: {src} batch {b['batch_id']} has {self.expiry_left[key]:,.2f} units "
                      f"forecast to expire unused in {b['days_to_expiry']} days; {dest} can consume "
                      f"{window:,.2f} units between arrival (day {self.transit}) and expiry")
            self._record(src, dest, b, qty, TYPE_EXPIRY, window, reason)

    def fill_shortages(self) -> None:
        order = sorted((w for w in self.wh if self.remaining_shortage[w] > _EPS),
                       key=lambda w: (_RISK_ORDER.get(self.wh[w].risk_status, 3),
                                      -self.remaining_shortage[w], w))
        for dest in order:
            for src, b in self._expiry_candidates(dest):
                if self.remaining_shortage[dest] <= _EPS:
                    break
                self._try_expiry_transfer(src, dest, b, f"Shortage at {dest} and expiry risk at {src}")
            for src, b in self._excess_candidates(dest):
                need = self.remaining_shortage[dest]
                if need <= _EPS:
                    break
                key = (src, b["batch_id"])
                window = self.destination_window(dest, b["days_to_expiry"])
                excess = self.source_excess(src)
                qty = int(min(math.ceil(need - _EPS),
                              math.floor(excess + _EPS),
                              self.batch_left[key],
                              math.floor(window + _EPS),
                              self.capacity_room(dest)))
                if qty >= 1:
                    reason = (f"Shortage at {dest} ({need:,.2f} units); {src} holds "
                              f"{excess:,.2f} units above its required stock "
                              f"({self.wh[src].required_stock:,.2f})")
                    self._record(src, dest, b, qty, TYPE_EXCESS, window, reason)

    def prevent_expiry(self) -> None:
        keys = sorted(((w, b) for w, s in self.wh.items() for b in s.batches
                       if self._eligible(b, self.transit) and self.expiry_left[(w, b["batch_id"])] >= 1),
                      key=lambda x: (x[1]["days_to_expiry"], x[0], x[1]["batch_id"]))
        for src, b in keys:
            # No cross-shipping: a warehouse that already sent this SKU out is not a destination here.
            dests = sorted((w for w in self.wh if w != src and self.outbound_qty[w] == 0),
                           key=lambda w: (-self.destination_window(w, b["days_to_expiry"]), w))
            for dest in dests:
                key = (src, b["batch_id"])
                if self.expiry_left[key] < 1 or self.batch_left[key] < 1:
                    break
                self._try_expiry_transfer(src, dest, b, "Expiry prevention")

    # ---- decisions ----------------------------------------------------------

    def decisions(self) -> dict[str, WarehouseDecision]:
        out = {}
        for w, s in self.wh.items():
            inbound = self.transfers_to(w)
            outbound = [t for t in self.transfers if t.source_warehouse == w]
            remaining = round(self.remaining_shortage[w], config.REPORT_DECIMALS)
            reorder = supplementary = 0
            if inbound:
                action = ACTION_TRANSFER
                srcs = sorted({t.source_warehouse for t in inbound})
                reason = (f"Receive {sum(t.transfer_quantity for t in inbound):,} units from "
                          f"{', '.join(srcs)} ({', '.join(sorted({t.transfer_type for t in inbound}))})")
                if remaining > _EPS:
                    supplementary = min(math.ceil(remaining - _EPS), self.capacity_room(w))
                    reason += (f"; {remaining:,.2f} units of shortage remain, so supplementary "
                               f"reorder of {supplementary:,} units")
                elif s.shortage > _EPS:
                    reason += f"; covers the full shortage of {s.shortage:,.2f} units"
            elif remaining > _EPS:
                action = ACTION_REORDER
                reorder = min(math.ceil(remaining - _EPS), self.capacity_room(w))
                reason = (f"Shortage of {remaining:,.2f} units (required {s.required_stock:,.2f}, "
                          f"usable {s.usable_stock:,}) and no other warehouse has eligible "
                          f"excess or expiry-risk stock")
                if reorder < math.ceil(remaining - _EPS):
                    reason += f"; reorder capped at remaining capacity ({reorder:,} units)"
            else:
                action = ACTION_NO_ACTION
                reason = (f"Usable stock ({s.usable_stock:,}) covers required stock "
                          f"({s.required_stock:,.2f})")
            if outbound:
                reason += (f"; supplies {sum(t.transfer_quantity for t in outbound):,} units to "
                           f"{', '.join(sorted({t.destination_warehouse for t in outbound}))}")
            out[w] = WarehouseDecision(
                sku_id=self.sku_id, warehouse_id=w, recommended_action=action,
                risk_status=s.risk_status, usable_stock_before=s.usable_stock,
                usable_stock_after=self.usable[w], required_stock=round(s.required_stock, config.REPORT_DECIMALS),
                shortage_before=round(s.shortage, config.REPORT_DECIMALS),
                transferred_in=self.inbound_qty[w], transferred_out=self.outbound_qty[w],
                remaining_shortage=remaining, reorder_quantity=int(reorder),
                supplementary_reorder_quantity=int(supplementary),
                source_warehouses=sorted({t.source_warehouse for t in inbound}), reason=reason,
            )
        return out

    def transfers_to(self, w: str) -> list[Transfer]:
        return [t for t in self.transfers if t.destination_warehouse == w]


def allocate(snapshots: list[WarehouseSnapshot]) -> AllocationResult:
    """Run the allocation rules for one SKU across all its warehouses."""
    engine = _Allocator(snapshots)
    engine.fill_shortages()
    engine.prevent_expiry()
    return AllocationResult(sku_id=engine.sku_id, transfers=engine.transfers,
                            decisions=engine.decisions())


# ------------------------------------------------------------------
# Database-backed entry point
# ------------------------------------------------------------------

def build_snapshot(sku_id: str, warehouse_id: str, forecast, db_path=None) -> WarehouseSnapshot:
    """Snapshot from the existing Phase 5-7 calculations (formulas unchanged)."""
    from database import database as db
    from decision_engine.expiry_engine import evaluate_expiry
    from decision_engine.forecast_utils import validated_forecast_values
    from decision_engine.inventory_calculator import calculate_inventory
    from decision_engine.risk_engine import assess_risk

    calc = calculate_inventory(sku_id, warehouse_id, forecast, db_path=db_path)
    risk = assess_risk(calc)
    expiry = evaluate_expiry(sku_id, warehouse_id, forecast, db_path=db_path)
    item = db.get_inventory_item(sku_id, warehouse_id, db_path=db_path)
    return WarehouseSnapshot(
        sku_id=sku_id, warehouse_id=warehouse_id,
        recorded_stock=calc.current_stock, usable_stock=calc.usable_stock,
        required_stock=calc.required_stock, safety_stock=calc.safety_stock,
        capacity=item["capacity"], risk_status=risk["risk_status"],
        forecast=validated_forecast_values(forecast, sku_id, warehouse_id),
        batches=[b for b in expiry.batch_details if b["usable"]],
    )


def allocate_sku(sku_id: str, forecast_provider=None, db_path=None) -> AllocationResult:
    """
    Allocation for every warehouse stocking `sku_id`.
    forecast_provider(sku_id, warehouse_id) -> forecast DataFrame
    (defaults to integration.ml_service.get_forecast).
    """
    from database import database as db
    if forecast_provider is None:
        from integration.ml_service import get_forecast as forecast_provider
    warehouses = db.get_inventory_for_sku(sku_id, db_path=db_path)["warehouse_id"].tolist()
    if not warehouses:
        raise AllocationError(f"No inventory records for SKU {sku_id}")
    snapshots = [build_snapshot(sku_id, w, forecast_provider(sku_id, w), db_path=db_path)
                 for w in warehouses]
    return allocate(snapshots)
