"""
Phase 9: alert engine (owner: team lead).

Alerts are generated only from final decisions, which are themselves built
from Phase 5-8 outputs. There are no hard-coded alerts and no scoring model.

    STOCKOUT_RISK_HIGH    P1 risk HIGH before allocation      value: shortage (units)
    STOCKOUT_RISK_MEDIUM  P1 risk MEDIUM before allocation    value: shortage (units)
    EXPIRY_RISK_HIGH      expiry risk HIGH                    value: potential expiry excess
    EXPIRY_RISK_MEDIUM    expiry risk MEDIUM                  value: potential expiry excess
    E1_THRESHOLD          usable stock <= min threshold       value: units below threshold
                          (independent of P1)
    TRANSFER_RECOMMENDED  transfer_recommended is True        value: units in + expiry units out
    REORDER_RECOMMENDED   final action REORDER                value: reorder quantity

Display order (presentation only, not a score):
    1 STOCKOUT_RISK_HIGH  2 EXPIRY_RISK_HIGH  3 STOCKOUT_RISK_MEDIUM
    4 EXPIRY_RISK_MEDIUM  5 E1_THRESHOLD      6 TRANSFER / REORDER (informational)
Within a rank: larger supporting value first, then SKU and warehouse.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import config
from decision_engine.final_decision import FinalDecision

STOCKOUT_RISK_HIGH = "STOCKOUT_RISK_HIGH"
STOCKOUT_RISK_MEDIUM = "STOCKOUT_RISK_MEDIUM"
E1_THRESHOLD = "E1_THRESHOLD"
EXPIRY_RISK_HIGH = "EXPIRY_RISK_HIGH"
EXPIRY_RISK_MEDIUM = "EXPIRY_RISK_MEDIUM"
TRANSFER_RECOMMENDED = "TRANSFER_RECOMMENDED"
REORDER_RECOMMENDED = "REORDER_RECOMMENDED"

SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"
SEVERITY_INFO = "INFO"

DISPLAY_RANK = {
    STOCKOUT_RISK_HIGH: 1,
    EXPIRY_RISK_HIGH: 2,
    STOCKOUT_RISK_MEDIUM: 3,
    EXPIRY_RISK_MEDIUM: 4,
    E1_THRESHOLD: 5,
    TRANSFER_RECOMMENDED: 6,
    REORDER_RECOMMENDED: 6,
}

SEVERITY = {
    STOCKOUT_RISK_HIGH: SEVERITY_HIGH,
    EXPIRY_RISK_HIGH: SEVERITY_HIGH,
    STOCKOUT_RISK_MEDIUM: SEVERITY_MEDIUM,
    EXPIRY_RISK_MEDIUM: SEVERITY_MEDIUM,
    E1_THRESHOLD: SEVERITY_LOW,
    TRANSFER_RECOMMENDED: SEVERITY_INFO,
    REORDER_RECOMMENDED: SEVERITY_INFO,
}


@dataclass
class Alert:
    sku_id: str
    warehouse_id: str
    severity: str
    alert_type: str
    message: str
    value: float
    value_label: str
    recommended_action: str
    display_rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def _alert(d: FinalDecision, alert_type: str, message: str, value: float, label: str) -> Alert:
    return Alert(sku_id=d.sku_id, warehouse_id=d.warehouse_id, severity=SEVERITY[alert_type],
                 alert_type=alert_type, message=message,
                 value=round(float(value), config.REPORT_DECIMALS), value_label=label,
                 recommended_action=d.final_action, display_rank=DISPLAY_RANK[alert_type])


def alerts_for_decision(d: FinalDecision) -> list[Alert]:
    """All alerts implied by one final decision."""
    alerts: list[Alert] = []
    where = f"{d.sku_id} at {d.warehouse_id}"

    if d.risk_status in (config.RISK_HIGH, config.RISK_MEDIUM):
        atype = STOCKOUT_RISK_HIGH if d.risk_status == config.RISK_HIGH else STOCKOUT_RISK_MEDIUM
        alerts.append(_alert(
            d, atype,
            f"{d.risk_status} stock-out risk for {where}: projected usable inventory "
            f"{d.projected_inventory:,.2f} after {d.lead_time_days}-day lead-time demand of "
            f"{d.lead_time_demand:,.2f} (safety stock {d.safety_stock}). Shortage "
            f"{d.shortage_before:,.2f} units."
            + (f" Seasonal demand signal: {d.seasonal_signal} (historical seasonal demand pattern)."
               if d.seasonal_signal != config.SEASON_SIGNAL_NORMAL else ""),
            d.shortage_before, "shortage_units"))

    if d.expiry_risk in ("HIGH", "MEDIUM"):
        atype = EXPIRY_RISK_HIGH if d.expiry_risk == "HIGH" else EXPIRY_RISK_MEDIUM
        moved = d.expiry_transfer_out_quantity
        tail = (f" {moved:,} units scheduled for transfer to {', '.join(d.destination_warehouses)}."
                if moved else (f" It cannot be moved: it expires within the {config.TRANSFER_TRANSIT_DAYS}-day "
                               f"transfer transit or no other warehouse can use it before expiry."))
        alerts.append(_alert(
            d, atype,
            f"{d.expiry_risk} expiry risk for {where}: {d.potential_expiry_excess:,.2f} usable units are "
            f"forecast to expire before they are consumed.{tail}",
            d.potential_expiry_excess, "potential_expiry_excess_units"))

    if d.e1_threshold_alert:
        gap = d.min_threshold - d.usable_stock_before
        alerts.append(_alert(
            d, E1_THRESHOLD,
            f"E1 threshold alert for {where}: usable stock {d.usable_stock_before:,} is at or below "
            f"the minimum threshold {d.min_threshold:,}.",
            gap, "units_below_threshold"))

    if d.transfer_recommended:
        parts = []
        if d.transfer_in_quantity:
            parts.append(f"receive {d.transfer_in_quantity:,} units from {', '.join(d.source_warehouses)}")
        if d.expiry_transfer_out_quantity:
            parts.append(f"send {d.expiry_transfer_out_quantity:,} expiry-risk units")
        alerts.append(_alert(
            d, TRANSFER_RECOMMENDED,
            f"Transfer recommended for {where}: {'; '.join(parts)}.",
            d.transfer_in_quantity + d.expiry_transfer_out_quantity, "transfer_units"))

    if d.final_action == "REORDER":
        alerts.append(_alert(
            d, REORDER_RECOMMENDED,
            f"Reorder {d.reorder_quantity:,} units for {where}: remaining shortage "
            f"{d.remaining_shortage:,.2f} after transfers.",
            d.reorder_quantity, "reorder_units"))
    return alerts


def sort_alerts(alerts: Iterable[Alert]) -> list[Alert]:
    """Presentation order: display rank, then larger value, then SKU and warehouse."""
    return sorted(alerts, key=lambda a: (a.display_rank, -a.value, a.sku_id, a.warehouse_id, a.alert_type))


def generate_alerts(decisions: Iterable[FinalDecision]) -> list[Alert]:
    """Alerts for many decisions, in display order."""
    return sort_alerts(a for d in decisions for a in alerts_for_decision(d))
