"""
Phase 6: the single authoritative stock-risk engine.

Two separate, explainable signals, both based on USABLE stock
(expired stock is never counted):

P1 stock-out risk (predictive: forecast + lead time)
    ProjectedInventory = UsableStock - LeadTimeDemand
    1. ProjectedInventory <= 0            -> HIGH
       Forecast demand during the supplier lead time uses up all usable stock.
    2. ProjectedInventory <= SafetyStock  -> MEDIUM
       Usable stock survives the lead time but eats into the safety buffer.
    3. Otherwise                          -> LOW

E1 inventory alert (real-time threshold, independent of the forecast)
    UsableStock <= MinThreshold           -> e1_threshold_alert = True

Neither signal is an action. TRANSFER / REORDER / NO_ACTION is decided later.
"""

from __future__ import annotations

import config
from decision_engine.inventory_calculator import InventoryCalculation


def _fmt(value: float) -> str:
    return f"{value:,.{config.REPORT_DECIMALS}f}"


def classify_risk(projected_inventory: float, safety_stock: int) -> tuple[str, str]:
    """P1 stock-out risk: (risk_status, risk_reason) from projected usable inventory."""
    if projected_inventory <= 0:
        return (config.RISK_HIGH,
                f"Projected usable inventory ({_fmt(projected_inventory)}) falls to zero or below "
                f"during the lead time: forecast demand exceeds usable stock")
    if projected_inventory <= safety_stock:
        return (config.RISK_MEDIUM,
                f"Projected usable inventory ({_fmt(projected_inventory)}) is at or below "
                f"safety stock ({safety_stock}) by the end of the lead time")
    return (config.RISK_LOW,
            f"Projected usable inventory ({_fmt(projected_inventory)}) stays above "
            f"safety stock ({safety_stock}) through the lead time")


def e1_threshold_alert(usable_stock: int, min_threshold: int) -> tuple[bool, str]:
    """E1 real-time alert: usable stock at or below the minimum threshold."""
    if usable_stock <= min_threshold:
        return True, (f"Usable stock ({usable_stock}) is at or below the minimum threshold "
                      f"({min_threshold})")
    return False, f"Usable stock ({usable_stock}) is above the minimum threshold ({min_threshold})"


def assess_risk(calc: InventoryCalculation) -> dict:
    """P1 risk and E1 alert for one inventory calculation."""
    status, reason = classify_risk(calc.projected_inventory, calc.safety_stock)
    alert, alert_reason = e1_threshold_alert(calc.usable_stock, calc.min_threshold)
    return {
        "sku_id": calc.sku_id,
        "warehouse_id": calc.warehouse_id,
        "risk_status": status,
        "risk_reason": reason,
        "projected_inventory": calc.projected_inventory,
        "safety_stock": calc.safety_stock,
        "min_threshold": calc.min_threshold,
        "usable_stock": calc.usable_stock,
        "e1_threshold_alert": alert,
        "e1_reason": alert_reason,
    }
