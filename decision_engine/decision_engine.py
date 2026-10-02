"""
Unified decision engine (owner: team lead).

evaluate_inventory_state() = inventory calculations (usable stock)
                           + P1 stock-out risk + E1 threshold alert
                           + expiry summary                         (Phases 5-7)
evaluate_sku()             = states for every warehouse of one SKU
                           + cross-warehouse allocation (Phase 8)
                           + final action per warehouse + alerts    (Phase 9)
evaluate_all()             = evaluate_sku() for every SKU (180 items)
Each final decision also carries the seasonal demand signal (visibility only).
"""

from __future__ import annotations

import pandas as pd

import config
from decision_engine.alert_engine import Alert, generate_alerts, sort_alerts
from decision_engine.allocation_engine import Transfer, allocate, build_snapshot
from decision_engine.expiry_engine import evaluate_expiry
from decision_engine.final_decision import FinalDecision, build_final_decisions
from decision_engine.inventory_calculator import calculate_inventory
from decision_engine.risk_engine import assess_risk
from decision_engine.seasonal_signal import seasonal_signal_for_item

REPORT_FIELDS = [
    ("SKU", "sku_id"),
    ("Warehouse", "warehouse_id"),
    ("Recorded Stock", "current_stock"),
    ("Expired Stock", "expired_stock"),
    ("Usable Stock", "usable_stock"),
    ("Average Daily Demand", "average_daily_demand"),
    ("Lead Time (days)", "lead_time_days"),
    ("Lead-Time Demand", "lead_time_demand"),
    ("Safety Stock", "safety_stock"),
    ("Min Threshold", "min_threshold"),
    ("Required Stock", "required_stock"),
    ("Projected Inventory", "projected_inventory"),
    ("Days of Stock", "days_of_stock"),
    ("Excess Inventory", "excess_inventory"),
    ("P1 Risk Status", "risk_status"),
    ("P1 Risk Reason", "risk_reason"),
    ("E1 Threshold Alert", "e1_threshold_alert"),
    ("E1 Reason", "e1_reason"),
    ("Expiry Risk", "expiry_risk"),
    ("Potential Expiry Excess", "potential_expiry_excess"),
    ("Expiry Reason", "expiry_reason"),
]


def evaluate_inventory_state(sku_id: str, warehouse_id: str, forecast: pd.DataFrame,
                             db_path=None) -> dict:
    """
    Inventory calculations, P1 risk, E1 alert and expiry summary for one
    SKU + warehouse, as one flat dict. `forecast` is the Chronos-2 output
    for this item (integration.ml_service.get_forecast).
    """
    calc = calculate_inventory(sku_id, warehouse_id, forecast, db_path=db_path)
    risk = assess_risk(calc)
    expiry = evaluate_expiry(sku_id, warehouse_id, forecast, db_path=db_path)
    return {
        **calc.to_dict(),
        "risk_status": risk["risk_status"],
        "risk_reason": risk["risk_reason"],
        "e1_threshold_alert": risk["e1_threshold_alert"],
        "e1_reason": risk["e1_reason"],
        "near_expiry_quantity": expiry.near_expiry_quantity,
        "potential_expiry_excess": expiry.potential_expiry_excess,
        "transfer_eligible_quantity": expiry.transfer_eligible_quantity,
        "expiry_risk": expiry.expiry_risk,
        "expiry_reason": expiry.expiry_reason,
    }


def format_state_report(state: dict) -> str:
    """Readable checkpoint report for one evaluated item."""
    lines = []
    for label, key in REPORT_FIELDS:
        value = state[key]
        if key == "days_of_stock" and value is None:
            text = "n/a (no historical demand)"
        elif isinstance(value, float):
            text = f"{value:,.{config.REPORT_DECIMALS}f}"
        else:
            text = str(value)
        lines.append(f"{label + ':':<24} {text}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Phase 9: full decision for one SKU / all SKUs
# ------------------------------------------------------------------

def _state_with_capacity(sku_id, warehouse_id, forecast, db_path=None) -> dict:
    from database import database as db
    state = evaluate_inventory_state(sku_id, warehouse_id, forecast, db_path=db_path)
    state["capacity"] = db.get_inventory_item(sku_id, warehouse_id, db_path=db_path)["capacity"]
    return state


def evaluate_sku(sku_id: str, forecast_provider=None, db_path=None
                 ) -> tuple[dict[str, FinalDecision], list[Transfer], list[Alert]]:
    """
    Final decisions, transfers and alerts for every warehouse of one SKU.
    forecast_provider(sku_id, warehouse_id) -> forecast DataFrame
    (defaults to integration.ml_service.get_forecast).
    """
    from database import database as db
    if forecast_provider is None:
        from integration.ml_service import get_forecast as forecast_provider
    warehouses = db.get_inventory_for_sku(sku_id, db_path=db_path)["warehouse_id"].tolist()
    if not warehouses:
        raise ValueError(f"No inventory records for SKU {sku_id}")
    forecasts = {w: forecast_provider(sku_id, w) for w in warehouses}
    states = {w: _state_with_capacity(sku_id, w, forecasts[w], db_path) for w in warehouses}
    for w in warehouses:
        sig = seasonal_signal_for_item(sku_id, w, forecasts[w].sort_values("date")["forecast_demand"].to_numpy(),
                                       db_path=db_path)
        states[w].update(seasonal_signal=sig["seasonal_signal"], seasonal_reason=sig["seasonal_reason"],
                         seasonal_covariate_gap=sig["covariate_gap"])
    allocation = allocate([build_snapshot(sku_id, w, forecasts[w], db_path=db_path) for w in warehouses])
    decisions = build_final_decisions(states, allocation)
    return decisions, allocation.transfers, generate_alerts(decisions.values())


def forecast_provider_from_frame(forecasts: pd.DataFrame):
    """Serve per-item forecasts from one DataFrame (e.g. get_all_forecasts output)."""
    groups = {key: grp.sort_values("date").reset_index(drop=True)
              for key, grp in forecasts.groupby(["sku_id", "warehouse_id"])}

    def provider(sku_id: str, warehouse_id: str) -> pd.DataFrame:
        if (sku_id, warehouse_id) not in groups:
            raise ValueError(f"No forecast for {sku_id} / {warehouse_id}")
        return groups[(sku_id, warehouse_id)]
    return provider


def evaluate_all(forecast_provider=None, db_path=None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Decisions (one row per SKU + warehouse), transfers and alerts for every SKU.
    By default the saved 180-series Chronos-2 forecast is loaded once
    (integration.ml_service.get_all_forecasts).
    """
    from database import database as db
    if forecast_provider is None:
        from integration.ml_service import get_all_forecasts
        forecast_provider = forecast_provider_from_frame(get_all_forecasts()[0])
    skus = sorted({s for s, _ in db.list_sku_warehouse_pairs(db_path=db_path)})
    decisions, transfers, alerts = [], [], []
    for sku in skus:
        d, t, a = evaluate_sku(sku, forecast_provider, db_path=db_path)
        decisions += [x.to_dict() for x in d.values()]
        transfers += [x.to_dict() for x in t]
        alerts += a
    alerts_df = pd.DataFrame([a.to_dict() for a in sort_alerts(alerts)])
    return pd.DataFrame(decisions), pd.DataFrame(transfers), alerts_df

