"""
Phase 5: inventory calculations (owner: team lead).

Deterministic formulas using Ayush's inventory, batches, historical demand
and the Chronos-2 forecast. All operational figures use USABLE stock:

    RecordedStock      = inventory.current_stock (= sum of all batches; audit only)
    ExpiredStock       = batches with expiry_date <= SIMULATION_DATE (audit only)
    UsableStock        = batches with expiry_date >  SIMULATION_DATE

    AverageDailyDemand = total historical demand / number of historical days
    LeadTimeDemand     = sum(forecast_demand for the first lead_time_days days)
    RequiredStock      = LeadTimeDemand + SafetyStock
    ProjectedInventory = UsableStock - LeadTimeDemand
    DaysOfStock        = UsableStock / AverageDailyDemand     (None if demand is 0)
    ExcessInventory    = UsableStock - RequiredStock          (+ excess / - shortage)

No transfer, reorder or expiry logic lives here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine.expiry_engine import usable_stock_for_item
from decision_engine.forecast_utils import ForecastInputError, validated_forecast_values


class InventoryCalculationError(Exception):
    """Inputs are missing or inconsistent, so no calculation is produced."""


@dataclass(frozen=True)
class InventoryCalculation:
    sku_id: str
    warehouse_id: str
    current_stock: int               # recorded stock from inventory.csv (includes expired)
    expired_stock: int
    usable_stock: int                # basis for every operational calculation
    batch_total_matches_inventory: bool
    min_threshold: int
    safety_stock: int
    capacity: int
    lead_time_days: int
    history_days: int
    average_daily_demand: float
    lead_time_demand: float
    required_stock: float
    projected_inventory: float
    days_of_stock: float | None      # None = no historical demand (coverage not finite)
    excess_inventory: float

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Pure formulas (unit-tested directly)
# ------------------------------------------------------------------

def average_daily_demand(demand_qty: pd.Series | np.ndarray) -> float:
    """Mean demand per historical day. Returns 0.0 for an all-zero series; raises if empty."""
    values = np.asarray(demand_qty, dtype=float)
    if values.size == 0:
        raise InventoryCalculationError("No historical demand days available")
    if np.isnan(values).any():
        raise InventoryCalculationError("Historical demand contains missing values")
    return float(values.sum() / values.size)


def lead_time_demand(forecast_values: pd.Series | np.ndarray, lead_time_days: int) -> float:
    """
    Sum of the first `lead_time_days` forecast values.
    lead_time_days = 0 gives 0. A lead time longer than the forecast raises,
    rather than inventing demand for days that were not forecast.
    """
    if lead_time_days < 0:
        raise InventoryCalculationError(f"lead_time_days cannot be negative ({lead_time_days})")
    values = np.asarray(forecast_values, dtype=float)
    if lead_time_days > values.size:
        raise InventoryCalculationError(
            f"lead_time_days ({lead_time_days}) exceeds the forecast horizon ({values.size} days)"
        )
    return float(values[:lead_time_days].sum())


def required_stock(lead_time_demand_value: float, safety_stock: int) -> float:
    return float(lead_time_demand_value + safety_stock)


def projected_inventory(current_stock: int, lead_time_demand_value: float) -> float:
    return float(current_stock - lead_time_demand_value)


def days_of_stock(current_stock: int, avg_daily_demand: float) -> float | None:
    """CurrentStock / AverageDailyDemand, or None when average demand is zero."""
    if avg_daily_demand <= 0:
        return None
    return float(current_stock / avg_daily_demand)


def excess_inventory(current_stock: int, required_stock_value: float) -> float:
    return float(current_stock - required_stock_value)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def calculate_inventory(sku_id: str, warehouse_id: str, forecast: pd.DataFrame,
                        db_path=None) -> InventoryCalculation:
    """
    Inventory calculations for one SKU + warehouse.

    Inventory and batches come from the database (Ayush's data), demand
    history ends on SIMULATION_DATE, and lead-time demand comes from the
    Chronos-2 forecast. Usable (unexpired) stock drives every formula.
    Raises RecordNotFoundError or InventoryCalculationError instead of
    producing substitute values.
    """
    item = db.get_inventory_item(sku_id, warehouse_id, db_path=db_path)
    history = db.get_demand_history(sku_id, warehouse_id, end_date=config.SIMULATION_DATE,
                                    db_path=db_path)
    if history.empty:
        raise InventoryCalculationError(f"No demand history for {sku_id} / {warehouse_id}")

    try:
        forecast_values = validated_forecast_values(forecast, sku_id, warehouse_id)
    except ForecastInputError as exc:
        raise InventoryCalculationError(str(exc)) from exc

    stock = usable_stock_for_item(sku_id, warehouse_id, db_path=db_path)
    usable = stock["usable"]
    avg = average_daily_demand(history["demand_qty"])
    ltd = lead_time_demand(forecast_values, item["lead_time_days"])
    req = required_stock(ltd, item["safety_stock"])

    return InventoryCalculation(
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        current_stock=item["current_stock"],
        expired_stock=stock["expired"],
        usable_stock=usable,
        batch_total_matches_inventory=stock["recorded"] == item["current_stock"],
        min_threshold=item["min_threshold"],
        safety_stock=item["safety_stock"],
        capacity=item["capacity"],
        lead_time_days=item["lead_time_days"],
        history_days=int(len(history)),
        average_daily_demand=avg,
        lead_time_demand=ltd,
        required_stock=req,
        projected_inventory=projected_inventory(usable, ltd),
        days_of_stock=days_of_stock(usable, avg),
        excess_inventory=excess_inventory(usable, req),
    )
