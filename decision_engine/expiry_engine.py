"""
Phase 7: expiry engine (owner: team lead).

STOCK DEFINITIONS (Option A; Ayush's data is never changed)
    recorded stock = sum of all batch quantities
    expired stock  = batches with expiry_date <= SIMULATION_DATE   (unavailable)
    usable stock   = batches with expiry_date >  SIMULATION_DATE
                   = recorded stock - expired stock
Expired stock stays visible for audit and the dashboard, but it is never
transferred and never counted in risk or replenishment calculations.

DATES (operational rule)
    days_to_expiry = expiry_date - SIMULATION_DATE
    SIMULATION_DATE means the END of that day; forecast day 1 is
    SIMULATION_DATE + 1. A batch expiring ON the simulation date (d = 0)
    cannot serve future demand, so it is unavailable. A batch is usable
    only when d > 0 (is_usable), and it can then cover forecast days 1..d.

EXPIRY STATUS (calendar label only; availability is decided by is_usable)
    d < 0 EXPIRED | 0-30 NEAR_EXPIRY | 31-60 EXPIRING_SOON | > 60 NORMAL
    A d = 0 batch keeps the NEAR_EXPIRY label but has usable = False and is
    counted in expired_quantity. Status quantity totals
    (near_expiry_quantity, ...) count usable batches only.

EXPIRY RISK (based on forecast consumption, not on days alone)
    For each usable batch, in earliest-expiry-first (FEFO) order:
      forecast_demand_through_expiry = forecast demand for days 1..d
      expected_consumption   = min(quantity, demand through expiry
                                   - demand already used by earlier-expiring batches)
      potential_expiry_excess = quantity - expected_consumption   (>= 0)
    FEFO order stops the same demand from being counted against two batches.
    If d is beyond the 30-day forecast, days after day 30 are extended at the
    mean daily forecast (flagged in consumption_basis).

    expiry_risk:
      HIGH   - excess > 0 in a NEAR_EXPIRY batch
      MEDIUM - excess > 0 in an EXPIRING_SOON batch
      LOW    - no excess within 60 days (any later excess is still reported)

TRANSFER ELIGIBILITY (used later by the allocation engine)
    is_usable(d) and d > TRANSFER_TRANSIT_DAYS
    (still usable after the simulated transit period).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import config
from database import database as db
from decision_engine.forecast_utils import ForecastInputError, validated_forecast_values

EXPIRY_RISK_HIGH = "HIGH"
EXPIRY_RISK_MEDIUM = "MEDIUM"
EXPIRY_RISK_LOW = "LOW"
EXPIRY_RISK_LEVELS = (EXPIRY_RISK_HIGH, EXPIRY_RISK_MEDIUM, EXPIRY_RISK_LOW)

BASIS_FORECAST = "forecast"
BASIS_EXTENDED = "forecast + extended at forecast mean"


class ExpiryCalculationError(Exception):
    """Batch or forecast data is invalid for expiry calculations."""


@dataclass(frozen=True)
class ExpiryResult:
    sku_id: str
    warehouse_id: str
    as_of_date: str
    total_recorded_quantity: int
    expired_quantity: int
    usable_quantity: int
    near_expiry_quantity: int
    expiring_soon_quantity: int
    normal_quantity: int
    potential_expiry_excess: float
    near_expiry_excess: float
    expiring_soon_excess: float
    transfer_eligible_quantity: int
    expiry_risk: str
    expiry_reason: str
    batch_count: int
    batch_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Pure functions
# ------------------------------------------------------------------

def days_to_expiry(expiry_date, as_of=None) -> int:
    """Calendar days from as_of (default SIMULATION_DATE) to expiry_date. Negative = expired."""
    as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else config.simulation_timestamp()
    exp = pd.Timestamp(expiry_date)
    if pd.isna(exp):
        raise ExpiryCalculationError(f"Invalid expiry date: {expiry_date!r}")
    return int((exp.normalize() - as_of_ts).days)


def is_usable(days: int) -> bool:
    """Operational rule: usable for future demand only if it expires AFTER the simulation date."""
    return days > 0


def classify_batch(days: int) -> str:
    """EXPIRED (<0), NEAR_EXPIRY (0-30), EXPIRING_SOON (31-60), NORMAL (>60)."""
    if days < 0:
        return config.BATCH_EXPIRED
    if days <= config.NEAR_EXPIRY_DAYS:
        return config.BATCH_NEAR_EXPIRY
    if days <= config.EXPIRING_SOON_DAYS:
        return config.BATCH_EXPIRING_SOON
    return config.BATCH_NORMAL


def cumulative_forecast_demand(forecast_values: np.ndarray, days: int) -> tuple[float, str]:
    """
    Forecast demand for days 1..days. Days beyond the forecast horizon are
    extended at the mean daily forecast. Returns (demand, consumption_basis).
    """
    if days <= 0:
        return 0.0, BASIS_FORECAST
    horizon = len(forecast_values)
    if days <= horizon:
        return float(np.sum(forecast_values[:days])), BASIS_FORECAST
    extra = (days - horizon) * float(np.mean(forecast_values))
    return float(np.sum(forecast_values)) + extra, BASIS_EXTENDED


def split_stock(batches: pd.DataFrame, as_of=None) -> dict:
    """Recorded / expired / usable quantities for a set of batches."""
    as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else config.simulation_timestamp()
    if batches is None or len(batches) == 0:
        return {"recorded": 0, "expired": 0, "usable": 0}
    exp = pd.to_datetime(batches["expiry_date"])
    recorded = int(batches["quantity"].sum())
    usable = int(batches.loc[exp > as_of_ts, "quantity"].sum())
    return {"recorded": recorded, "expired": recorded - usable, "usable": usable}


def _clean_batches(batches: pd.DataFrame, sku_id: str, warehouse_id: str) -> pd.DataFrame:
    rows = batches.copy() if batches is not None else pd.DataFrame()
    if len(rows) == 0:
        return rows
    missing = [c for c in ("batch_id", "quantity", "expiry_date") if c not in rows.columns]
    if missing:
        raise ExpiryCalculationError(f"Batches missing columns {missing}")
    qty = pd.to_numeric(rows["quantity"], errors="coerce")
    if qty.isna().any() or (qty < 0).any():
        raise ExpiryCalculationError(f"{sku_id}/{warehouse_id}: invalid batch quantities")
    rows["expiry_date"] = pd.to_datetime(rows["expiry_date"], errors="coerce")
    if rows["expiry_date"].isna().any():
        raise ExpiryCalculationError(f"{sku_id}/{warehouse_id}: invalid batch expiry dates")
    return rows.sort_values(["expiry_date", "batch_id"]).reset_index(drop=True)


def analyse_batches(batches: pd.DataFrame, sku_id: str, warehouse_id: str,
                    forecast_values: np.ndarray, as_of=None) -> ExpiryResult:
    """Batch-level expiry analysis for one item using its forecast (day 1 = as_of + 1)."""
    values = np.asarray(forecast_values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ExpiryCalculationError(f"{sku_id}/{warehouse_id}: forecast must be non-empty, finite, >= 0")
    as_of_ts = pd.Timestamp(as_of).normalize() if as_of is not None else config.simulation_timestamp()
    rows = _clean_batches(batches, sku_id, warehouse_id)

    details: list[dict] = []
    consumed_so_far = 0.0
    for r in rows.itertuples(index=False):
        d = days_to_expiry(r.expiry_date, as_of_ts)
        status = classify_batch(d)
        detail = {
            "batch_id": str(r.batch_id),
            "quantity": int(r.quantity),
            "expiry_date": r.expiry_date.strftime("%Y-%m-%d"),
            "days_to_expiry": d,
            "expiry_status": status,
            "usable": is_usable(d),
            "transfer_eligible": is_usable(d) and d > config.TRANSFER_TRANSIT_DAYS,
        }
        if not detail["usable"]:
            detail.update(forecast_demand_through_expiry=None, expected_consumption=0.0,
                          potential_expiry_excess=0.0, consumption_basis=None)
        else:
            demand_to_expiry, basis = cumulative_forecast_demand(values, d)
            available = max(0.0, demand_to_expiry - consumed_so_far)
            used = min(float(r.quantity), available)
            consumed_so_far += used
            detail.update(
                forecast_demand_through_expiry=round(demand_to_expiry, config.REPORT_DECIMALS),
                expected_consumption=round(used, config.REPORT_DECIMALS),
                potential_expiry_excess=round(float(r.quantity) - used, config.REPORT_DECIMALS),
                consumption_basis=basis,
            )
        details.append(detail)

    def qty(status: str) -> int:
        return int(sum(x["quantity"] for x in details
                       if x["usable"] and x["expiry_status"] == status))

    def excess(status: str | None = None) -> float:
        return round(float(sum(x["potential_expiry_excess"] for x in details
                               if status is None or x["expiry_status"] == status)),
                     config.REPORT_DECIMALS)

    split = split_stock(rows, as_of_ts)
    near_x, soon_x, total_x = excess(config.BATCH_NEAR_EXPIRY), excess(config.BATCH_EXPIRING_SOON), excess()

    if not details:
        risk, reason = EXPIRY_RISK_LOW, "No batch records for this item"
    elif near_x > 0:
        risk = EXPIRY_RISK_HIGH
        reason = (f"{near_x:,.2f} units in batches expiring within {config.NEAR_EXPIRY_DAYS} days "
                  f"exceed forecast demand before their expiry")
    elif soon_x > 0:
        risk = EXPIRY_RISK_MEDIUM
        reason = (f"{soon_x:,.2f} units in batches expiring in {config.NEAR_EXPIRY_DAYS + 1}-"
                  f"{config.EXPIRING_SOON_DAYS} days exceed forecast demand before their expiry")
    else:
        risk = EXPIRY_RISK_LOW
        reason = (f"Forecast demand is expected to consume all usable stock expiring within "
                  f"{config.EXPIRING_SOON_DAYS} days")
        if total_x > 0:
            reason += f"; {total_x:,.2f} units expiring later may exceed extended forecast demand"
    if split["expired"] > 0:
        reason += f"; {split['expired']:,} units already expired (excluded from usable stock)"

    return ExpiryResult(
        sku_id=sku_id,
        warehouse_id=warehouse_id,
        as_of_date=as_of_ts.strftime("%Y-%m-%d"),
        total_recorded_quantity=split["recorded"],
        expired_quantity=split["expired"],
        usable_quantity=split["usable"],
        near_expiry_quantity=qty(config.BATCH_NEAR_EXPIRY),
        expiring_soon_quantity=qty(config.BATCH_EXPIRING_SOON),
        normal_quantity=qty(config.BATCH_NORMAL),
        potential_expiry_excess=total_x,
        near_expiry_excess=near_x,
        expiring_soon_excess=soon_x,
        transfer_eligible_quantity=int(sum(x["quantity"] for x in details if x["transfer_eligible"])),
        expiry_risk=risk,
        expiry_reason=reason,
        batch_count=len(details),
        batch_details=details,
    )


# ------------------------------------------------------------------
# Database-backed entry points
# ------------------------------------------------------------------

def evaluate_expiry(sku_id: str, warehouse_id: str, forecast: pd.DataFrame,
                    db_path=None) -> ExpiryResult:
    """Expiry analysis for one SKU + warehouse from the database and its Chronos-2 forecast."""
    db.get_inventory_item(sku_id, warehouse_id, db_path=db_path)  # raises RecordNotFoundError
    try:
        values = validated_forecast_values(forecast, sku_id, warehouse_id)
    except ForecastInputError as exc:
        raise ExpiryCalculationError(str(exc)) from exc
    batches = db.get_batches_for_item(sku_id, warehouse_id, db_path=db_path)
    return analyse_batches(batches, sku_id, warehouse_id, values)


def usable_stock_for_item(sku_id: str, warehouse_id: str, db_path=None) -> dict:
    """Recorded / expired / usable quantities for one item (no forecast needed)."""
    return split_stock(db.get_batches_for_item(sku_id, warehouse_id, db_path=db_path))


def usable_stock_by_item(db_path=None) -> pd.DataFrame:
    """Recorded, usable and expired quantities for every inventory item (0 if no batches)."""
    inv = db.get_all_inventory(db_path=db_path)[["sku_id", "warehouse_id", "current_stock"]]
    b = db.get_all_batches(db_path=db_path)
    b["usable_quantity"] = b["quantity"].where(b["expiry_date"] > config.simulation_timestamp(), 0)
    b["expired_quantity"] = b["quantity"] - b["usable_quantity"]
    sums = b.groupby(["sku_id", "warehouse_id"], as_index=False)[
        ["quantity", "usable_quantity", "expired_quantity"]].sum()
    out = inv.merge(sums.rename(columns={"quantity": "total_recorded_quantity"}),
                    on=["sku_id", "warehouse_id"], how="left")
    for col in ("total_recorded_quantity", "usable_quantity", "expired_quantity"):
        out[col] = out[col].fillna(0).astype(int)
    return out
