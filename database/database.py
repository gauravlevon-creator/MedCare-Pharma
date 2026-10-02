"""
SQLite data-access layer for MedCare.

Ayush's CSVs in data/ are the master data. `init_database()` rebuilds
the SQLite file from them; every other module reads through the query
functions below (parameterised SQL, connections always closed).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

import config

logger = logging.getLogger(__name__)

DEMAND_COLUMNS = ["date", "sku_id", "warehouse_id", "demand_qty", "promotion_flag", "season_flag"]
INVENTORY_COLUMNS = ["sku_id", "warehouse_id", "current_stock", "min_threshold",
                     "safety_stock", "capacity", "lead_time_days"]
BATCH_COLUMNS = ["batch_id", "sku_id", "warehouse_id", "quantity", "expiry_date"]
SALES_COLUMNS = ["date", "sku_id", "warehouse_id", "units_sold", "unit_price", "revenue"]
TRANSACTION_COLUMNS = ["run_id", "sequence", "txn_date", "sku_id", "warehouse_id", "batch_id",
                       "txn_type", "txn_subtype", "quantity", "requested_quantity",
                       "unfulfilled_quantity", "on_hand_after", "usable_after", "reference", "reason"]
NOTIFICATION_COLUMNS = ["notification_id", "created_at_utc", "alert_date", "sku_id", "warehouse_id",
                        "alert_type", "severity", "recommended_action", "channel", "recipient",
                        "subject", "message", "status", "delivery_ref", "error"]


class DatabaseError(Exception):
    """Base error for the data-access layer."""


class DatabaseNotInitializedError(DatabaseError):
    """The SQLite file is missing or has no MedCare tables."""


class DataValidationError(DatabaseError):
    """Source CSV data failed validation."""


class RecordNotFoundError(DatabaseError):
    """A requested SKU / warehouse does not exist."""


# ------------------------------------------------------------------
# Connections
# ------------------------------------------------------------------

def _resolve(db_path: str | Path | None) -> Path:
    return Path(db_path) if db_path is not None else Path(config.DATABASE_PATH)


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection to an existing database and always close it."""
    path = _resolve(db_path)
    if not path.exists():
        raise DatabaseNotInitializedError(
            f"Database not found at {path}. Run: python main.py init-db"
        )
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def _query(sql: str, params: tuple = (), db_path: str | Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
            raise DatabaseNotInitializedError(
                f"Database at {_resolve(db_path)} is not initialised ({exc}). "
                "Run: python main.py init-db"
            ) from exc


# ------------------------------------------------------------------
# Source-data validation
# ------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing columns {missing}")


def _require_non_negative(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"{name}.{col} is not numeric")
        bad = int((df[col] < 0).sum())
        if bad:
            raise DataValidationError(f"{name}.{col} has {bad} negative values")


def _require_valid_dates(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    parsed = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
    bad = int(parsed.isna().sum())
    if bad:
        raise DataValidationError(f"{name}.{col} has {bad} invalid dates (expected YYYY-MM-DD)")
    return parsed


def validate_source_data(demand: pd.DataFrame, inventory: pd.DataFrame,
                         batches: pd.DataFrame) -> list[str]:
    """
    Validate the three master datasets.

    Raises DataValidationError for problems that would corrupt the database.
    Returns a list of non-fatal warnings (logged by the caller).
    """
    warnings: list[str] = []

    _require_columns(demand, DEMAND_COLUMNS, "demand")
    _require_columns(inventory, INVENTORY_COLUMNS, "inventory")
    _require_columns(batches, BATCH_COLUMNS, "batches")

    for name, df in (("demand", demand), ("inventory", inventory), ("batches", batches)):
        nulls = int(df.isnull().sum().sum())
        if nulls:
            raise DataValidationError(f"{name}: {nulls} missing values")

    _require_non_negative(demand, ["demand_qty"], "demand")
    _require_non_negative(inventory, INVENTORY_COLUMNS[2:], "inventory")
    _require_non_negative(batches, ["quantity"], "batches")
    _require_valid_dates(demand, "date", "demand")
    _require_valid_dates(batches, "expiry_date", "batches")

    for flag in config.COVARIATE_COLUMNS:
        bad = set(demand[flag].unique()) - {0, 1}
        if bad:
            raise DataValidationError(f"demand.{flag} has non-binary values {sorted(bad)}")

    if demand.duplicated(["date", "sku_id", "warehouse_id"]).any():
        raise DataValidationError("demand: duplicate (date, sku_id, warehouse_id) rows")
    if inventory.duplicated(["sku_id", "warehouse_id"]).any():
        raise DataValidationError("inventory: duplicate (sku_id, warehouse_id) rows")
    if batches["batch_id"].duplicated().any():
        raise DataValidationError("batches: duplicate batch_id values")

    inv_keys = set(map(tuple, inventory[["sku_id", "warehouse_id"]].values))
    dem_keys = set(map(tuple, demand[["sku_id", "warehouse_id"]].drop_duplicates().values))
    if inv_keys != dem_keys:
        warnings.append(
            f"inventory/demand key mismatch: {len(inv_keys - dem_keys)} only in inventory, "
            f"{len(dem_keys - inv_keys)} only in demand"
        )

    over_capacity = inventory[inventory["current_stock"] > inventory["capacity"]]
    if len(over_capacity):
        warnings.append(f"{len(over_capacity)} inventory rows have current_stock > capacity")

    batch_totals = batches.groupby(["sku_id", "warehouse_id"])["quantity"].sum()
    merged = inventory.set_index(["sku_id", "warehouse_id"])["current_stock"].to_frame().join(
        batch_totals.rename("batch_total"), how="left")
    mismatched = merged[merged["batch_total"].fillna(0) != merged["current_stock"]]
    if len(mismatched):
        warnings.append(f"{len(mismatched)} inventory rows do not equal the sum of their batches")

    return warnings


def validate_sales_data(sales: pd.DataFrame, inventory: pd.DataFrame) -> None:
    """Checks for the sales panel. Raises DataValidationError on problems."""
    _require_columns(sales, SALES_COLUMNS, "sales")
    if int(sales.isnull().sum().sum()):
        raise DataValidationError("sales: missing values")
    _require_non_negative(sales, ["units_sold", "unit_price", "revenue"], "sales")
    _require_valid_dates(sales, "date", "sales")
    if sales.duplicated(["date", "sku_id", "warehouse_id"]).any():
        raise DataValidationError("sales: duplicate (date, sku_id, warehouse_id) rows")
    keys = set(map(tuple, sales[["sku_id", "warehouse_id"]].drop_duplicates().values))
    inv_keys = set(map(tuple, inventory[["sku_id", "warehouse_id"]].values))
    if keys - inv_keys:
        raise DataValidationError(f"sales: {len(keys - inv_keys)} SKU/warehouse pairs not in inventory")


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Source file not found: {path}")
    return pd.read_csv(path, dtype={"sku_id": str, "warehouse_id": str, "batch_id": str})


def init_database(db_path: str | Path | None = None,
                  demand_csv: str | Path | None = None,
                  inventory_csv: str | Path | None = None,
                  batches_csv: str | Path | None = None,
                  schema_path: str | Path | None = None,
                  sales_csv: str | Path | None = None) -> dict[str, int]:
    """
    Rebuild the SQLite database from Ayush's CSVs.

    Demand is loaded from the date-shifted simulation copy
    (config.SIMULATION_DEMAND_CSV), which is regenerated from Ayush's
    data/demand.csv on every run. Inventory and batches are loaded unchanged.
    Sales history is loaded from the date-shifted sales copy
    (config.SIMULATION_SALES_CSV), regenerated from data/sales.csv. The E1
    tables (inventory_transactions, notifications) start empty.

    The new database is written to a temporary file and swapped in only
    after all rows load, so a failed load never leaves a half-built DB.
    Returns row counts per table.
    """
    path = _resolve(db_path)
    if demand_csv is None:
        # Regenerate the date-shifted simulation copy from Ayush's source file
        # so it always matches the current SIMULATION_DATE.
        from database.simulation_data import build_simulation_demand
        build_simulation_demand()
    if sales_csv is None:
        from database.simulation_data import build_simulation_sales
        build_simulation_sales()
    demand = _read_csv(Path(demand_csv or config.DEMAND_CSV))
    sales = _read_csv(Path(sales_csv or config.SIMULATION_SALES_CSV))
    inventory = _read_csv(Path(inventory_csv or config.INVENTORY_CSV))
    batches = _read_csv(Path(batches_csv or config.BATCHES_CSV))

    for warning in validate_source_data(demand, inventory, batches):
        logger.warning("Source data: %s", warning)
    validate_sales_data(sales, inventory)

    # Store dates as ISO text so SQLite comparisons work lexicographically.
    demand = demand[DEMAND_COLUMNS].copy()
    demand["date"] = pd.to_datetime(demand["date"]).dt.strftime("%Y-%m-%d")
    batches = batches[BATCH_COLUMNS].copy()
    batches["expiry_date"] = pd.to_datetime(batches["expiry_date"]).dt.strftime("%Y-%m-%d")
    inventory = inventory[INVENTORY_COLUMNS].copy()
    sales = sales[SALES_COLUMNS].copy()
    sales["date"] = pd.to_datetime(sales["date"]).dt.strftime("%Y-%m-%d")

    schema_sql = Path(schema_path or config.SCHEMA_PATH).read_text(encoding="utf-8")
    schema_sql += "\n" + Path(config.SCHEMA_EXTENSIONS_PATH).read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)
    try:
        conn.executescript(schema_sql)
        for table, df, cols in (("demand", demand, DEMAND_COLUMNS),
                                ("inventory", inventory, INVENTORY_COLUMNS),
                                ("batches", batches, BATCH_COLUMNS),
                                ("sales_history", sales, SALES_COLUMNS)):
            placeholders = ", ".join("?" for _ in cols)
            rows = [tuple(v.item() if hasattr(v, "item") else v for v in row)
                    for row in df[cols].itertuples(index=False, name=None)]
            conn.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", rows
            )
        conn.commit()
    except Exception:
        conn.close()
        tmp_path.unlink(missing_ok=True)
        raise
    conn.close()

    os.replace(tmp_path, path)
    counts = {"demand": len(demand), "inventory": len(inventory), "batches": len(batches),
              "sales_history": len(sales)}
    logger.info("Database initialised at %s: %s", path, counts)
    return counts


# ------------------------------------------------------------------
# Query functions
# ------------------------------------------------------------------

def get_demand_history(sku_id: str, warehouse_id: str,
                       end_date: str | None = None,
                       db_path: str | Path | None = None) -> pd.DataFrame:
    """
    Daily demand for one SKU + warehouse, sorted by date, up to and
    including end_date (defaults to config.SIMULATION_DATE).
    Returns an empty DataFrame if the pair has no history.
    """
    end = pd.Timestamp(end_date or config.SIMULATION_DATE).strftime("%Y-%m-%d")
    df = _query(
        "SELECT date, sku_id, warehouse_id, demand_qty, promotion_flag, season_flag "
        "FROM demand WHERE sku_id = ? AND warehouse_id = ? AND date <= ? ORDER BY date",
        (sku_id, warehouse_id, end), db_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_all_demand(end_date: str | None = None,
                   db_path: str | Path | None = None) -> pd.DataFrame:
    """All demand rows up to end_date (defaults to SIMULATION_DATE)."""
    end = pd.Timestamp(end_date or config.SIMULATION_DATE).strftime("%Y-%m-%d")
    df = _query(
        "SELECT date, sku_id, warehouse_id, demand_qty, promotion_flag, season_flag "
        "FROM demand WHERE date <= ? ORDER BY sku_id, warehouse_id, date",
        (end,), db_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_inventory_item(sku_id: str, warehouse_id: str,
                       db_path: str | Path | None = None) -> dict:
    """Inventory record for one SKU + warehouse. Raises RecordNotFoundError if absent."""
    df = _query(
        "SELECT sku_id, warehouse_id, current_stock, min_threshold, safety_stock, "
        "capacity, lead_time_days FROM inventory WHERE sku_id = ? AND warehouse_id = ?",
        (sku_id, warehouse_id), db_path)
    if df.empty:
        raise RecordNotFoundError(f"No inventory record for {sku_id} / {warehouse_id}")
    row = df.iloc[0]
    return {c: (int(row[c]) if c not in ("sku_id", "warehouse_id") else str(row[c]))
            for c in INVENTORY_COLUMNS}


def get_all_inventory(db_path: str | Path | None = None) -> pd.DataFrame:
    """All inventory rows sorted by SKU, warehouse."""
    return _query(
        "SELECT sku_id, warehouse_id, current_stock, min_threshold, safety_stock, "
        "capacity, lead_time_days FROM inventory ORDER BY sku_id, warehouse_id",
        (), db_path)


def get_inventory_for_sku(sku_id: str, db_path: str | Path | None = None) -> pd.DataFrame:
    """Inventory rows for one SKU across every warehouse (used for transfer search)."""
    return _query(
        "SELECT sku_id, warehouse_id, current_stock, min_threshold, safety_stock, "
        "capacity, lead_time_days FROM inventory WHERE sku_id = ? ORDER BY warehouse_id",
        (sku_id,), db_path)


def get_batches_for_item(sku_id: str, warehouse_id: str,
                         db_path: str | Path | None = None) -> pd.DataFrame:
    """Batches for one SKU + warehouse sorted by expiry. May be empty."""
    df = _query(
        "SELECT batch_id, sku_id, warehouse_id, quantity, expiry_date FROM batches "
        "WHERE sku_id = ? AND warehouse_id = ? ORDER BY expiry_date, batch_id",
        (sku_id, warehouse_id), db_path)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df


def get_all_batches(db_path: str | Path | None = None) -> pd.DataFrame:
    """All batches sorted by SKU, warehouse, expiry."""
    df = _query(
        "SELECT batch_id, sku_id, warehouse_id, quantity, expiry_date FROM batches "
        "ORDER BY sku_id, warehouse_id, expiry_date, batch_id",
        (), db_path)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df


def list_sku_warehouse_pairs(db_path: str | Path | None = None) -> list[tuple[str, str]]:
    """Every (sku_id, warehouse_id) in inventory, sorted."""
    df = _query("SELECT sku_id, warehouse_id FROM inventory ORDER BY sku_id, warehouse_id",
                (), db_path)
    return list(df.itertuples(index=False, name=None))


# ------------------------------------------------------------------
# E1 tables: sales history, inventory transactions, notifications
# ------------------------------------------------------------------

def get_sales_history(sku_id: str, warehouse_id: str, start_date: str | None = None,
                      end_date: str | None = None, db_path=None) -> pd.DataFrame:
    """Daily sales for one item between start_date and end_date (inclusive; end defaults to SIMULATION_DATE)."""
    end = pd.Timestamp(end_date or config.SIMULATION_DATE).strftime("%Y-%m-%d")
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d") if start_date else "0000-01-01"
    df = _query("SELECT date, sku_id, warehouse_id, units_sold, unit_price, revenue FROM sales_history "
                "WHERE sku_id = ? AND warehouse_id = ? AND date >= ? AND date <= ? ORDER BY date",
                (sku_id, warehouse_id, start, end), db_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_all_sales(end_date: str | None = None, db_path=None) -> pd.DataFrame:
    """All sales history up to end_date (defaults to SIMULATION_DATE)."""
    end = pd.Timestamp(end_date or config.SIMULATION_DATE).strftime("%Y-%m-%d")
    df = _query("SELECT date, sku_id, warehouse_id, units_sold, unit_price, revenue FROM sales_history "
                "WHERE date <= ? ORDER BY sku_id, warehouse_id, date", (end,), db_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _insert_rows(table: str, columns: list[str], rows: list[dict], db_path=None) -> int:
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    values = [tuple((r.get(c).item() if hasattr(r.get(c), "item") else r.get(c)) for c in columns)
              for r in rows]
    with get_connection(db_path) as conn:
        try:
            conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", values)
            conn.commit()
        except sqlite3.OperationalError as exc:
            raise DatabaseNotInitializedError(
                f"Table {table} missing ({exc}). Run: python main.py init-db") from exc
    return len(values)


def save_transactions(rows: list[dict], db_path=None) -> int:
    """Append ledger rows (dicts keyed by TRANSACTION_COLUMNS)."""
    return _insert_rows("inventory_transactions", TRANSACTION_COLUMNS, rows, db_path)


def get_transactions(run_id: str | None = None, db_path=None) -> pd.DataFrame:
    sql = f"SELECT txn_id, {', '.join(TRANSACTION_COLUMNS)} FROM inventory_transactions"
    params: tuple = ()
    if run_id:
        sql += " WHERE run_id = ?"
        params = (run_id,)
    return _query(sql + " ORDER BY run_id, sequence", params, db_path)


def save_notifications(rows: list[dict], db_path=None) -> int:
    """Append notification-log rows (dicts keyed by NOTIFICATION_COLUMNS)."""
    return _insert_rows("notifications", NOTIFICATION_COLUMNS, rows, db_path)


def get_notifications(db_path=None) -> pd.DataFrame:
    return _query(f"SELECT {', '.join(NOTIFICATION_COLUMNS)} FROM notifications "
                  "ORDER BY created_at_utc, notification_id", (), db_path)
