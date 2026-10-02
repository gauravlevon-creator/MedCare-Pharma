"""
Builds the date-shifted SIMULATION copies of the demand and sales data.

Sales: data/sales.csv (2025-09-23 .. 2026-09-22) is shifted +8 days to
data/derived/sales_simulation.csv (2025-10-01 .. 2026-09-30) with the same
constant-shift routine. The rest of this docstring describes demand.

Builds the date-shifted SIMULATION copy of Ayush's demand data.

Ayush's data/demand.csv (2025-09-23 .. 2026-09-22, from data/medcare_mysql.sql) is NOT modified.
This module writes data/derived/demand_simulation.csv, where every date
is moved by the same number of days so the last day of history lands on
config.SIMULATION_DATE. Every row keeps its SKU, warehouse, demand_qty,
promotion_flag and season_flag. Only the date labels move.

    shift_days = SIMULATION_DATE - max(source date)       (constant)

The source has 365 daily rows per series, so with SIMULATION_DATE =
2026-09-30 the simulation history is 2025-10-01 .. 2026-09-30.

Side effects of a constant shift (documented, not hidden):
* Weekday labels move by (shift_days mod 7).
* season_flag keeps its values but now sits on different calendar months.
Chronos-2 receives plain arrays with no timestamps, so its inputs, and
therefore its forecast values, are the same as before. Only the dates change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

VALUE_COLUMNS = ["sku_id", "warehouse_id", "demand_qty", "promotion_flag", "season_flag"]
SALES_VALUE_COLUMNS = ["sku_id", "warehouse_id", "units_sold", "unit_price", "revenue"]


class SimulationDataError(Exception):
    """A source daily panel cannot be shifted safely."""


def _content_hash(df: pd.DataFrame, value_columns: list[str] = VALUE_COLUMNS) -> str:
    ordered = df.sort_values(["sku_id", "warehouse_id", "date"])[value_columns]
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_shift_days(source: pd.DataFrame) -> int:
    source_end = pd.to_datetime(source["date"]).max()
    return int((config.simulation_timestamp() - source_end).days)


def shift_panel(source: pd.DataFrame, name: str = "data") -> tuple[pd.DataFrame, int]:
    """
    Shift a clean daily (date, sku_id, warehouse_id) panel so it ends on
    SIMULATION_DATE. Returns (shifted copy, shift_days); only dates change.
    """
    df = source.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any():
        raise SimulationDataError(f"Source {name} has invalid dates")
    if df.duplicated(["date", "sku_id", "warehouse_id"]).any():
        raise SimulationDataError(f"Source {name} has duplicate (date, sku, warehouse) rows")

    spans = df.groupby(["sku_id", "warehouse_id"])["date"].agg(["min", "max", "count"])
    if spans["min"].nunique() != 1 or spans["max"].nunique() != 1:
        raise SimulationDataError(f"{name}: series do not share the same start and end dates")
    expected = (spans["max"] - spans["min"]).dt.days + 1
    if (expected != spans["count"]).any():
        raise SimulationDataError(f"{name}: some series have missing days")

    shift = compute_shift_days(df)
    df["date"] = df["date"] + pd.Timedelta(days=shift)
    return df, shift


def shift_demand(source: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Backward-compatible name for the demand shift."""
    return shift_panel(source, "demand")


def _build_shifted_copy(src_path: Path, out_path: Path, meta_path: Path,
                        value_columns: list[str], name: str, description: str) -> dict:
    if not src_path.exists():
        raise SimulationDataError(f"Source {name} file not found: {src_path}")
    source_hash_before = file_sha256(src_path)
    source = pd.read_csv(src_path, dtype={"sku_id": str, "warehouse_id": str})
    missing = [c for c in ["date", *value_columns] if c not in source.columns]
    if missing:
        raise SimulationDataError(f"Source {name} missing columns {missing}")
    shifted, shift = shift_panel(source, name)

    source_dates = pd.to_datetime(source["date"])
    if len(shifted) != len(source):
        raise SimulationDataError("Row count changed during shift")
    if _content_hash(shifted, value_columns) != _content_hash(source.assign(date=source_dates), value_columns):
        raise SimulationDataError("Values changed during shift")
    if shifted["date"].max() != config.simulation_timestamp():
        raise SimulationDataError("Shifted data does not end on SIMULATION_DATE")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = shifted.sort_values(["sku_id", "warehouse_id", "date"])[["date", *value_columns]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)
    if file_sha256(src_path) != source_hash_before:
        raise SimulationDataError(f"Source {name} file changed while building the copy")

    meta = {
        "description": description,
        "source_file": src_path.name,
        "source_sha256": source_hash_before,
        "source_start": source_dates.min().strftime("%Y-%m-%d"),
        "source_end": source_dates.max().strftime("%Y-%m-%d"),
        "shift_days": shift,
        "simulation_date": config.SIMULATION_DATE,
        "simulation_start": shifted["date"].min().strftime("%Y-%m-%d"),
        "simulation_end": shifted["date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(out)),
        "series": int(out[["sku_id", "warehouse_id"]].drop_duplicates().shape[0]),
        "weekday_label_shift": shift % 7,
        "value_hash": _content_hash(shifted, value_columns),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Simulation %s written to %s (%s .. %s, shift %d days)",
                name, out_path, meta["simulation_start"], meta["simulation_end"], shift)
    return meta


def build_simulation_demand(source_csv: Path | None = None,
                            output_csv: Path | None = None,
                            metadata_json: Path | None = None) -> dict:
    """
    Write the shifted demand copy and its metadata. Deterministic: rerunning
    with the same source and SIMULATION_DATE produces the same file.
    """
    return _build_shifted_copy(
        Path(source_csv or config.SOURCE_DEMAND_CSV),
        Path(output_csv or config.SIMULATION_DEMAND_CSV),
        Path(metadata_json or config.SIMULATION_DEMAND_METADATA),
        VALUE_COLUMNS, "demand",
        "DATE-SHIFTED SIMULATION COPY of data/demand.csv. Only the date column differs; every "
        "other value is identical row for row. Generated by database/simulation_data.py. "
        "Do not edit by hand.")


def build_simulation_sales(source_csv: Path | None = None,
                           output_csv: Path | None = None,
                           metadata_json: Path | None = None) -> dict:
    """
    Write the shifted sales copy (source 2025-09-23..2026-09-22 -> +8 days ->
    2025-10-01..2026-09-30) and its metadata. Only dates change.
    """
    return _build_shifted_copy(
        Path(source_csv or config.SOURCE_SALES_CSV),
        Path(output_csv or config.SIMULATION_SALES_CSV),
        Path(metadata_json or config.SIMULATION_SALES_METADATA),
        SALES_VALUE_COLUMNS, "sales",
        "DATE-SHIFTED SIMULATION COPY of data/sales.csv. Only the date column differs; units_sold, "
        "unit_price and revenue are identical row for row. Used for E1 operational sales "
        "simulation, NOT for forecasting. Generated by database/simulation_data.py.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(build_simulation_demand(), indent=2))
    print(json.dumps(build_simulation_sales(), indent=2))
