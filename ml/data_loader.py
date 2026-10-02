"""
Demand data loading for the forecasting module.

Nishit's original loader read data/demand.csv directly. The system now
reads from the SQLite database (built from Ayush's master CSVs), and only
history on or before config.SIMULATION_DATE is ever returned.
"""

from __future__ import annotations

import pandas as pd

import config
from database import database as db


def load_series(sku_id: str, warehouse_id: str, db_path=None) -> pd.DataFrame:
    """Demand history for one SKU + warehouse up to SIMULATION_DATE."""
    return db.get_demand_history(sku_id, warehouse_id,
                                 end_date=config.SIMULATION_DATE, db_path=db_path)


def load_demand_data(db_path=None) -> pd.DataFrame:
    """All demand history up to SIMULATION_DATE."""
    return db.get_all_demand(end_date=config.SIMULATION_DATE, db_path=db_path)


if __name__ == "__main__":
    df = load_demand_data()
    print("Shape:", df.shape)
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print(df.head())
