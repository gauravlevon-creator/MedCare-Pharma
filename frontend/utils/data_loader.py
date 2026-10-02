"""
Thin data-access layer between the Phase 9 backend and the Streamlit UI.

Rules (Gaurav owns presentation only):
* Business values come from the backend outputs written by
  `python main.py forecast-all` and `python main.py decide-all`
  (outputs/forecast_output.csv, decisions.csv, transfers.csv, alerts.csv),
  or from backend functions (database.database, expiry_engine.analyse_batches,
  ml.evaluate.evaluate_series).
* Nothing here computes a forecast, a risk level, a transfer, a reorder
  quantity or an action. Only counting, filtering and parsing for display.
* Every cached reader is keyed on the file's modification time, so a new
  backend run is picked up automatically on the next rerun.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Make the Phase 9 backend importable (this file lives in medcare/frontend/utils)
# ------------------------------------------------------------------
MEDCARE_DIR = Path(__file__).resolve().parents[2]
if str(MEDCARE_DIR) not in sys.path:
    sys.path.insert(0, str(MEDCARE_DIR))

import config  # noqa: E402  (backend configuration: paths, dates, horizon, model)

DECISIONS_CSV = config.OUTPUT_DIR / "decisions.csv"
TRANSFERS_CSV = config.OUTPUT_DIR / "transfers.csv"
ALERTS_CSV = config.OUTPUT_DIR / "alerts.csv"
EVALUATION_JSON = config.OUTPUT_DIR / "model_evaluation.json"

REAL_PIPELINE = "Chronos2Pipeline"

# Default focus item for the demo walkthrough. Only a default *selection*;
# every value shown for it is read from the backend outputs.
DEMO_SKU = "M001"
DEMO_WAREHOUSE = "W002"

LIST_COLUMNS = ("source_warehouses", "destination_warehouses", "transfers")
BOOL_COLUMNS = ("transfer_recommended", "e1_threshold_alert", "seasonal_covariate_gap")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _parse_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


# ------------------------------------------------------------------
# Backend output files
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read_decisions(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sku_id": str, "warehouse_id": str})
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list)
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)
    return df


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sku_id": str, "warehouse_id": str, "source_warehouse": str,
                                    "destination_warehouse": str, "batch_id": str})


def get_decisions() -> pd.DataFrame | None:
    """Phase 9 final decision per SKU + warehouse (outputs/decisions.csv), or None."""
    m = _mtime(DECISIONS_CSV)
    return None if m is None else _read_decisions(str(DECISIONS_CSV), m)


# Spec names ---------------------------------------------------------
def get_inventory_data() -> pd.DataFrame | None:
    """Inventory view = backend decision rows (usable stock, required stock, risk, action...)."""
    return get_decisions()


def get_recommendations() -> pd.DataFrame | None:
    return get_decisions()


def get_risk_data() -> pd.DataFrame | None:
    return get_decisions()


def get_transfer_data() -> pd.DataFrame | None:
    """Phase 8/9 transfer records (outputs/transfers.csv), or None if the file is missing."""
    m = _mtime(TRANSFERS_CSV)
    return None if m is None else _read_csv(str(TRANSFERS_CSV), m)


def get_alerts() -> pd.DataFrame | None:
    """Phase 9 alert-engine output in backend display order (outputs/alerts.csv)."""
    m = _mtime(ALERTS_CSV)
    if m is None:
        return None
    df = _read_csv(str(ALERTS_CSV), m)
    return df


# ------------------------------------------------------------------
# Forecasts (Chronos-2 output saved by integration.ml_service)
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _read_forecasts(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], dtype={"sku_id": str, "warehouse_id": str})


def get_all_forecasts() -> pd.DataFrame | None:
    m = _mtime(config.FORECAST_OUTPUT_CSV)
    return None if m is None else _read_forecasts(str(config.FORECAST_OUTPUT_CSV), m)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_forecast_metadata() -> dict:
    m = _mtime(config.FORECAST_METADATA_JSON)
    return {} if m is None else _read_json(str(config.FORECAST_METADATA_JSON), m)


def get_forecast_data(sku_id: str, warehouse_id: str) -> pd.DataFrame:
    """
    The saved 30-day Chronos-2 forecast for one item (date, forecast_demand).
    Empty DataFrame when no forecast exists or the backend recorded an error.
    Never runs the model from the UI.
    """
    fc = get_all_forecasts()
    if fc is None:
        return pd.DataFrame(columns=["date", "sku_id", "warehouse_id", "forecast_demand"])
    rows = fc[(fc["sku_id"] == sku_id) & (fc["warehouse_id"] == warehouse_id)]
    return rows.sort_values("date").reset_index(drop=True)


def forecast_error_for(sku_id: str, warehouse_id: str) -> str | None:
    return (get_forecast_metadata().get("errors") or {}).get(f"{sku_id}/{warehouse_id}")


def forecast_is_real_chronos() -> bool | None:
    meta = get_forecast_metadata()
    if not meta:
        return None
    return meta.get("pipeline", REAL_PIPELINE) == REAL_PIPELINE


@st.cache_data(show_spinner=False)
def _cache_freshness(meta_mtime: float, db_mtime: float) -> str:
    """'fresh' / 'stale' / 'unknown' using the backend's own cache-validity check."""
    try:
        from integration import ml_service
        from ml import data_loader as ml_data
        expected = ml_service._expected_metadata(ml_data.load_demand_data())
        meta = json.loads(config.FORECAST_METADATA_JSON.read_text(encoding="utf-8"))
        return "stale" if any(meta.get(k) != v for k, v in expected.items()) else "fresh"
    except Exception:  # noqa: BLE001 - status only
        return "unknown"


def forecast_cache_freshness() -> str:
    mm, dm = _mtime(config.FORECAST_METADATA_JSON), _mtime(config.DATABASE_PATH)
    if mm is None or dm is None:
        return "unknown"
    return _cache_freshness(mm, dm)


# ------------------------------------------------------------------
# Database (demand history, batches) via database/database.py
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _demand_history(sku_id: str, warehouse_id: str, db_mtime: float) -> pd.DataFrame:
    from database import database as db
    return db.get_demand_history(sku_id, warehouse_id)


def get_demand_history(sku_id: str, warehouse_id: str) -> pd.DataFrame | None:
    """Historical daily demand up to the simulation date, or None if the DB is missing."""
    m = _mtime(config.DATABASE_PATH)
    if m is None:
        return None
    try:
        return _demand_history(sku_id, warehouse_id, m)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def _batch_expiry_all(db_mtime: float, fc_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Batch-level expiry intelligence for every item, produced by the backend's
    own expiry engine (decision_engine.expiry_engine.analyse_batches) on the
    saved Chronos-2 forecast. Returns (batch_rows, item_summary, errors).
    """
    from database import database as db
    from decision_engine.expiry_engine import analyse_batches
    from decision_engine.forecast_utils import validated_forecast_values

    batches = db.get_all_batches()
    forecasts = get_all_forecasts()
    rows, summaries, errors = [], [], {}
    fc_groups = {k: g for k, g in forecasts.groupby(["sku_id", "warehouse_id"])}
    for (sku, wh), grp in batches.groupby(["sku_id", "warehouse_id"]):
        fc = fc_groups.get((sku, wh))
        if fc is None:
            errors[f"{sku}/{wh}"] = "no saved forecast"
            continue
        try:
            values = validated_forecast_values(fc, sku, wh)
            res = analyse_batches(grp, sku, wh, values)
        except Exception as exc:  # noqa: BLE001
            errors[f"{sku}/{wh}"] = str(exc)
            continue
        for d in res.batch_details:
            rows.append({"sku_id": sku, "warehouse_id": wh, "item_expiry_risk": res.expiry_risk, **d})
        s = res.to_dict()
        s.pop("batch_details", None)
        summaries.append(s)
    return pd.DataFrame(rows), pd.DataFrame(summaries), errors


def get_batch_expiry() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    dm, fm = _mtime(config.DATABASE_PATH), _mtime(config.FORECAST_OUTPUT_CSV)
    if dm is None or fm is None:
        return None
    try:
        return _batch_expiry_all(dm, fm)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), pd.DataFrame(), {"_all": str(exc)}


# ------------------------------------------------------------------
# KPIs (counts of backend fields only)
# ------------------------------------------------------------------

def get_dashboard_kpis() -> dict | None:
    d = get_decisions()
    if d is None:
        return None
    alerts = get_alerts()
    action = d["final_action"].value_counts() if "final_action" in d else pd.Series(dtype=int)
    risk = d["risk_status"].value_counts() if "risk_status" in d else pd.Series(dtype=int)
    exp = d["expiry_risk"].value_counts() if "expiry_risk" in d else pd.Series(dtype=int)
    if alerts is not None and "alert_type" in alerts:
        e1 = int((alerts["alert_type"] == "E1_THRESHOLD").sum())
    elif "e1_threshold_alert" in d:
        e1 = int(d["e1_threshold_alert"].sum())
    else:
        e1 = None
    return {
        "evaluated": len(d),
        "reorder": int(action.get("REORDER", 0)),
        "transfer": int(action.get("TRANSFER", 0)),
        "no_action": int(action.get("NO_ACTION", 0)),
        "stockout_high": int(risk.get("HIGH", 0)),
        "stockout_medium": int(risk.get("MEDIUM", 0)),
        "expiry_high": int(exp.get("HIGH", 0)),
        "e1_alerts": e1,
        "alerts_total": None if alerts is None else len(alerts),
        "transfer_records": None if get_transfer_data() is None else len(get_transfer_data()),
    }


# ------------------------------------------------------------------
# Model evaluation (only real results from ml.evaluate)
# ------------------------------------------------------------------

def get_saved_evaluations() -> list[dict]:
    m = _mtime(EVALUATION_JSON)
    if m is None:
        return []
    data = _read_json(str(EVALUATION_JSON), m)
    return data.get("results", []) if isinstance(data, dict) else []


def run_model_evaluation(sku_id: str, warehouse_id: str, horizon: int) -> dict:
    """
    Run the backend holdout evaluation (ml.evaluate.evaluate_series) with the
    real Chronos-2 model and save the result so MAE/RMSE can be shown.
    Raises if the model is unavailable.
    """
    from ml.evaluate import evaluate_series
    result = evaluate_series(sku_id, warehouse_id, horizon)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["model_name"] = config.CHRONOS_MODEL_NAME
    existing = [r for r in get_saved_evaluations()
                if not (r.get("sku_id") == sku_id and r.get("warehouse_id") == warehouse_id
                        and r.get("horizon_days") == horizon)]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_JSON.write_text(json.dumps({"results": existing + [result]}, indent=2), encoding="utf-8")
    return result


# ------------------------------------------------------------------
# Backend status + running the backend CLI
# ------------------------------------------------------------------

def _fmt_mtime(m: float | None) -> str | None:
    return None if m is None else datetime.fromtimestamp(m).strftime("%Y-%m-%d %H:%M")


def get_backend_status() -> dict:
    files = {
        "Database (medcare.db)": config.DATABASE_PATH,
        "forecast_output.csv": config.FORECAST_OUTPUT_CSV,
        "decisions.csv": DECISIONS_CSV,
        "transfers.csv": TRANSFERS_CSV,
        "alerts.csv": ALERTS_CSV,
    }
    return {name: _fmt_mtime(_mtime(p)) for name, p in files.items()}


def run_backend_command(*args: str, timeout: int = 1800) -> tuple[bool, str]:
    """Run `python main.py <args>` in the medcare folder (the backend's own CLI)."""
    try:
        proc = subprocess.run([sys.executable, "main.py", *args], cwd=MEDCARE_DIR,
                              capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, out.strip()


def clear_caches() -> None:
    st.cache_data.clear()


# ------------------------------------------------------------------
# Small lookups for the UI
# ------------------------------------------------------------------

def decision_row(sku_id: str, warehouse_id: str) -> dict | None:
    d = get_decisions()
    if d is None:
        return None
    r = d[(d["sku_id"] == sku_id) & (d["warehouse_id"] == warehouse_id)]
    return None if r.empty else r.iloc[0].to_dict()


def list_skus() -> list[str]:
    d = get_decisions()
    if d is not None:
        return sorted(d["sku_id"].unique().tolist())
    fc = get_all_forecasts()
    return [] if fc is None else sorted(fc["sku_id"].unique().tolist())


def list_warehouses(sku_id: str | None = None) -> list[str]:
    d = get_decisions()
    if d is None:
        fc = get_all_forecasts()
        if fc is None:
            return []
        d = fc
    if sku_id:
        d = d[d["sku_id"] == sku_id]
    return sorted(d["warehouse_id"].unique().tolist())


# ------------------------------------------------------------------
# Sales history (data/sales.csv — source file, never modified)
# Descriptive totals/averages only; sales are not used for decisions.
# ------------------------------------------------------------------

SALES_CSV = config.SOURCE_SALES_CSV
SALES_COLUMNS = ["date", "sku_id", "warehouse_id", "units_sold", "unit_price", "revenue"]


@st.cache_data(show_spinner=False)
def _read_sales(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], dtype={"sku_id": str, "warehouse_id": str})
    missing = [c for c in SALES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"sales.csv is missing columns {missing}")
    return df


def get_sales() -> pd.DataFrame | None:
    m = _mtime(SALES_CSV)
    if m is None:
        return None
    try:
        return _read_sales(str(SALES_CSV), m)
    except Exception:  # noqa: BLE001
        return None


def sales_summary(sales: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Totals and per-day averages by period. freq: 'M' (month), 'Y' (year) or 'ALL'.
    Days = calendar days with sales records in that period (partial months/years are flagged).
    """
    if sales is None or sales.empty:
        return pd.DataFrame()
    s = sales.assign(period="Full period" if freq == "ALL" else sales["date"].dt.to_period(freq).astype(str))
    g = s.groupby("period", sort=True).agg(
        start=("date", "min"), end=("date", "max"), days=("date", "nunique"),
        units=("units_sold", "sum"), revenue=("revenue", "sum"))
    g["avg_units_day"] = g["units"] / g["days"]
    g["avg_revenue_day"] = g["revenue"] / g["days"]
    if freq == "M":
        full = g["start"].dt.days_in_month
        g["partial"] = g["days"] < full
    elif freq == "Y":
        g["partial"] = g["days"] < g["start"].dt.is_leap_year.map({True: 366, False: 365})
    else:
        g["partial"] = False
    return g.reset_index()
