"""
Build the MedCare HTML dashboard: ONE self-contained file with the real Phase 9 outputs embedded.

    python html_frontend/build_html.py          (run from the medcare/ folder, after forecast-all + decide-all)

Output: medcare_dashboard.html in the medcare/ folder. Double-click it: it opens in Safari/Chrome,
no server, no Python and no internet needed.

The page only DISPLAYS backend output. Nothing is forecast, classified or decided in the browser.
    outputs/decisions.csv, transfers.csv, alerts.csv     Phase 9 decision + alert engine
    outputs/forecast_output.csv, forecast_metadata.json  Chronos-2 30-day forecast
    outputs/model_evaluation.json (optional)             MAE/RMSE from ml/evaluate.py, if it was run
    data/batches.csv + decision_engine.expiry_engine     batch-level expiry analysis (backend function)
    data/derived/demand_simulation.csv                   demand history the model used
    data/sales.csv                                       sales history (display only)
Re-run this script whenever the backend outputs change.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from decision_engine.expiry_engine import analyse_batches  # noqa: E402
from decision_engine.forecast_utils import validated_forecast_values  # noqa: E402

OUT = config.OUTPUT_DIR
TARGET = ROOT / "medcare_dashboard.html"

DECISION_FIELDS = [
    "sku_id", "warehouse_id", "final_action", "transfer_recommended", "transfer_role", "transfer_in_quantity",
    "transfer_out_quantity", "expiry_transfer_out_quantity", "source_warehouses", "destination_warehouses",
    "reorder_quantity", "risk_status", "risk_reason", "e1_threshold_alert", "e1_reason", "expiry_risk",
    "expiry_reason", "potential_expiry_excess", "near_expiry_quantity", "current_stock", "expired_stock",
    "usable_stock_before", "usable_stock_after", "min_threshold", "safety_stock", "capacity", "lead_time_days",
    "average_daily_demand", "lead_time_demand", "required_stock", "projected_inventory", "days_of_stock",
    "shortage_before", "remaining_shortage", "reason", "seasonal_signal", "seasonal_reason", "seasonal_covariate_gap",
]


def die(msg: str) -> None:
    print(f"\nERROR: {msg}\n")
    sys.exit(1)


def clean(v):
    if v is None:
        return None
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 4)
    if hasattr(v, "item"):
        return clean(v.item())
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def as_list(v):
    if isinstance(v, list):
        return v
    try:
        x = ast.literal_eval(str(v))
        return x if isinstance(x, list) else []
    except (ValueError, SyntaxError):
        return []


def as_bool(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1")


def read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    need = {"decisions.csv": OUT / "decisions.csv", "transfers.csv": OUT / "transfers.csv",
            "alerts.csv": OUT / "alerts.csv", "forecast_output.csv": config.FORECAST_OUTPUT_CSV}
    missing = [n for n, p in need.items() if not p.exists()]
    if missing:
        die(f"Missing backend outputs: {', '.join(missing)}.\n"
            "Run first:  python main.py init-db && python main.py forecast-all && python main.py decide-all")

    # ---- Phase 9 outputs
    dec = pd.read_csv(OUT / "decisions.csv", dtype={"sku_id": str, "warehouse_id": str})
    dec = dec[[c for c in DECISION_FIELDS if c in dec.columns]].copy()
    for c in ("source_warehouses", "destination_warehouses"):
        if c in dec:
            dec[c] = dec[c].apply(as_list)
    for c in ("transfer_recommended", "e1_threshold_alert", "seasonal_covariate_gap"):
        if c in dec:
            dec[c] = dec[c].apply(as_bool)
    transfers = pd.read_csv(OUT / "transfers.csv", dtype=str) if (OUT / "transfers.csv").stat().st_size else pd.DataFrame()
    for c in ("transfer_quantity", "days_to_expiry", "destination_consumption_before_expiry", "source_usable_before",
              "source_usable_after", "destination_usable_before", "destination_usable_after"):
        if c in transfers:
            transfers[c] = pd.to_numeric(transfers[c], errors="coerce")
    alerts = pd.read_csv(OUT / "alerts.csv", dtype={"sku_id": str, "warehouse_id": str}) \
        if (OUT / "alerts.csv").stat().st_size else pd.DataFrame()

    # ---- Chronos-2 forecast
    meta = read_json(config.FORECAST_METADATA_JSON)
    fc = pd.read_csv(config.FORECAST_OUTPUT_CSV, parse_dates=["date"], dtype={"sku_id": str, "warehouse_id": str})
    fc = fc.sort_values(["sku_id", "warehouse_id", "date"])
    fc_start = fc["date"].min().strftime("%Y-%m-%d")
    fcd = {f"{s}|{w}": [round(float(x), 3) for x in g["forecast_demand"]] for (s, w), g in fc.groupby(["sku_id", "warehouse_id"])}

    # ---- Demand history the model used (up to the simulation date)
    dem = pd.read_csv(config.SIMULATION_DEMAND_CSV, parse_dates=["date"], dtype={"sku_id": str, "warehouse_id": str})
    dem = dem[dem["date"] <= config.simulation_timestamp()].sort_values(["sku_id", "warehouse_id", "date"])
    hist = {f"{s}|{w}": {"start": g["date"].min().strftime("%Y-%m-%d"), "v": g["demand_qty"].astype(int).tolist(),
                         "promo": g["promotion_flag"].astype(int).tolist(),
                         "season": g["season_flag"].astype(int).tolist()}
            for (s, w), g in dem.groupby(["sku_id", "warehouse_id"])}

    # ---- Batch-level expiry intelligence: the backend's own expiry engine on the saved forecast
    batches = pd.read_csv(config.BATCHES_CSV, parse_dates=["expiry_date"], dtype={"sku_id": str, "warehouse_id": str})
    fc_groups = {k: g for k, g in fc.groupby(["sku_id", "warehouse_id"])}
    brows, bsum, berr = [], [], {}
    for (s, w), g in batches.groupby(["sku_id", "warehouse_id"]):
        f = fc_groups.get((s, w))
        if f is None:
            berr[f"{s}/{w}"] = "no forecast"
            continue
        try:
            r = analyse_batches(g, s, w, validated_forecast_values(f, s, w))
        except Exception as exc:  # noqa: BLE001
            berr[f"{s}/{w}"] = str(exc)
            continue
        for d in r.batch_details:
            brows.append({"sku_id": s, "warehouse_id": w, "item_risk": r.expiry_risk, **{k: clean(v) for k, v in d.items()}})
        x = r.to_dict()
        x.pop("batch_details", None)
        bsum.append({k: clean(v) for k, v in x.items()})

    # ---- Sales history (display only)
    sales = {}
    sales_range = None
    if config.SOURCE_SALES_CSV.exists():
        sd = pd.read_csv(config.SOURCE_SALES_CSV, parse_dates=["date"], dtype={"sku_id": str, "warehouse_id": str})
        sd = sd.sort_values(["sku_id", "warehouse_id", "date"])
        sales_range = [sd["date"].min().strftime("%Y-%m-%d"), sd["date"].max().strftime("%Y-%m-%d")]
        for (s, w), g in sd.groupby(["sku_id", "warehouse_id"]):
            full = g.set_index("date").reindex(pd.date_range(g["date"].min(), g["date"].max(), freq="D"))
            sales[f"{s}|{w}"] = {"start": g["date"].min().strftime("%Y-%m-%d"),
                                 "u": [None if pd.isna(x) else int(x) for x in full["units_sold"]],
                                 "r": [None if pd.isna(x) else round(float(x), 2) for x in full["revenue"]]}

    evals = read_json(OUT / "model_evaluation.json").get("results", [])

    # ---- Email notification log (written by red_alert_mailer.py / main.py notify), if any
    notif = []
    if config.NOTIFICATION_LOG_CSV.exists() and config.NOTIFICATION_LOG_CSV.stat().st_size:
        nl = pd.read_csv(config.NOTIFICATION_LOG_CSV, dtype=str).fillna("")
        keep = [c for c in ("created_at_utc", "alert_date", "sku_id", "warehouse_id", "alert_type", "severity",
                            "recommended_action", "channel", "recipient", "status", "error") if c in nl.columns]
        notif = nl[keep].sort_values("created_at_utc", ascending=False).head(500).to_dict("records")

    payload = {
        "asOf": config.SIMULATION_DATE, "horizon": config.FORECAST_HORIZON_DAYS,
        "transit": config.TRANSFER_TRANSIT_DAYS, "fcStart": fc_start,
        "model": {"name": meta.get("model_name", config.CHRONOS_MODEL_NAME), "pipeline": meta.get("pipeline"),
                  "generated": meta.get("generated_at_utc"), "quantile": meta.get("point_forecast_quantile"),
                  "strategy": meta.get("future_covariate_strategy"), "covariates": config.COVARIATE_COLUMNS,
                  "errors": meta.get("errors", {})},
        "evals": evals,
        "decisions": records(dec), "transfers": records(transfers), "alerts": records(alerts),
        "forecast": fcd, "hist": hist, "batches": brows, "batchSummary": bsum, "batchErrors": berr,
        "sales": sales, "salesRange": sales_range, "notifications": notif,
        "builtAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    data_js = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    try:
        import plotly
        pj = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        plotly_tag = "<script>" + pj.read_text(encoding="utf-8").replace("</script", "<\\/script") + "</script>"
    except Exception:  # noqa: BLE001 - falls back to the CDN (needs internet)
        plotly_tag = '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>'

    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("<!--PLOTLY-->", plotly_tag).replace("/*__DATA__*/null", data_js)
    TARGET.write_text(html, encoding="utf-8")
    real = meta.get("pipeline", "Chronos2Pipeline") == "Chronos2Pipeline"
    print(f"Wrote {TARGET}  ({TARGET.stat().st_size / 1e6:.1f} MB)")
    print(f"  items {len(dec)} · transfers {len(transfers)} · alerts {len(alerts)} · batches {len(brows)} · "
          f"sales series {len(sales)} · forecast {'REAL Chronos-2' if real else 'STAND-IN (not real)'}")
    print("  Double-click medcare_dashboard.html to open it.")


if __name__ == "__main__":
    main()
