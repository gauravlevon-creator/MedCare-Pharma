"""Page 3 — Demand Forecast (Chronos-2, 30-day horizon)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from frontend.components import charts
from frontend.components.kpi_cards import kpi_row
from frontend.components.styles import banner, page_title, section
from frontend.utils import data_loader as dl
from frontend.utils.formatting import esc, num, risk_badge, smart_num
from frontend.components.sales_panel import sales_explorer
from frontend.views.common import focus_item, item_picker

HISTORY_WINDOWS = {"Last 60 days": 60, "Last 120 days": 120, "Last 180 days": 180, "Full year": None}


def render() -> None:
    page_title("📈 Demand Forecast",
               f"Past demand, the {config.FORECAST_HORIZON_DAYS}-day Chronos-2 AI forecast, and sales history.")
    tab_fc, tab_sales = st.tabs(["📈 Demand forecast", "🛒 Sales history"])
    with tab_fc:
        _forecast_tab()
    with tab_sales:
        sku, wh = focus_item()
        sales_explorer("fc_sales", sku, wh)


def _forecast_tab() -> None:
    if dl.get_all_forecasts() is None:
        banner("<b>No forecast available.</b> Run <code>python main.py forecast-all</code> in the "
               "<code>medcare/</code> folder to produce <code>outputs/forecast_output.csv</code>.", "warn")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        sku, wh = item_picker("fc")
    with c2:
        window = st.selectbox("History shown", list(HISTORY_WINDOWS), index=1)
    if not sku:
        st.info("No SKU/warehouse items found.")
        return

    fc = dl.get_forecast_data(sku, wh)
    err = dl.forecast_error_for(sku, wh)
    history = dl.get_demand_history(sku, wh)
    if history is None:
        st.warning("Historical demand unavailable: database not found. Run `python main.py init-db`.")
    elif HISTORY_WINDOWS[window]:
        history = history.tail(HISTORY_WINDOWS[window])

    if fc.empty:
        st.warning(f"No forecast available for this SKU/warehouse.{' Backend error: ' + err if err else ''}")
    else:
        values = fc["forecast_demand"].astype(float).to_numpy()
        # Spec section 9: summaries are derived from the single 30-day forecast.
        kpi_row([
            ("Next day forecast", round(values[0], 2) if len(values) >= 1 else None, "info", "forecast day 1"),
            ("Next 7 days", round(values[:7].sum(), 2) if len(values) >= 7 else None, "info",
             "derived from 30-day Chronos-2 forecast"),
            (f"Next {len(values)} days", round(values.sum(), 2), "info", "sum of all forecast days"),
            ("Average daily demand", round(values.mean(), 2), "info", f"mean of {len(values)} forecast values"),
        ])
        st.caption("“Next 7 Days” is the sum of forecast days 1–7 of the single 30-day Chronos-2 forecast — "
                   "there is no separate 7-day model.")

    st.plotly_chart(charts.demand_forecast(history, fc, pd.Timestamp(config.SIMULATION_DATE),
                                           f"{sku} / {wh} — historical demand vs Chronos-2 forecast"),
                    width="stretch", config=charts.CONFIG, theme=None)

    d = dl.decision_row(sku, wh)
    left, right = st.columns([1.3, 1])
    with left:
        section("How the forecast feeds the decision")
        if d is None:
            st.info("No recommendation available for this SKU/warehouse.")
        else:
            st.markdown(
                f"""<div class="mc-card" style="border-left-color:#2a78d6;">
                <div class="grid">
                  <div>Lead time<b>{smart_num(d.get('lead_time_days'))} days</b></div>
                  <div>Lead-time demand<b>{num(d.get('lead_time_demand'), 2)}</b></div>
                  <div>Safety stock<b>{smart_num(d.get('safety_stock'))}</b></div>
                  <div>Required stock<b>{num(d.get('required_stock'), 2)}</b></div>
                  <div>Usable stock<b>{smart_num(d.get('usable_stock_before'))}</b></div>
                  <div>Projected inventory<b>{num(d.get('projected_inventory'), 2)}</b></div>
                  <div>Historical avg daily demand<b>{num(d.get('average_daily_demand'), 2)}</b></div>
                  <div>Days of stock<b>{num(d.get('days_of_stock'), 2)}</b></div>
                </div>
                <div class="reason">P1 stock-out risk: {risk_badge(d.get('risk_status'))} — {esc(d.get('risk_reason'))}</div>
                </div>""", unsafe_allow_html=True)
            sig = d.get("seasonal_signal")
            if sig and sig != config.SEASON_SIGNAL_NORMAL:
                gap = " · covariate gap" if d.get("seasonal_covariate_gap") else ""
                banner(f"<b>Seasonal demand signal: {esc(sig)}{gap}</b><br>{esc(d.get('seasonal_reason'))}", "info")
    with right:
        section("Model")
        meta = dl.get_forecast_metadata()
        pipeline = meta.get("pipeline", "—")
        st.markdown(
            f"""<div class="mc-card"><div class="grid">
            <div>Model<b>Chronos-2</b></div>
            <div>Checkpoint<b>{esc(meta.get('model_name', config.CHRONOS_MODEL_NAME))}</b></div>
            <div>Horizon<b>{esc(meta.get('forecast_horizon_days', config.FORECAST_HORIZON_DAYS))} days</b></div>
            <div>Point forecast<b>q{esc(meta.get('point_forecast_quantile', config.POINT_FORECAST_QUANTILE))} (median)</b></div>
            <div>Covariates<b>{esc(', '.join(config.COVARIATE_COLUMNS))}</b></div>
            <div>Future covariates<b>{esc(meta.get('future_covariate_strategy', '—'))}</b></div>
            <div>Pipeline<b>{esc(pipeline)}</b></div>
            <div>Generated (UTC)<b>{esc(str(meta.get('generated_at_utc', '—'))[:16].replace('T', ' '))}</b></div>
            </div></div>""", unsafe_allow_html=True)
        _metrics(sku, wh)

    if not fc.empty:
        with st.expander("Forecast values (30 days)"):
            t = fc[["date", "forecast_demand"]].copy()
            t.insert(0, "Day", range(1, len(t) + 1))
            t["date"] = t["date"].dt.strftime("%Y-%m-%d")
            t = t.rename(columns={"date": "Date", "forecast_demand": "Forecast demand"})
            st.dataframe(t, hide_index=True, width="stretch",
                         column_config={"Forecast demand": st.column_config.NumberColumn(format="%.2f")})


def _metrics(sku: str, wh: str) -> None:
    """MAE/RMSE only from a real backend evaluation (ml.evaluate) — never invented."""
    results = [r for r in dl.get_saved_evaluations()
               if r.get("sku_id") == sku and r.get("warehouse_id") == wh]
    st.markdown("**Holdout accuracy (MAE / RMSE)**")
    if results:
        rows = [{"Horizon": f"{r['horizon_days']} d", "Window": r.get("evaluation_period"),
                 "Chronos-2 MAE": r.get("chronos_mae"), "Chronos-2 RMSE": r.get("chronos_rmse"),
                 "7-day MA MAE": r.get("baseline_mae"), "7-day MA RMSE": r.get("baseline_rmse")}
                for r in results]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Single series, single holdout window (ml/evaluate.py). Not overall model accuracy.")
    else:
        st.caption("Not yet measured for this SKU / warehouse. Click below to test the model on past data.")
    if st.button("Run holdout evaluation (30 days)", key=f"eval_{sku}_{wh}"):
        with st.spinner("Running Chronos-2 holdout evaluation…"):
            try:
                dl.run_model_evaluation(sku, wh, config.FORECAST_HORIZON_DAYS)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Evaluation not available: {exc}")
