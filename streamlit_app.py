"""
MedCare Pharma — Inventory & Demand Intelligence (Streamlit dashboard, owner: Gaurav).

Run from the medcare/ folder:
    streamlit run streamlit_app.py

Presentation only. All forecasts, risks, expiry analysis, transfers,
reorders, decisions and alerts come from the Phase 9 backend
(outputs/*.csv written by `python main.py forecast-all` and `decide-all`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="MedCare Pharma · Inventory & Demand Intelligence", page_icon="💊",
                   layout="wide", initial_sidebar_state="expanded")

import config  # noqa: E402
from frontend.components.styles import banner, inject_css  # noqa: E402
from frontend.utils import data_loader as dl  # noqa: E402
from frontend.utils.formatting import esc  # noqa: E402
from frontend.views import (dashboard, forecast, inventory, recommendations,  # noqa: E402
                            risk_alerts)
from frontend.views.common import FOCUS_SKU, set_focus  # noqa: E402

inject_css()


def _init_focus() -> None:
    if FOCUS_SKU in st.session_state:
        return
    skus = dl.list_skus()
    if not skus:
        return
    sku = dl.DEMO_SKU if dl.DEMO_SKU in skus else skus[0]
    whs = dl.list_warehouses(sku)
    set_focus(sku, dl.DEMO_WAREHOUSE if dl.DEMO_WAREHOUSE in whs else (whs[0] if whs else None))


def _header() -> None:
    meta = dl.get_forecast_metadata()
    as_of = pd.Timestamp(config.SIMULATION_DATE)
    chips = [f"📅 Data as of {as_of:%d %b %Y}",
             f"📈 {config.FORECAST_HORIZON_DAYS}-day demand forecast (Chronos-2 AI model)",
             f"🚚 Warehouse transfer time: {config.TRANSFER_TRANSIT_DAYS} days"]
    if meta.get("generated_at_utc"):
        try:
            run = pd.Timestamp(meta["generated_at_utc"]).to_pydatetime().astimezone()
            chips.append(f"🔄 Forecast updated {run:%d %b %Y, %H:%M}")
        except Exception:  # noqa: BLE001
            pass
    st.markdown(
        f"""<div class="mc-header"><div><div class="brand">💊 MEDCARE PHARMA</div>
        <div class="sub">Inventory &amp; Demand Intelligence · Smart Restock Alerts · Demand Sensing &amp; Replenishment</div></div>
        <div class="meta">{''.join(f'<span class="mc-chip">{esc(c)}</span>' for c in chips)}</div></div>""",
        unsafe_allow_html=True)
    real = dl.forecast_is_real_chronos()
    if real is False:
        banner(f"<b>⚠ STAND-IN FORECAST — not a real Chronos-2 result.</b> The saved forecast was produced by "
               f"<code>{esc(meta.get('pipeline'))}</code>. Run <code>python main.py forecast-all --refresh</code> "
               f"then <code>python main.py decide-all</code> before the demo.", "crit")
    if dl.forecast_cache_freshness() == "stale":
        banner("The saved forecast no longer matches the current data/config. "
               "Run <code>python main.py forecast-all</code> then <code>decide-all</code>.", "warn")


def _sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        with st.expander("⚙️ Backend status", expanded=dl.get_decisions() is None):
            for name, ts in dl.get_backend_status().items():
                st.markdown(f"{'✅' if ts else '❌'} **{name}**  \n<small>{ts or 'missing'}</small>",
                            unsafe_allow_html=True)
            meta = dl.get_forecast_metadata()
            if meta:
                st.caption(f"Forecast pipeline: {meta.get('pipeline', '—')} · "
                           f"series {meta.get('series_forecast', '—')} · errors {len(meta.get('errors', {}))}")
            if st.button("🔄 Reload outputs", width="stretch"):
                dl.clear_caches()
                st.rerun()
            if st.button("▶ Run decide-all", width="stretch",
                         help="Runs `python main.py decide-all` (Phase 9) using the saved Chronos-2 forecast."):
                with st.spinner("Running Phase 9 decide-all…"):
                    ok, out = dl.run_backend_command("decide-all")
                dl.clear_caches()
                (st.success if ok else st.error)("decide-all finished" if ok else "decide-all failed")
                st.code(out[-1500:] or "(no output)")
        st.caption(f"Data as of {pd.Timestamp(config.SIMULATION_DATE):%d %b %Y} · "
                   "Forecasts by Chronos-2 · Actions by the MedCare decision engine")


pages = [
    st.Page(dashboard.render, title="Executive Dashboard", icon="📊", url_path="dashboard", default=True),
    st.Page(inventory.render, title="Inventory Monitor", icon="📦", url_path="inventory"),
    st.Page(forecast.render, title="Demand Forecast", icon="📈", url_path="forecast"),
    st.Page(risk_alerts.render, title="Risk & Alerts", icon="🚨", url_path="risk-alerts"),
    st.Page(recommendations.render, title="Recommendations", icon="💡", url_path="recommendations"),
]

nav = st.navigation(pages, position="sidebar")
_init_focus()
_header()
_sidebar_status()
nav.run()
