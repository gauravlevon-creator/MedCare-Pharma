"""Shared view helpers: the global focus item and missing-data messages."""

from __future__ import annotations

import streamlit as st

from frontend.components.styles import banner
from frontend.utils import data_loader as dl

FOCUS_SKU = "focus_sku"
FOCUS_WH = "focus_wh"


def require_decisions():
    """Return decisions.csv or show how to generate it and stop the page."""
    d = dl.get_decisions()
    if d is None or d.empty:
        banner("<b>No Phase 9 decisions found.</b> The dashboard reads <code>outputs/decisions.csv</code>, "
               "<code>transfers.csv</code> and <code>alerts.csv</code>. From the <code>medcare/</code> folder run:"
               "<br><code>python main.py init-db</code> → <code>python main.py forecast-all</code> → "
               "<code>python main.py decide-all</code><br>or use <b>Backend status → Run decide-all</b> "
               "in the sidebar.", "warn")
        st.stop()
    return d


def focus_item() -> tuple[str | None, str | None]:
    return st.session_state.get(FOCUS_SKU), st.session_state.get(FOCUS_WH)


def set_focus(sku: str, wh: str) -> None:
    st.session_state[FOCUS_SKU] = sku
    st.session_state[FOCUS_WH] = wh


def item_picker(key: str, label_prefix: str = "") -> tuple[str | None, str | None]:
    """SKU + warehouse selectors bound to the global focus item."""
    skus = dl.list_skus()
    if not skus:
        return None, None
    cur_sku, cur_wh = focus_item()
    c1, c2 = st.columns(2)
    sku = c1.selectbox(f"{label_prefix}SKU", skus,
                       index=skus.index(cur_sku) if cur_sku in skus else 0, key=f"{key}_sku")
    whs = dl.list_warehouses(sku)
    wh = c2.selectbox(f"{label_prefix}Warehouse", whs,
                      index=whs.index(cur_wh) if cur_wh in whs else 0, key=f"{key}_wh")
    if (sku, wh) != (cur_sku, cur_wh):
        set_focus(sku, wh)
    return sku, wh


def transfers_for(sku: str, wh: str, direction: str = "in") -> list[dict]:
    t = dl.get_transfer_data()
    if t is None or t.empty:
        return []
    col = "destination_warehouse" if direction == "in" else "source_warehouse"
    return t[(t["sku_id"] == sku) & (t[col] == wh)].to_dict("records")


def forecast_total_30(sku: str, wh: str) -> float | None:
    fc = dl.get_forecast_data(sku, wh)
    return None if fc.empty else float(fc["forecast_demand"].sum())
