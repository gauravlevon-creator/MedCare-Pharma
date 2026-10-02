"""Page 1 — Executive Dashboard."""

from __future__ import annotations

import streamlit as st

from frontend.components import charts
from frontend.components.kpi_cards import kpi_row
from frontend.components.recommendation_cards import decision_trace, top_recommendation_card
from frontend.components.sales_panel import sales_overview
from frontend.components.styles import page_title, section
from frontend.utils import data_loader as dl
from frontend.views.common import forecast_total_30, item_picker, require_decisions, transfers_for


def render() -> None:
    page_title("📊 Executive Dashboard",
               "Stock position, risks and recommended actions across every SKU and warehouse, "
               "based on a 30-day AI demand forecast.")
    decisions = require_decisions()
    k = dl.get_dashboard_kpis()

    kpi_row([
        ("SKU / warehouse items evaluated", k["evaluated"], "info", "every SKU in every warehouse"),
        ("REORDER", k["reorder"], "warning", "order from supplier"),
        ("TRANSFER", k["transfer"], "info", "move stock between warehouses"),
        ("NO ACTION", k["no_action"], "healthy", "stock is sufficient"),
    ])
    st.write("")
    kpi_row([
        ("🔴 HIGH stock-out risk", k["stockout_high"], "critical", "will run out within lead time"),
        ("🟠 MEDIUM stock-out risk", k["stockout_medium"], "warning", "will dip into safety stock"),
        ("🔴 HIGH expiry risk", k["expiry_high"], "critical", "stock may expire within 30 days"),
        ("🟡 E1 threshold alerts", k["e1_alerts"], "attention", "usable stock ≤ min threshold"),
    ])

    section("Inventory & risk distribution")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.risk_by_action(decisions), width="stretch", config=charts.CONFIG, theme=None)
    with c2:
        st.plotly_chart(charts.risk_by_warehouse(decisions), width="stretch", config=charts.CONFIG, theme=None)

    section("Decision trace — end-to-end demo")
    st.caption("Follow one item from today's stock to the final decision. Pick any SKU and warehouse.")
    sku, wh = item_picker("dash")
    d = dl.decision_row(sku, wh) if sku else None
    if d is None:
        st.info("No recommendation available for this SKU/warehouse.")
    else:
        decision_trace(d, forecast_total_30(sku, wh), transfers_for(sku, wh, "in"))

    section("Top recommendations")
    st.caption("The most urgent items, highest risk first.")
    alerts = dl.get_alerts()
    cards, seen = [], set()
    if alerts is not None and not alerts.empty:
        for a in alerts.to_dict("records"):
            key = (a["sku_id"], a["warehouse_id"])
            if key in seen:
                continue
            seen.add(key)
            row = dl.decision_row(*key)
            if row is None or row.get("final_action") == "NO_ACTION":
                continue
            cards.append(top_recommendation_card(row, a))
            if len(cards) == 4:
                break
    if not cards:
        st.info("No recommendation available.")
    else:
        cols = st.columns(len(cards))
        for c, html in zip(cols, cards):
            c.markdown(html, unsafe_allow_html=True)

    section("Sales performance")
    sales_overview("dash_sales")
