"""Sales history panel: monthly, yearly and average-per-day sales from data/sales.csv."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import charts
from frontend.components.kpi_cards import kpi_row
from frontend.utils import data_loader as dl
from frontend.utils.formatting import GRID, INK, SERIES_BLUE, SERIES_BLUE_LIGHT

METRICS = {
    "Units sold": ("units", "avg_units_day", "units"),
    "Revenue": ("revenue", "avg_revenue_day", "revenue"),
}


def _period_label(p: str, freq: str) -> str:
    if freq == "M":
        return pd.Period(p, "M").strftime("%b %Y")
    return p


def _chart(summary: pd.DataFrame, freq: str, metric: str, basis: str, title: str) -> go.Figure:
    total_col, avg_col, unit = METRICS[metric]
    col = total_col if basis == "Total" else avg_col
    labels = [_period_label(p, freq) for p in summary["period"]]
    colors = [SERIES_BLUE_LIGHT if p else SERIES_BLUE for p in summary["partial"]]
    fig = go.Figure(go.Bar(
        x=labels, y=summary[col], marker_color=colors,
        customdata=summary[["days", "units", "revenue", "avg_units_day", "avg_revenue_day"]].to_numpy(),
        hovertemplate="<b>%{x}</b> · %{customdata[0]} days<br>Units sold: %{customdata[1]:,.0f}"
                      "<br>Revenue: %{customdata[2]:,.2f}<br>Avg units/day: %{customdata[3]:,.1f}"
                      "<br>Avg revenue/day: %{customdata[4]:,.2f}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text=f"{metric}{' per day' if basis != 'Total' else ''}", gridcolor=GRID)
    fig = charts._layout(fig, height=340, legend=False)
    fig.update_layout(margin=dict(t=48))
    return fig


def _table(summary: pd.DataFrame, freq: str) -> pd.DataFrame:
    t = summary.copy()
    t["Period"] = [(_period_label(p, freq) + (" (partial)" if part else ""))
                   for p, part in zip(t["period"], t["partial"])]
    t["From"] = t["start"].dt.strftime("%Y-%m-%d")
    t["To"] = t["end"].dt.strftime("%Y-%m-%d")
    t["revenue"] = t["revenue"].round(0).astype("int64")
    t["avg_revenue_day"] = t["avg_revenue_day"].round(0).astype("int64")
    t["avg_units_day"] = t["avg_units_day"].round(1)
    return t[["Period", "From", "To", "days", "units", "revenue", "avg_units_day", "avg_revenue_day"]].rename(columns={
        "days": "Days", "units": "Units Sold", "revenue": "Revenue",
        "avg_units_day": "Avg Units / Day", "avg_revenue_day": "Avg Revenue / Day"})


TABLE_CFG = {
    "Units Sold": st.column_config.NumberColumn(format="localized"),
    "Revenue": st.column_config.NumberColumn(format="localized"),
    "Avg Units / Day": st.column_config.NumberColumn(format="%.1f"),
    "Avg Revenue / Day": st.column_config.NumberColumn(format="localized"),
}


def _missing() -> None:
    st.info("Sales data not available (data/sales.csv not found).")


def sales_overview(key: str) -> None:
    """Compact company-wide sales block for the Executive Dashboard."""
    sales = dl.get_sales()
    if sales is None or sales.empty:
        _missing()
        return
    total = dl.sales_summary(sales, "ALL").iloc[0]
    kpi_row([
        ("Units sold", int(total["units"]), "info",
         f"{total['start']:%d %b %Y} – {total['end']:%d %b %Y}"),
        ("Revenue", int(round(float(total["revenue"]))), "info", f"{int(total['days'])} days of sales"),
        ("Avg units / day", int(round(float(total["avg_units_day"]))), "info", "all SKUs & warehouses"),
        ("Avg revenue / day", int(round(float(total["avg_revenue_day"]))), "info", "all SKUs & warehouses"),
    ])
    metric = st.radio("Metric", list(METRICS), horizontal=True, key=f"{key}_metric",
                      label_visibility="collapsed")
    monthly = dl.sales_summary(sales, "M")
    st.plotly_chart(_chart(monthly, "M", metric, "Total", f"Monthly {metric.lower()} — all SKUs & warehouses"),
                    width="stretch", config=charts.CONFIG, theme=None)
    t_year, t_month = st.tabs(["By year", "By month"])
    with t_year:
        st.dataframe(_table(dl.sales_summary(sales, "Y"), "Y"), hide_index=True, width="stretch",
                     column_config=TABLE_CFG)
    with t_month:
        st.dataframe(_table(monthly, "M"), hide_index=True, width="stretch", column_config=TABLE_CFG)
    st.caption("Light bars / “partial” = months or years only partly covered by the sales data "
               f"({total['start']:%d %b %Y} – {total['end']:%d %b %Y}). Average per day = total ÷ days.")


def sales_explorer(key: str, default_sku: str | None = None, default_wh: str | None = None) -> None:
    """Full sales view with SKU / warehouse filters and month / year / average-per-day breakdowns."""
    sales = dl.get_sales()
    if sales is None or sales.empty:
        _missing()
        return
    skus = ["All SKUs"] + sorted(sales["sku_id"].unique())
    whs = ["All warehouses"] + sorted(sales["warehouse_id"].unique())
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    sku = c1.selectbox("SKU", skus, index=skus.index(default_sku) if default_sku in skus else 0, key=f"{key}_sku")
    wh = c2.selectbox("Warehouse", whs, index=whs.index(default_wh) if default_wh in whs else 0, key=f"{key}_wh")
    view = c3.selectbox("View by", ["Month", "Year"], key=f"{key}_view")
    metric = c4.selectbox("Metric", list(METRICS), key=f"{key}_metric")
    basis = st.radio("Show", ["Total", "Average per day"], horizontal=True, key=f"{key}_basis")

    s = sales
    if sku != "All SKUs":
        s = s[s["sku_id"] == sku]
    if wh != "All warehouses":
        s = s[s["warehouse_id"] == wh]
    if s.empty:
        st.info("No sales records for this selection.")
        return

    scope = f"{sku if sku != 'All SKUs' else 'all SKUs'} · {wh if wh != 'All warehouses' else 'all warehouses'}"
    total = dl.sales_summary(s, "ALL").iloc[0]
    kpi_row([
        ("Units sold", int(total["units"]), "info", scope),
        ("Revenue", int(round(float(total["revenue"]))), "info", f"{int(total['days'])} days"),
        ("Avg units / day", int(round(float(total["avg_units_day"]))), "info", "total ÷ days"),
        ("Avg revenue / day", int(round(float(total["avg_revenue_day"]))), "info", "total ÷ days"),
    ])
    if sku != "All SKUs" and "unit_price" in s:
        prices = s["unit_price"]
        st.caption(f"Unit price in this selection: {prices.min():,.2f} – {prices.max():,.2f} "
                   f"(average {prices.mean():,.2f})")

    freq = "M" if view == "Month" else "Y"
    summary = dl.sales_summary(s, freq)
    st.plotly_chart(_chart(summary, freq, metric, basis,
                           f"{metric} by {view.lower()}{' — average per day' if basis != 'Total' else ''} ({scope})"),
                    width="stretch", config=charts.CONFIG, theme=None)
    st.dataframe(_table(summary, freq), hide_index=True, width="stretch", column_config=TABLE_CFG)
    st.caption(f"Source: sales dataset ({total['start']:%d %b %Y} – {total['end']:%d %b %Y}). "
               "Average per day = total ÷ days with sales records in the period. "
               "Sales history is shown for context; forecasts and decisions use the demand data.")
