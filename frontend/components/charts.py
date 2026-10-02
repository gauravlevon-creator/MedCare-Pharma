"""Plotly charts. They plot backend fields as-is; no business rule is added."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from frontend.utils.formatting import (ALERT_TYPE_LABEL, ALERT_TYPE_ORDER, GRID, INK, INK_2, MUTED,
                                       SERIES_BLUE, SERIES_BLUE_LIGHT, SERIES_ORANGE, STATUS,
                                       action_label, tone_for)

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK_2)
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _layout(fig: go.Figure, height: int = 340, legend: bool = True, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=64, r=16, t=84, b=56), font=FONT,
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", hovermode=kw.pop("hovermode", "closest"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
        title_y=0.97, title_yanchor="top", title_x=0.01,
        hoverlabel=dict(bgcolor="#ffffff", font_size=12, bordercolor=GRID),
        bargap=0.25, **kw)
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED), automargin=True)
    return fig


def risk_by_action(decisions: pd.DataFrame) -> go.Figure:
    """Items per P1 stock-out risk level, stacked by the backend's final action."""
    levels = [x for x in ("HIGH", "MEDIUM", "LOW") if x in set(decisions["risk_status"])]
    fig = go.Figure()
    for action in ("REORDER", "TRANSFER", "NO_ACTION"):
        counts = [int(((decisions["risk_status"] == lv) & (decisions["final_action"] == action)).sum())
                  for lv in levels]
        fig.add_bar(x=[f"{STATUS[tone_for('risk', lv)]['icon']} {lv}" for lv in levels], y=counts,
                    name=action_label(action), marker_color=STATUS[tone_for("action", action)]["bar"],
                    marker_line=dict(color="#ffffff", width=2),
                    hovertemplate="%{x} stock-out risk<br>" + action_label(action) + ": %{y} items<extra></extra>")
    fig.update_layout(barmode="stack", title=dict(text="Items by P1 stock-out risk → final action",
                                                  font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text="SKU/warehouse items")
    return _layout(fig)


def risk_by_warehouse(decisions: pd.DataFrame) -> go.Figure:
    """HIGH / MEDIUM stock-out and HIGH expiry risk counts per warehouse."""
    whs = sorted(decisions["warehouse_id"].unique())
    fig = go.Figure()
    series = [("Stock-out HIGH", decisions["risk_status"] == "HIGH", STATUS["critical"]["bar"]),
              ("Stock-out MEDIUM", decisions["risk_status"] == "MEDIUM", STATUS["warning"]["bar"]),
              ("Expiry HIGH", decisions["expiry_risk"] == "HIGH", "#4a3aa7")]
    for name, mask, color in series:
        counts = [int((mask & (decisions["warehouse_id"] == w)).sum()) for w in whs]
        fig.add_bar(x=whs, y=counts, name=name, marker_color=color,
                    hovertemplate="%{x}<br>" + name + ": %{y} items<extra></extra>")
    fig.update_layout(barmode="group", title=dict(text="Risk items per warehouse", font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text="Items")
    return _layout(fig)


def usable_vs_required(df: pd.DataFrame, max_items: int = 40) -> tuple[go.Figure, bool]:
    """
    Usable stock (before and after planned transfers) against required stock,
    per SKU/warehouse. Bars coloured by the backend's P1 risk.
    Returns (figure, truncated).
    """
    data = df.copy()
    truncated = len(data) > max_items
    if truncated:  # keep the items with the largest backend shortage_before
        data = data.sort_values("shortage_before", ascending=False).head(max_items)
    data = data.sort_values(["sku_id", "warehouse_id"])
    labels = data["sku_id"] + " · " + data["warehouse_id"]
    colors = [STATUS[tone_for("risk", r)]["bar"] for r in data["risk_status"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=data["usable_stock_before"], name="Usable stock now (colour = P1 risk)", marker_color=colors,
                customdata=data[["risk_status", "final_action", "shortage_before"]].to_numpy(),
                hovertemplate="<b>%{x}</b><br>Usable stock: %{y:,.0f}<br>P1 risk: %{customdata[0]}"
                              "<br>Shortage before transfers: %{customdata[2]:,.2f}"
                              "<br>Final action: %{customdata[1]}<extra></extra>")
    if "usable_stock_after" in data:
        fig.add_bar(x=labels, y=data["usable_stock_after"], name="Usable after planned transfers",
                    marker_color=SERIES_BLUE_LIGHT,
                    hovertemplate="<b>%{x}</b><br>Usable after transfers: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=labels, y=data["required_stock"], mode="markers", name="Required stock",
                    marker=dict(symbol="line-ew", size=22, line=dict(width=3, color=INK)),
                    hovertemplate="<b>%{x}</b><br>Required stock: %{y:,.2f}<extra></extra>")
    if "min_threshold" in data:
        fig.add_scatter(x=labels, y=data["min_threshold"], mode="markers", name="Min threshold (E1)",
                        marker=dict(symbol="diamond-open", size=9, color=MUTED, line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Min threshold: %{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group", title=dict(
        text="Usable stock vs required stock (bar colour = P1 risk: 🔴 HIGH 🟠 MEDIUM 🟢 LOW)",
        font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text="Units")
    fig.update_xaxes(tickangle=-45 if len(data) > 8 else 0)
    return _layout(fig, height=420), truncated


def demand_forecast(history: pd.DataFrame | None, forecast: pd.DataFrame, as_of: pd.Timestamp,
                    title: str) -> go.Figure:
    fig = go.Figure()
    if history is not None and not history.empty:
        fig.add_scatter(x=history["date"], y=history["demand_qty"], mode="lines", name="Historical demand",
                        line=dict(color=SERIES_BLUE, width=2),
                        hovertemplate="%{x|%d %b %Y}<br>Actual demand: %{y:,.0f}<extra></extra>")
    if forecast is not None and not forecast.empty:
        x, y = list(forecast["date"]), list(forecast["forecast_demand"])
        if history is not None and not history.empty:  # visually connect the two lines
            x = [history["date"].iloc[-1]] + x
            y = [history["demand_qty"].iloc[-1]] + y
        fig.add_scatter(x=x, y=y, mode="lines+markers", name="Chronos-2 forecast (30 days)",
                        line=dict(color=SERIES_ORANGE, width=2, dash="dash"), marker=dict(size=5),
                        hovertemplate="%{x|%d %b %Y}<br>Forecast demand: %{y:,.2f}<extra></extra>")
        fig.add_vrect(x0=as_of, x1=forecast["date"].max(), fillcolor="#fdf0e8", opacity=0.5, line_width=0,
                      annotation_text="Forecast window", annotation_position="top left",
                      annotation_font=dict(size=11, color=INK_2))
    fig.add_vline(x=as_of, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text="Units / day", rangemode="tozero")
    return _layout(fig, height=400, hovermode="x unified")


def alerts_by_type(alerts: pd.DataFrame) -> go.Figure:
    counts = alerts["alert_type"].value_counts()
    types = [t for t in ALERT_TYPE_ORDER if t in counts.index] + \
            [t for t in counts.index if t not in ALERT_TYPE_ORDER]
    sev = alerts.drop_duplicates("alert_type").set_index("alert_type")["severity"]
    fig = go.Figure(go.Bar(
        y=[ALERT_TYPE_LABEL.get(t, t) for t in types], x=[int(counts[t]) for t in types], orientation="h",
        marker_color=[STATUS[tone_for("severity", sev.get(t))]["bar"] for t in types],
        text=[int(counts[t]) for t in types], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} alerts<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor=GRID, showgrid=True)
    fig.update_layout(title=dict(text="Alerts by type (colour = severity)", font=dict(size=13, color=INK)))
    return _layout(fig, height=320, legend=False)


def expiry_excess_by_item(summary: pd.DataFrame, top: int = 20) -> go.Figure:
    data = summary[summary["potential_expiry_excess"] > 0].sort_values(
        "potential_expiry_excess", ascending=False).head(top)
    labels = data["sku_id"] + " · " + data["warehouse_id"]
    fig = go.Figure(go.Bar(
        x=labels, y=data["potential_expiry_excess"],
        marker_color=[STATUS[tone_for("risk", r)]["bar"] for r in data["expiry_risk"]],
        customdata=data[["expiry_risk"]].to_numpy(),
        hovertemplate="<b>%{x}</b><br>Potential expiry excess: %{y:,.2f}<br>Expiry risk: %{customdata[0]}<extra></extra>"))
    fig.update_layout(title=dict(text=f"Top {len(data)} items by potential expiry excess (colour = expiry risk)",
                                 font=dict(size=13, color=INK)))
    fig.update_yaxes(title_text="Units forecast to expire unused")
    fig.update_xaxes(tickangle=-45)
    return _layout(fig, height=360, legend=False)
