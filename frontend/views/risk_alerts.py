"""Page 4 — Risk & Alerts (actual Phase 9 alert-engine output + batch expiry intelligence)."""

from __future__ import annotations

import streamlit as st

from frontend.components import charts
from frontend.components.alert_cards import alert_card
from frontend.components.kpi_cards import kpi_row
from frontend.components.recommendation_cards import decision_trace
from frontend.components.styles import banner, page_title, section
from frontend.utils import data_loader as dl
from frontend.utils.formatting import (ALERT_TYPE_LABEL, ALERT_TYPE_ORDER, VALUE_LABEL, action_label,
                                       icon_text)
from frontend.views.common import forecast_total_30, require_decisions, set_focus, transfers_for

SEVERITIES = ["HIGH", "MEDIUM", "LOW", "INFO"]


def render() -> None:
    page_title("🚨 Risk & Alerts",
               "Alerts from the MedCare alert engine, most urgent first.")
    require_decisions()
    tab_alerts, tab_expiry = st.tabs(["🚨 Alerts", "⏳ Batch expiry intelligence"])
    with tab_alerts:
        _alerts_tab()
    with tab_expiry:
        _expiry_tab()


def _alerts_tab() -> None:
    alerts = dl.get_alerts()
    if alerts is None:
        banner("<b>outputs/alerts.csv not found.</b> Run <code>python main.py decide-all</code>.", "warn")
        return
    if alerts.empty:
        st.success("No active alerts.")
        return

    sev = alerts["severity"].value_counts()
    kpi_row([
        ("Total alerts", len(alerts), "info", ""),
        ("🔴 HIGH", int(sev.get("HIGH", 0)), "critical", "stock-out / expiry"),
        ("🟠 MEDIUM", int(sev.get("MEDIUM", 0)), "warning", "stock-out / expiry"),
        ("🟡 LOW", int(sev.get("LOW", 0)), "attention", "E1 threshold"),
        ("🔵 INFO", int(sev.get("INFO", 0)), "info", "transfer / reorder"),
    ])

    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.plotly_chart(charts.alerts_by_type(alerts), width="stretch", config=charts.CONFIG, theme=None)
    with c2:
        section("Most severe alerts")
        for a in alerts.head(3).to_dict("records"):
            st.markdown(alert_card(a), unsafe_allow_html=True)

    section("Alert table")
    f1, f2, f3, f4 = st.columns(4)
    sevs = f1.multiselect("Severity", [s for s in SEVERITIES if s in set(alerts["severity"])],
                          placeholder="All severities")
    type_opts = [t for t in ALERT_TYPE_ORDER if t in set(alerts["alert_type"])]
    types = f2.multiselect("Alert type", type_opts, format_func=lambda t: ALERT_TYPE_LABEL.get(t, t),
                           placeholder="All alert types")
    skus = f3.multiselect("SKU", sorted(alerts["sku_id"].unique()), placeholder="All SKUs")
    whs = f4.multiselect("Warehouse", sorted(alerts["warehouse_id"].unique()), placeholder="All warehouses")

    view = alerts
    if sevs:
        view = view[view["severity"].isin(sevs)]
    if types:
        view = view[view["alert_type"].isin(types)]
    if skus:
        view = view[view["sku_id"].isin(skus)]
    if whs:
        view = view[view["warehouse_id"].isin(whs)]
    view = view.reset_index(drop=True)
    if view.empty:
        st.info("No active alerts for these filters.")
        return

    table = view.assign(
        Severity=view["severity"].map(lambda v: icon_text("severity", v)),
        Type=view["alert_type"].map(lambda t: ALERT_TYPE_LABEL.get(t, t)),
        ValueLabel=view["value_label"].map(lambda v: VALUE_LABEL.get(v, v)),
        Action=view["recommended_action"].map(lambda v: icon_text("action", v)),
    )[["Severity", "sku_id", "warehouse_id", "Type", "message", "value", "ValueLabel", "Action"]].rename(columns={
        "sku_id": "SKU", "warehouse_id": "Warehouse", "Type": "Alert Type", "message": "Message",
        "value": "Value", "ValueLabel": "Value meaning", "Action": "Recommended Action"})
    st.caption(f"{len(view)} alerts · select a row to see full details")
    event = st.dataframe(table, hide_index=True, width="stretch", height=min(460, 38 + 35 * len(table)),
                         on_select="rerun", selection_mode="single-row", key="alert_table",
                         column_config={"Value": st.column_config.NumberColumn(format="%.2f"),
                                        "Message": st.column_config.TextColumn(width="medium")})
    rows = event.selection.rows if event and hasattr(event, "selection") else []
    if not rows:
        return
    a = view.iloc[rows[0]].to_dict()
    section(f"Alert detail — {a['sku_id']} / {a['warehouse_id']}")
    st.markdown(alert_card(a), unsafe_allow_html=True)
    d = dl.decision_row(a["sku_id"], a["warehouse_id"])
    if d is None:
        st.info("No recommendation available.")
        return
    set_focus(a["sku_id"], a["warehouse_id"])
    decision_trace(d, forecast_total_30(a["sku_id"], a["warehouse_id"]),
                   transfers_for(a["sku_id"], a["warehouse_id"], "in"))
    with st.expander("All details for this SKU / warehouse"):
        shown = {k: (", ".join(map(str, v)) if isinstance(v, list) and k != "transfers" else v)
                 for k, v in d.items() if k != "transfers"}
        st.json(shown, expanded=True)


def _expiry_tab() -> None:
    st.caption("Batch-by-batch expiry check. Stock is used earliest-expiry-first against the demand forecast; "
               "“Potential Expiry Excess” is stock expected to expire before it can be used.")
    res = dl.get_batch_expiry()
    if res is None:
        banner("Batch expiry needs the database and the saved forecast "
               "(<code>python main.py init-db</code>, <code>python main.py forecast-all</code>).", "warn")
        return
    batches, summary, errors = res
    if batches.empty:
        st.info(f"No batch expiry data available. {errors.get('_all', '')}")
        return
    decisions = dl.get_decisions()

    kpi_row([
        ("🔴 HIGH expiry-risk items", int((summary["expiry_risk"] == "HIGH").sum()), "critical", ""),
        ("🟠 MEDIUM expiry-risk items", int((summary["expiry_risk"] == "MEDIUM").sum()), "warning", ""),
        ("Batches with expiry excess", int((batches["potential_expiry_excess"] > 0).sum()), "critical", ""),
        ("Units forecast to expire unused", round(float(summary["potential_expiry_excess"].sum()), 2),
         "warning", "across all batches"),
        ("Expired units (unusable)", int(summary["expired_quantity"].sum()), "neutral", "excluded from usable"),
    ])
    st.plotly_chart(charts.expiry_excess_by_item(summary), width="stretch", config=charts.CONFIG, theme=None)

    # Join backend transfers for each batch (what the allocation engine does with it)
    transfers = dl.get_transfer_data()
    tmap = {}
    if transfers is not None and not transfers.empty:
        for t in transfers.to_dict("records"):
            tmap.setdefault((t["sku_id"], t["source_warehouse"], str(t["batch_id"])), []).append(
                f"{t['transfer_type'].replace('_', ' ').title()}: {t['transfer_quantity']} → {t['destination_warehouse']}")

    f1, f2, f3, f4 = st.columns(4)
    skus = f1.multiselect("SKU", sorted(batches["sku_id"].unique()), placeholder="All SKUs", key="ex_sku")
    whs = f2.multiselect("Warehouse", sorted(batches["warehouse_id"].unique()), placeholder="All warehouses",
                         key="ex_wh")
    risks = f3.multiselect("Item expiry risk", ["HIGH", "MEDIUM", "LOW"], default=["HIGH"], key="ex_risk")
    only_excess = f4.toggle("Only batches with expiry excess", value=True, key="ex_only")

    v = batches
    if skus:
        v = v[v["sku_id"].isin(skus)]
    if whs:
        v = v[v["warehouse_id"].isin(whs)]
    if risks:
        v = v[v["item_expiry_risk"].isin(risks)]
    if only_excess:
        v = v[v["potential_expiry_excess"] > 0]
    if v.empty:
        st.info("No batches match these filters.")
        return

    action_map = {} if decisions is None else {
        (r["sku_id"], r["warehouse_id"]): r["final_action"] for r in decisions.to_dict("records")}
    table = v.assign(
        Status=v["expiry_status"].map(lambda s: icon_text("expiry_status", s)),
        Risk=v["item_expiry_risk"].map(lambda s: icon_text("risk", s)),
        Usable=v["usable"].map(lambda b: "Yes" if b else "No (expired)"),
        TransferEligible=v["transfer_eligible"].map(lambda b: "Yes" if b else "No"),
        Transfer=[("; ".join(tmap.get((r.sku_id, r.warehouse_id, str(r.batch_id)), [])) or "—")
                  for r in v.itertuples()],
        Action=[action_label(action_map.get((r.sku_id, r.warehouse_id), "—")) for r in v.itertuples()],
    )[["batch_id", "sku_id", "warehouse_id", "quantity", "expiry_date", "days_to_expiry",
       "potential_expiry_excess", "Risk", "Status", "Transfer", "TransferEligible", "Usable",
       "forecast_demand_through_expiry", "expected_consumption", "Action"]].rename(columns={
        "batch_id": "Batch ID", "sku_id": "SKU", "warehouse_id": "Warehouse", "quantity": "Quantity",
        "expiry_date": "Expiry Date", "days_to_expiry": "Days to Expiry", "Status": "Expiry Status",
        "forecast_demand_through_expiry": "Forecast Demand to Expiry", "expected_consumption": "Expected Use",
        "potential_expiry_excess": "Potential Expiry Excess", "Risk": "Item Expiry Risk",
        "TransferEligible": "Transfer Eligible", "Transfer": "Planned Transfer",
        "Action": "Item Final Action"})
    st.dataframe(table, hide_index=True, width="stretch", height=min(520, 38 + 35 * len(table)),
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ("Forecast Demand to Expiry", "Expected Use", "Potential Expiry Excess")})
    if errors:
        st.caption(f"Expiry analysis unavailable for {len(errors)} item(s): "
                   + ", ".join(f"{k} ({v})" for k, v in list(errors.items())[:5]))
