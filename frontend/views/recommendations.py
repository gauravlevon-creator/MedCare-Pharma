"""Page 5 — Recommendations (final decision-engine output)."""

from __future__ import annotations

import streamlit as st

from frontend.components.kpi_cards import kpi_row
from frontend.components.recommendation_cards import (decision_trace, no_action_card, reorder_card,
                                                      transfer_card, transfer_decision_card)
from frontend.components.styles import page_title, section
from frontend.utils import data_loader as dl
from frontend.views.common import forecast_total_30, item_picker, require_decisions, transfers_for

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _grid(cards: list[str], per_row: int = 2) -> None:
    for i in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for c, html in zip(cols, cards[i:i + per_row]):
            c.markdown(html, unsafe_allow_html=True)


def render() -> None:
    page_title("💡 Recommendations",
               "One final action for every SKU and warehouse: TRANSFER stock between warehouses, "
               "REORDER from the supplier, or NO ACTION.")
    d = require_decisions()
    transfers = dl.get_transfer_data()

    section("Decision trace")
    sku, wh = item_picker("rec")
    row = dl.decision_row(sku, wh) if sku else None
    if row is None:
        st.info("No recommendation available for this SKU/warehouse.")
    else:
        decision_trace(row, forecast_total_30(sku, wh), transfers_for(sku, wh, "in"))
        tin, tout = transfers_for(sku, wh, "in"), transfers_for(sku, wh, "out")
        if tin or tout:
            with st.expander(f"Transfer records involving {sku} / {wh} ({len(tin)} in, {len(tout)} out)",
                             expanded=bool(tin)):
                _grid([transfer_card(t) for t in tin + tout])

    section("All recommendations")
    f1, f2, f3 = st.columns(3)
    skus = f1.multiselect("SKU", sorted(d["sku_id"].unique()), placeholder="All SKUs", key="r_sku")
    whs = f2.multiselect("Warehouse", sorted(d["warehouse_id"].unique()), placeholder="All warehouses", key="r_wh")
    limit = f3.selectbox("Cards per tab", [10, 20, 50, 200], index=0)
    view = d
    if skus:
        view = view[view["sku_id"].isin(skus)]
    if whs:
        view = view[view["warehouse_id"].isin(whs)]

    tv = transfers
    if tv is not None and not tv.empty:
        if skus:
            tv = tv[tv["sku_id"].isin(skus)]
        if whs:
            tv = tv[tv["source_warehouse"].isin(whs) | tv["destination_warehouse"].isin(whs)]

    reorders = view[view["final_action"] == "REORDER"]
    transfer_dec = view[view["final_action"] == "TRANSFER"]
    no_action = view[view["final_action"] == "NO_ACTION"]
    kpi_row([
        ("REORDER decisions", len(reorders), "warning", f"{int(reorders['reorder_quantity'].sum()):,} units"),
        ("TRANSFER decisions", len(transfer_dec), "info", ""),
        ("Transfer records", None if tv is None else len(tv), "info",
         "" if tv is None else f"{int(tv['transfer_quantity'].sum()):,} units moved"),
        ("NO ACTION", len(no_action), "healthy", "no shortage, nothing inbound"),
    ])

    t1, t2, t3 = st.tabs([f"🔁 TRANSFER ({len(transfer_dec)} decisions)",
                          f"🛒 REORDER ({len(reorders)})", f"✅ NO ACTION ({len(no_action)})"])
    with t1:
        if tv is None:
            st.warning("outputs/transfers.csv not found. Run `python main.py decide-all`.")
        elif tv.empty and transfer_dec.empty:
            st.info("No transfer recommendations for these filters.")
        else:
            st.caption("Transfer records (batch level) — expiry-prevention transfers first, earliest expiry first.")
            ordered = tv.assign(_t=(tv["transfer_type"] != "EXPIRY_PREVENTION").astype(int)).sort_values(
                ["_t", "days_to_expiry", "sku_id"]).drop(columns="_t")
            _grid([transfer_card(t) for t in ordered.head(limit).to_dict("records")])
            if len(ordered) > limit:
                st.caption(f"Showing {limit} of {len(ordered)} transfer records.")
            if not transfer_dec.empty:
                with st.expander(f"Warehouse-level TRANSFER decisions ({len(transfer_dec)})"):
                    _grid([transfer_decision_card(r) for r in transfer_dec.head(limit).to_dict("records")])
    with t2:
        if reorders.empty:
            st.info("No reorder recommendations for these filters.")
        else:
            ordered = reorders.assign(_r=reorders["risk_status"].map(RISK_ORDER)).sort_values(
                ["_r", "remaining_shortage"], ascending=[True, False]).drop(columns="_r")
            _grid([reorder_card(r) for r in ordered.head(limit).to_dict("records")])
            if len(ordered) > limit:
                st.caption(f"Showing {limit} of {len(ordered)} reorder decisions (HIGH risk, largest shortage first).")
    with t3:
        if no_action.empty:
            st.info("No NO ACTION items for these filters.")
        else:
            st.caption("The engine does not recommend an action for every item — these are covered as-is.")
            _grid([no_action_card(r) for r in no_action.head(limit).to_dict("records")])
            if len(no_action) > limit:
                st.caption(f"Showing {limit} of {len(no_action)} items.")

    with st.expander("Download recommendations (CSV)"):
        cols = ["sku_id", "warehouse_id", "final_action", "reorder_quantity", "transfer_in_quantity",
                "transfer_out_quantity", "risk_status", "expiry_risk", "usable_stock_before", "usable_stock_after",
                "required_stock", "remaining_shortage", "lead_time_days", "reason"]
        out = view[[c for c in cols if c in view.columns]]
        st.download_button("Download", out.to_csv(index=False).encode("utf-8"), "recommendations.csv", "text/csv")
