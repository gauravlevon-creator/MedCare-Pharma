"""Page 2 — Inventory Monitor (E1 inventory monitoring)."""

from __future__ import annotations

import streamlit as st

from frontend.components import charts
from frontend.components.kpi_cards import kpi_row
from frontend.components.styles import page_title, section
from frontend.utils.formatting import action_label, icon_text
from frontend.views.common import require_decisions

# (backend column, display name). Optional columns are skipped if absent.
COLUMNS = [
    ("sku_id", "SKU"),
    ("warehouse_id", "Warehouse"),
    ("current_stock", "Current Stock"),
    ("expired_stock", "Expired Stock"),
    ("usable_stock_before", "Usable Stock"),
    ("min_threshold", "Min Threshold"),
    ("safety_stock", "Safety Stock"),
    ("lead_time_demand", "Lead-Time Demand"),
    ("required_stock", "Required Stock"),
    ("projected_inventory", "Projected Inventory"),
    ("days_of_stock", "Days of Stock"),
    ("risk_status", "Risk"),
    ("e1_threshold_alert", "E1 Alert"),
    ("expiry_risk", "Expiry Risk"),
    ("usable_stock_after", "Usable After Transfers"),
    ("final_action", "Final Action"),
]


def render() -> None:
    page_title("📦 Inventory Monitor",
               "Stock position for every SKU and warehouse. Usable stock excludes expired batches; "
               "required stock = forecast demand during the supplier lead time + safety stock.")
    d = require_decisions()

    f1, f2, f3, f4 = st.columns(4)
    skus = f1.multiselect("SKU", sorted(d["sku_id"].unique()), placeholder="All SKUs")
    whs = f2.multiselect("Warehouse", sorted(d["warehouse_id"].unique()), placeholder="All warehouses")
    risk_opts = [r for r in ("HIGH", "MEDIUM", "LOW") if r in set(d["risk_status"])]
    risks = f3.multiselect("Risk (P1 stock-out)", risk_opts, placeholder="All risk levels")
    act_opts = [a for a in ("REORDER", "TRANSFER", "NO_ACTION") if a in set(d["final_action"])]
    acts = f4.multiselect("Action", act_opts, format_func=action_label, placeholder="All actions")
    e1_only = st.toggle("Only items with an E1 threshold alert", value=False)

    view = d
    if skus:
        view = view[view["sku_id"].isin(skus)]
    if whs:
        view = view[view["warehouse_id"].isin(whs)]
    if risks:
        view = view[view["risk_status"].isin(risks)]
    if acts:
        view = view[view["final_action"].isin(acts)]
    if e1_only and "e1_threshold_alert" in view:
        view = view[view["e1_threshold_alert"]]

    kpi_row([
        ("Items shown", len(view), "info", f"of {len(d)}"),
        ("🔴 HIGH risk", int((view["risk_status"] == "HIGH").sum()), "critical", ""),
        ("🟠 MEDIUM risk", int((view["risk_status"] == "MEDIUM").sum()), "warning", ""),
        ("🟡 E1 alerts", int(view["e1_threshold_alert"].sum()) if "e1_threshold_alert" in view else None,
         "attention", ""),
        ("REORDER", int((view["final_action"] == "REORDER").sum()), "warning", ""),
        ("TRANSFER", int((view["final_action"] == "TRANSFER").sum()), "info", ""),
    ])

    if view.empty:
        st.info("No SKU/warehouse items match these filters.")
        return

    section("Inventory table")
    cols = [(c, n) for c, n in COLUMNS if c in view.columns]
    table = view[[c for c, _ in cols]].rename(columns=dict(cols)).copy()
    if "Risk" in table:
        table["Risk"] = table["Risk"].map(lambda v: icon_text("risk", v))
    if "Expiry Risk" in table:
        table["Expiry Risk"] = table["Expiry Risk"].map(lambda v: icon_text("risk", v))
    if "Final Action" in table:
        table["Final Action"] = table["Final Action"].map(lambda v: icon_text("action", v))
    if "E1 Alert" in table:
        table["E1 Alert"] = table["E1 Alert"].map(lambda v: "🟡 Yes" if v else "No")
    num_cfg = {name: st.column_config.NumberColumn(name, format="%.2f")
               for name in ("Lead-Time Demand", "Required Stock", "Projected Inventory", "Days of Stock")
               if name in table}
    st.dataframe(table, hide_index=True, width="stretch", height=min(520, 38 + 35 * len(table)),
                 column_config=num_cfg)
    st.download_button("Download filtered table (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="inventory_monitor.csv", mime="text/csv")

    section("Usable stock vs required stock")
    fig, truncated = charts.usable_vs_required(view)
    if truncated:
        st.caption("Showing the 40 items with the largest shortage. Filter by SKU or warehouse to narrow down.")
    st.plotly_chart(fig, width="stretch", config=charts.CONFIG, theme=None)
    st.caption("A bar below its black required-stock tick is a shortage; above it is excess. "
               "Light-blue bars show usable stock after the planned transfers.")

    with st.expander("Why each item has its risk level"):
        reasons = view[["sku_id", "warehouse_id", "risk_reason", "e1_reason", "expiry_reason"]].rename(columns={
            "sku_id": "SKU", "warehouse_id": "Warehouse", "risk_reason": "P1 risk reason",
            "e1_reason": "E1 reason", "expiry_reason": "Expiry reason"})
        st.dataframe(reasons, hide_index=True, width="stretch")
