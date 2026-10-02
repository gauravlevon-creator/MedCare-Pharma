"""
Recommendation cards and the decision trace (end-to-end flow for one item).

Every number shown is a backend field from decisions.csv / transfers.csv /
forecast_output.csv. The only arithmetic is summing the saved 30-day
forecast for display (spec section 9).
"""

from __future__ import annotations

import streamlit as st

from frontend.utils.formatting import (STATUS, action_badge, action_label, badge, esc,
                                       num, risk_badge, smart_num, tone_for, warehouses_text)


def _card(tone: str, top_left: str, top_right: str, qty: str, grid: list[tuple[str, str]],
          reason: str = "") -> str:
    color = STATUS.get(tone, STATUS["neutral"])["bar"]
    cells = "".join(f"<div>{esc(k)}<b>{v}</b></div>" for k, v in grid)
    return (f'<div class="mc-card" style="border-left-color:{color};">'
            f'<div class="top"><span class="title">{top_left}</span><span>{top_right}</span></div>'
            + (f'<div class="qty">{qty}</div>' if qty else "")
            + (f'<div class="grid">{cells}</div>' if cells else "")
            + (f'<div class="reason">{esc(reason)}</div>' if reason else "") + "</div>")


def transfer_card(t: dict) -> str:
    ttype = str(t.get("transfer_type", ""))
    tone = "critical" if ttype == "EXPIRY_PREVENTION" else "info"
    title = f"{esc(t.get('sku_id'))} · {esc(t.get('source_warehouse'))} → {esc(t.get('destination_warehouse'))}"
    grid = [
        ("Batch", esc(t.get("batch_id", "—"))),
        ("Transfer type", esc(ttype.replace("_", " ").title() if ttype else "—")),
        ("Expiry date", esc(t.get("expiry_date", "—"))),
        ("Days to expiry", smart_num(t.get("days_to_expiry"))),
        ("Source usable", f"{smart_num(t.get('source_usable_before'))} → {smart_num(t.get('source_usable_after'))}"),
        ("Destination usable",
         f"{smart_num(t.get('destination_usable_before'))} → {smart_num(t.get('destination_usable_after'))}"),
    ]
    return _card(tone, title, badge(ttype.replace("_", " "), tone, icon=False),
                 f"{smart_num(t.get('transfer_quantity'))} units", grid, t.get("reason", ""))


def reorder_card(d: dict) -> str:
    tone = tone_for("risk", d.get("risk_status"))
    title = f"{esc(d['sku_id'])} · {esc(d['warehouse_id'])}"
    grid = [
        ("P1 stock-out risk", esc(d.get("risk_status", "—"))),
        ("Lead time", f"{smart_num(d.get('lead_time_days'))} days"),
        ("Remaining shortage", smart_num(d.get("remaining_shortage"))),
        ("Transfers in", f"{smart_num(d.get('transfer_in_quantity'))}"
                         + (f" from {esc(warehouses_text(d.get('source_warehouses')))}"
                            if d.get("source_warehouses") else "")),
        ("Usable stock", f"{smart_num(d.get('usable_stock_before'))} → {smart_num(d.get('usable_stock_after'))}"),
        ("Required stock", smart_num(d.get("required_stock"))),
    ]
    return _card(tone, title, action_badge("REORDER"),
                 f"Reorder {smart_num(d.get('reorder_quantity'))} units", grid, d.get("reason", ""))


def transfer_decision_card(d: dict) -> str:
    title = f"{esc(d['sku_id'])} · {esc(d['warehouse_id'])}"
    role = str(d.get("transfer_role") or "—").replace("_", " ").title()
    grid = [
        ("Role", esc(role)),
        ("Receives", f"{smart_num(d.get('transfer_in_quantity'))}"
                     + (f" from {esc(warehouses_text(d.get('source_warehouses')))}" if d.get("source_warehouses") else "")),
        ("Expiry-risk units out", smart_num(d.get("expiry_transfer_out_quantity"))),
        ("P1 risk", esc(d.get("risk_status", "—"))),
        ("Expiry risk", esc(d.get("expiry_risk", "—"))),
        ("Usable stock", f"{smart_num(d.get('usable_stock_before'))} → {smart_num(d.get('usable_stock_after'))}"),
    ]
    return _card("info", title, action_badge("TRANSFER"), "", grid, d.get("reason", ""))


def no_action_card(d: dict) -> str:
    title = f"{esc(d['sku_id'])} · {esc(d['warehouse_id'])}"
    grid = [
        ("P1 stock-out risk", esc(d.get("risk_status", "—"))),
        ("Usable stock", smart_num(d.get("usable_stock_before"))),
        ("Required stock", smart_num(d.get("required_stock"))),
        ("Days of stock", smart_num(d.get("days_of_stock"))),
        ("Transfer role", esc(str(d.get("transfer_role") or "—").replace("_", " ").title())),
        ("Expiry risk", esc(d.get("expiry_risk", "—"))),
    ]
    return _card("healthy", title, action_badge("NO_ACTION"), "", grid, d.get("reason", ""))


def top_recommendation_card(d: dict, alert: dict | None = None) -> str:
    """Compact card for the Executive Dashboard."""
    action = d.get("final_action")
    risk = d.get("risk_status")
    lines = []
    if d.get("transfer_in_quantity"):
        lines.append(f"{smart_num(d['transfer_in_quantity'])} units received through internal transfer"
                     f" from {esc(warehouses_text(d.get('source_warehouses')))}")
    if d.get("expiry_transfer_out_quantity"):
        lines.append(f"{smart_num(d['expiry_transfer_out_quantity'])} expiry-risk units sent to "
                     f"{esc(warehouses_text(d.get('destination_warehouses')))}")
    if action == "REORDER":
        lines.append(f"Remaining shortage ≈ {num(d.get('remaining_shortage'), 2)} units")
    if d.get("expiry_risk") == "HIGH":
        lines.append(f"HIGH expiry risk: {num(d.get('potential_expiry_excess'), 2)} units may expire unused")
    if action == "REORDER":
        headline = f"→ REORDER {smart_num(d.get('reorder_quantity'))} units"
    elif action == "TRANSFER":
        headline = "→ TRANSFER"
    else:
        headline = f"→ {action_label(action)}"
    tone = tone_for("risk", risk) if risk in ("HIGH", "MEDIUM") else tone_for("action", action)
    body = "".join(f'<div class="reason" style="margin-top:2px;">{x}</div>' for x in lines)
    color = STATUS[tone]["bar"]
    trigger = f'<div class="mc-muted" style="margin-top:4px;">Alert: {esc(alert["alert_type"])}</div>' if alert else ""
    return (f'<div class="mc-card" style="border-left-color:{color};">'
            f'<div class="top"><span class="title">{esc(d["sku_id"])} — {esc(d["warehouse_id"])}</span>'
            f'{risk_badge(risk, "STOCK-OUT ") if risk else ""}</div>'
            f'{body}<div class="qty" style="font-size:1.05rem;">{esc(headline)}</div>{trigger}</div>')


# ------------------------------------------------------------------
# Decision trace (the M001 / W002 demo journey, for any selected item)
# ------------------------------------------------------------------

def _step(key: str, value: str, detail: str = "", tone: str = "neutral") -> str:
    color = STATUS.get(tone, STATUS["neutral"])["bar"]
    return (f'<div class="mc-step" style="border-top-color:{color};"><div class="k">{esc(key)}</div>'
            f'<div class="v">{value}</div>' + (f'<div class="d">{detail}</div>' if detail else "") + "</div>")


def decision_trace(d: dict, forecast_total_30: float | None, transfers_in: list[dict]) -> None:
    """Render the end-to-end P1 decision flow for one SKU/warehouse from backend values."""
    risk = d.get("risk_status")
    action = d.get("final_action")
    steps = [
        _step("Item", f"{esc(d['sku_id'])} / {esc(d['warehouse_id'])}",
              f"Recorded {smart_num(d.get('current_stock'))} · expired {smart_num(d.get('expired_stock'))}"),
        _step("Usable stock", f"{smart_num(d.get('usable_stock_before'))} units",
              f"Min threshold {smart_num(d.get('min_threshold'))}"
              + (" · E1 alert" if d.get("e1_threshold_alert") else ""),
              "attention" if d.get("e1_threshold_alert") else "neutral"),
        _step("Chronos-2 forecast",
              "Not available" if forecast_total_30 is None else f"{num(forecast_total_30, 2)} units / 30 d",
              f"Lead-time demand ({smart_num(d.get('lead_time_days'))} d): {num(d.get('lead_time_demand'), 2)}",
              "info"),
        _step("P1 stock-out risk", f"{esc(risk)}",
              f"Projected usable {num(d.get('projected_inventory'), 2)} · safety {smart_num(d.get('safety_stock'))}",
              tone_for("risk", risk)),
    ]
    if d.get("transfer_in_quantity"):
        by_src: dict[str, list] = {}
        for t in transfers_in:
            by_src.setdefault(str(t.get("source_warehouse")), []).append(t.get("transfer_quantity") or 0)
        src = ", ".join(f"{esc(w)} ({smart_num(sum(q))} in {len(q)} batch{'es' if len(q) > 1 else ''})"
                        for w, q in by_src.items()) or esc(warehouses_text(d.get("source_warehouses")))
        steps.append(_step("Internal transfer in", f"+{smart_num(d.get('transfer_in_quantity'))} units",
                           f"from {src}", "info"))
    else:
        steps.append(_step("Internal transfer in", "None", "no inbound transfer planned"))
    steps += [
        _step("Usable after transfers", f"{smart_num(d.get('usable_stock_after'))} units"),
        _step("Required stock", f"≈ {num(d.get('required_stock'), 2)}",
              "lead-time demand + safety stock"),
        _step("Remaining shortage", f"≈ {num(d.get('remaining_shortage'), 2)}",
              f"before transfers: {num(d.get('shortage_before'), 2)}",
              "critical" if (d.get("remaining_shortage") or 0) > 0 else "healthy"),
    ]
    if action == "REORDER":
        final = f"REORDER {smart_num(d.get('reorder_quantity'))} units"
    else:
        final = action_label(action)
    steps.append(_step("Final decision", esc(final), esc(d.get("reason", ""))[:160], tone_for("action", action)))
    st.markdown(f'<div class="mc-flow">{"".join(steps)}</div>', unsafe_allow_html=True)
