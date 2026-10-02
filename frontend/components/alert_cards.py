"""Alert cards for the Risk & Alerts page."""

from __future__ import annotations

from frontend.utils.formatting import (ALERT_TYPE_LABEL, STATUS, VALUE_LABEL, action_badge, esc,
                                       num, severity_badge, tone_for)


def alert_card(a: dict) -> str:
    tone = tone_for("severity", a.get("severity"))
    color = STATUS[tone]["bar"]
    atype = str(a.get("alert_type", ""))
    vlabel = VALUE_LABEL.get(str(a.get("value_label")), str(a.get("value_label", "Value")))
    return (f'<div class="mc-card" style="border-left-color:{color};">'
            f'<div class="top"><span class="title">{esc(a.get("sku_id"))} · {esc(a.get("warehouse_id"))} — '
            f'{esc(ALERT_TYPE_LABEL.get(atype, atype))}</span>'
            f'<span>{severity_badge(a.get("severity"))} {action_badge(a.get("recommended_action"))}</span></div>'
            f'<div class="reason">{esc(a.get("message"))}</div>'
            f'<div class="mc-muted" style="margin-top:6px;">{esc(vlabel)}: <b>{num(a.get("value"), 2)}</b></div>'
            f"</div>")
