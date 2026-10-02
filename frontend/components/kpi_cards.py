"""KPI cards (HTML inside Streamlit)."""

from __future__ import annotations

import streamlit as st

from frontend.utils.formatting import STATUS, esc, smart_num


def kpi_card(label: str, value, tone: str = "neutral", hint: str = "") -> str:
    color = STATUS.get(tone, STATUS["neutral"])["bar"]
    shown = "Not available" if value is None else smart_num(value)
    return (f'<div class="mc-kpi" style="border-left-color:{color};">'
            f'<div class="lbl">{esc(label)}</div><div class="val">{shown}</div>'
            + (f'<div class="hint">{esc(hint)}</div>' if hint else "") + "</div>")


def kpi_row(items: list[tuple], columns: int | None = None) -> None:
    """items: (label, value, tone, hint)."""
    cols = st.columns(columns or len(items))
    for i, item in enumerate(items):
        label, value, tone, hint = (list(item) + ["neutral", ""])[:4]
        with cols[i % len(cols)]:
            st.markdown(kpi_card(label, value, tone, hint), unsafe_allow_html=True)
