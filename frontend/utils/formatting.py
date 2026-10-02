"""Display formatting only: numbers, labels, status badges. No business logic."""

from __future__ import annotations

import html
import math

# Status semantics from the frontend spec:
#   🔴 High / Critical   🟠 Warning   🟡 Attention   🟢 Healthy
STATUS = {
    "critical": {"icon": "🔴", "fg": "#9f1f1f", "bg": "#fdecec", "bar": "#d03b3b"},
    "warning": {"icon": "🟠", "fg": "#8a3d12", "bg": "#fdf0e8", "bar": "#ec835a"},
    "attention": {"icon": "🟡", "fg": "#6b4e00", "bg": "#fff7dc", "bar": "#fab219"},
    "healthy": {"icon": "🟢", "fg": "#0b5e0b", "bg": "#e9f6e9", "bar": "#0ca30c"},
    "info": {"icon": "🔵", "fg": "#184f95", "bg": "#e8f1fc", "bar": "#2a78d6"},
    "neutral": {"icon": "⚪", "fg": "#52514e", "bg": "#f0efec", "bar": "#898781"},
}

# Colours used for charts (validated reference palette)
SERIES_BLUE = "#2a78d6"
SERIES_BLUE_LIGHT = "#9ec5f4"
SERIES_ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

RISK_TONE = {"HIGH": "critical", "MEDIUM": "warning", "LOW": "healthy"}
SEVERITY_TONE = {"HIGH": "critical", "MEDIUM": "warning", "LOW": "attention", "INFO": "info"}
ACTION_TONE = {"REORDER": "warning", "TRANSFER": "info", "NO_ACTION": "healthy"}
EXPIRY_STATUS_TONE = {"EXPIRED": "neutral", "NEAR_EXPIRY": "critical",
                      "EXPIRING_SOON": "warning", "NORMAL": "healthy"}

ACTION_LABEL = {"REORDER": "REORDER", "TRANSFER": "TRANSFER", "NO_ACTION": "NO ACTION"}

ALERT_TYPE_LABEL = {
    "STOCKOUT_RISK_HIGH": "Stock-out risk · HIGH",
    "STOCKOUT_RISK_MEDIUM": "Stock-out risk · MEDIUM",
    "EXPIRY_RISK_HIGH": "Expiry risk · HIGH",
    "EXPIRY_RISK_MEDIUM": "Expiry risk · MEDIUM",
    "E1_THRESHOLD": "E1 threshold",
    "TRANSFER_RECOMMENDED": "Transfer recommended",
    "REORDER_RECOMMENDED": "Reorder recommended",
}
ALERT_TYPE_ORDER = list(ALERT_TYPE_LABEL)

VALUE_LABEL = {
    "shortage_units": "Shortage (units)",
    "potential_expiry_excess_units": "Potential expiry excess (units)",
    "units_below_threshold": "Units below threshold",
    "transfer_units": "Transfer units",
    "reorder_units": "Reorder units",
}


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return isinstance(value, float) and math.isnan(value)
    except TypeError:
        return False


def num(value, decimals: int = 0, missing: str = "—") -> str:
    if is_missing(value) or value == "":
        return missing
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals == 0:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def smart_num(value, missing: str = "—") -> str:
    """Integers without decimals, fractional values with 2 decimals."""
    if is_missing(value) or value == "":
        return missing
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def action_label(action) -> str:
    return ACTION_LABEL.get(str(action), str(action))


def tone_for(kind: str, value) -> str:
    table = {"risk": RISK_TONE, "severity": SEVERITY_TONE, "action": ACTION_TONE,
             "expiry_status": EXPIRY_STATUS_TONE}.get(kind, {})
    return table.get(str(value), "neutral")


def badge(text, tone: str = "neutral", icon: bool = True) -> str:
    s = STATUS.get(tone, STATUS["neutral"])
    ic = f"{s['icon']} " if icon else ""
    return (f'<span class="mc-badge" style="color:{s["fg"]};background:{s["bg"]};">'
            f'{ic}{esc(text)}</span>')


def risk_badge(level, prefix: str = "") -> str:
    return badge(f"{prefix}{level}", tone_for("risk", level))


def action_badge(action) -> str:
    return badge(action_label(action), tone_for("action", action), icon=False)


def severity_badge(sev) -> str:
    return badge(sev, tone_for("severity", sev))


def icon_text(kind: str, value) -> str:
    """Plain-text '🔴 HIGH' for use inside st.dataframe cells."""
    if is_missing(value) or value == "":
        return "—"
    tone = tone_for(kind, value)
    label = action_label(value) if kind == "action" else str(value)
    return f"{STATUS[tone]['icon']} {label}"


def warehouses_text(values) -> str:
    if not values:
        return "—"
    return ", ".join(str(v) for v in values)
