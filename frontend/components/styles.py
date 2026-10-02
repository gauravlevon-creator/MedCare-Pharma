"""Global CSS for the Streamlit app (HTML/CSS inside Streamlit, no separate website)."""

import streamlit as st

CSS = """
<style>
:root{
  --mc-ink:#0b0b0b; --mc-ink-2:#52514e; --mc-muted:#7a7873; --mc-line:#e3e5e8;
  --mc-surface:#ffffff; --mc-plane:#f7f8fa; --mc-brand:#104281; --mc-brand-2:#1c5cab;
  --mc-crit:#d03b3b; --mc-warn:#ec835a; --mc-attn:#fab219; --mc-good:#0ca30c; --mc-info:#2a78d6;
}
.block-container{padding-top:1.2rem; padding-bottom:2rem; max-width:1400px;}
[data-testid="stSidebar"]{background:#0f2a4a;}
[data-testid="stSidebar"] *{color:#e8eef6;}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#0b0b0b;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15);}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:rgba(255,255,255,.14);}
[data-testid="stSidebar"] .stButton button{background:#1c5cab;border:0;color:#fff;}
[data-testid="stSidebar"] [data-testid="stExpander"] details{border-color:rgba(255,255,255,.2);}

.mc-header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
  background:linear-gradient(90deg,#0f2a4a 0%,#1c5cab 100%);color:#fff;border-radius:12px;
  padding:14px 20px;margin-bottom:14px;}
.mc-header .brand{font-weight:800;letter-spacing:.12em;font-size:1.05rem;}
.mc-header .sub{opacity:.85;font-size:.9rem;margin-top:2px;}
.mc-header .meta{display:flex;gap:8px;flex-wrap:wrap;}
.mc-chip{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);border-radius:999px;
  padding:3px 10px;font-size:.78rem;white-space:nowrap;}

.mc-page-title{font-size:1.35rem;font-weight:700;color:var(--mc-ink);margin:2px 0 2px;}
.mc-page-sub{color:var(--mc-ink-2);font-size:.9rem;margin-bottom:12px;}
.mc-section{font-size:1rem;font-weight:700;color:var(--mc-ink);margin:18px 0 8px;
  padding-bottom:4px;border-bottom:1px solid var(--mc-line);}

.mc-kpi{background:var(--mc-surface);border:1px solid var(--mc-line);border-radius:10px;
  padding:12px 14px;height:100%;border-left:4px solid var(--mc-line);}
.mc-kpi .lbl{font-size:.74rem;color:var(--mc-ink-2);text-transform:uppercase;letter-spacing:.04em;
  font-weight:600;line-height:1.2;min-height:2.4em;}
.mc-kpi .val{font-size:1.75rem;font-weight:750;color:var(--mc-ink);line-height:1.15;margin-top:4px;}
.mc-kpi .hint{font-size:.74rem;color:var(--mc-muted);margin-top:2px;}

.mc-badge{display:inline-block;border-radius:6px;padding:2px 8px;font-size:.74rem;font-weight:700;
  letter-spacing:.02em;white-space:nowrap;}

.mc-card{background:var(--mc-surface);border:1px solid var(--mc-line);border-radius:10px;
  padding:12px 14px;margin-bottom:10px;border-left:4px solid var(--mc-line);}
.mc-card .top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;}
.mc-card .title{font-weight:750;font-size:1rem;color:var(--mc-ink);}
.mc-card .qty{font-size:1.25rem;font-weight:800;color:var(--mc-ink);margin:6px 0 2px;}
.mc-card .reason{font-size:.83rem;color:var(--mc-ink-2);margin-top:6px;line-height:1.4;}
.mc-card .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:6px 12px;margin-top:8px;}
.mc-card .grid div{font-size:.8rem;color:var(--mc-ink-2);}
.mc-card .grid b{display:block;color:var(--mc-ink);font-size:.92rem;}

.mc-flow{display:flex;flex-wrap:wrap;gap:0;align-items:stretch;margin:6px 0 4px;}
.mc-step{flex:1 1 130px;background:var(--mc-surface);border:1px solid var(--mc-line);border-radius:10px;
  padding:10px 12px;position:relative;margin:4px 14px 4px 0;border-top:4px solid var(--mc-line);}
.mc-step:not(:last-child)::after{content:"➜";position:absolute;right:-14px;top:50%;transform:translateY(-50%);
  color:#9aa3ad;font-size:.9rem;}
.mc-step .k{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mc-ink-2);font-weight:650;}
.mc-step .v{font-size:1.15rem;font-weight:780;color:var(--mc-ink);margin-top:3px;}
.mc-step .d{font-size:.74rem;color:var(--mc-muted);margin-top:2px;line-height:1.3;}

.mc-banner{border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:.88rem;}
.mc-banner.warn{background:#fff4e5;border:1px solid #f5c68a;color:#6b3a00;}
.mc-banner.crit{background:#fdecec;border:1px solid #f1a9a9;color:#7d1a1a;}
.mc-banner.info{background:#e8f1fc;border:1px solid #b7d3f6;color:#123a6b;}

.mc-muted{color:var(--mc-muted);font-size:.8rem;}
[data-testid="stDataFrame"]{border:1px solid var(--mc-line);border-radius:8px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="mc-page-title">{title}</div>'
                + (f'<div class="mc-page-sub">{subtitle}</div>' if subtitle else ""),
                unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="mc-section">{title}</div>', unsafe_allow_html=True)


def banner(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="mc-banner {kind}">{text}</div>', unsafe_allow_html=True)
