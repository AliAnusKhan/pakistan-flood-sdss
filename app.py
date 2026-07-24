"""
app.py
------
Pakistan Agricultural Flood Impact Assessment & Spatial Decision Support
System (SDSS). Executive SaaS-grade dashboard interface.

Prepared by Ali Anus

NOTE ON THIS REVISION
---------------------
This is a VISUAL / UX redesign only. All backend logic is untouched:
  * gee_engine.py is imported and called exactly as before
  * sidebar filters, session_state keys, @st.cache_resource / @st.cache_data
    caching, and the PDF export pipeline are all preserved 1:1
  * only presentation-layer code (CSS, layout, chart styling, small derived
    "insight" helper functions) has been added or reworked
"""

import streamlit as st
import pandas as pd
import io
import matplotlib
matplotlib.use("Agg")  # headless rendering - works on any server, no Chrome/browser dependency
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from fpdf import FPDF

from gee_engine import init_gee, get_district_10_parameters
from ai_insights import generate_ai_recommendations, is_configured as ai_is_configured
from config import (
    CURRENT_YEAR,
    PROVINCE_DISTRICT_MAP,
    DISTRICT_COORDS,
    SEVERITY_STYLE,
    ACCENT_ENV,
    ACCENT_ECO,
    ACCENT_NEUTRAL,
    DATA_SOURCES,
    GEMINI_MODEL,
    friendly_error,
)

# ----------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistan Agricultural Flood SDSS",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Executive SaaS UI CSS (refined light theme)
# ----------------------------------------------------------------------------
EXECUTIVE_SAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html {
    scrollbar-gutter: stable;
    overflow-x: hidden;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f7f9fc !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a;
}
#MainMenu, footer {visibility: hidden;}
.block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px;
}

/* ---------- Anti-jitter fixes ----------
   Prevents the whole page from nudging/shaking when the cursor moves over
   hover-animated cards/buttons or when the mouse wheel is scrolled. The
   cause was hover transforms + shadows repainting near the scrollbar edge
   and the scrollbar itself appearing/disappearing as content height changed. */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"] {
    overflow-anchor: none;
}
.kpi-tile, .ui-card, .step-card, .confidence-card, .hero-pill, .action-bar,
[data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button {
    will-change: transform;
    backface-visibility: hidden;
    transform: translateZ(0);
}
iframe { display: block; }

/* ---------- Header ---------- */
.executive-header {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e6eaf0;
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -1px rgba(15, 23, 42, 0.03);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.executive-header h1 {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}
.executive-header p {
    font-size: 0.86rem !important;
    color: #64748b !important;
    margin: 4px 0 0 0 !important;
}
.live-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;
    padding: 6px 12px; border-radius: 20px; font-size: 0.76rem; font-weight: 600;
    white-space: nowrap;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #10b981;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.18);
}

/* ---------- Generic card wrapper ---------- */
.ui-card {
    background: #ffffff;
    border: 1px solid #e6eaf0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(15,23,42,0.03), 0 2px 4px -2px rgba(15,23,42,0.02);
    margin-bottom: 18px;
}
.ui-card h4, .ui-card h5 { margin-top: 0 !important; }

/* ---------- Section labels ---------- */
.section-label {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin: 4px 0 10px 2px;
    display: flex; align-items: center; gap: 6px;
}

/* ---------- KPI tiles (custom, replaces bare st.metric look) ---------- */
.kpi-tile {
    background: #ffffff;
    border: 1px solid #e6eaf0;
    border-top: 3px solid var(--accent, #0ea5e9);
    border-radius: 14px;
    padding: 14px 16px 12px 16px;
    box-shadow: 0 2px 4px rgba(15,23,42,0.03);
    min-height: 104px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(15,23,42,0.06);
}
.kpi-icon { font-size: 1.15rem; opacity: 0.9; margin-bottom: 4px; }
.kpi-label {
    font-size: 0.72rem; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.kpi-value {
    font-size: 1.42rem; font-weight: 800; color: #0f172a;
    margin-top: 3px; letter-spacing: -0.01em;
}
.kpi-sub { font-size: 0.72rem; color: #94a3b8; margin-top: 3px; }

/* ---------- Insight callouts ---------- */
.insight-callout {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.86rem;
    color: #1e3a5f;
    margin: 6px 0 4px 0;
}
.insight-callout b { color: #1d4ed8; }
.insight-callout.warn { background:#fff7ed; border-color:#fed7aa; border-left-color:#ea580c; color:#7c2d12; }
.insight-callout.warn b { color:#c2410c; }
.insight-callout.good { background:#f0fdf4; border-color:#bbf7d0; border-left-color:#16a34a; color:#14532d; }
.insight-callout.good b { color:#15803d; }

/* ---------- Confidence / data-quality badges ---------- */
.confidence-card {
    display: flex; align-items: center; gap: 14px;
    background: #ffffff; border: 1px solid #e6eaf0; border-radius: 14px;
    padding: 14px 18px;
}
.confidence-pill {
    padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700;
    white-space: nowrap;
}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px; background-color: #eef1f6; padding: 6px;
    border-radius: 12px; border: 1px solid #e6eaf0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    height: 42px; border-radius: 9px; background-color: transparent;
    border: none !important; color: #64748b !important; font-weight: 500;
    font-size: 0.89rem; padding: 0 18px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(90deg, #ffffff, #ffffff) padding-box,
                linear-gradient(90deg, #2563eb, #16a34a) border-box !important;
    border: 1.5px solid transparent !important;
    color: #0f172a !important;
    font-weight: 700 !important; box-shadow: 0 2px 5px rgba(15,23,42,0.06) !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #fafbff 100%) !important;
    border-right: 1px solid #e6eaf0 !important;
}
[data-testid="stSidebar"] h3 {
    background: linear-gradient(90deg, #2563eb, #16a34a);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-weight: 800 !important;
}
.dss-brand {
    text-align: center; font-size: 0.78rem; color: #94a3b8;
    padding-top: 14px; border-top: 1px solid #e6eaf0; margin-top: 20px;
}
.sidebar-step {
    font-size: 0.72rem; font-weight: 800; color: #ffffff;
    text-transform: uppercase; letter-spacing: 0.05em; margin: 16px 0 8px 0;
    display: inline-block; padding: 4px 12px; border-radius: 8px;
    background: linear-gradient(90deg, #2563eb, #16a34a);
}

/* ---------- Colorful form controls (sidebar selects, sliders, inputs) ---------- */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1.5px solid #dbe3ee !important;
    background-color: #ffffff !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.12) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.16) !important;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] input[type="text"] {
    border-radius: 10px !important;
    border: 1.5px solid #dbe3ee !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background-color: #16a34a !important;
    box-shadow: 0 0 0 4px rgba(22,163,74,0.18) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #2563eb, #16a34a) !important;
}

/* ---------- Colorful buttons ---------- */
/* Primary action button — Run Satellite Analysis (solid gradient, glow) */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb 0%, #16a34a 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 4px 12px rgba(22,120,90,0.28) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(22,120,90,0.36) !important;
}
/* Secondary action button — Reset Filters (colored outline, not flat grey) */
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
    background: #fff7ed !important;
    border: 1.5px solid #fdba74 !important;
    color: #c2410c !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
    background: #ffedd5 !important;
    border-color: #f97316 !important;
    transform: translateY(-1px);
}
/* Back to Overview button (main panel) — teal outline, distinct from sidebar actions */
.actionbar-back-marker + div [data-testid="stButton"] button {
    background: #ecfeff !important;
    border: 1.5px solid #67e8f9 !important;
    color: #0e7490 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.actionbar-back-marker + div [data-testid="stButton"] button:hover {
    background: #cffafe !important;
    border-color: #22d3ee !important;
    transform: translateY(-1px);
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(90deg, #16a34a 0%, #0ea5e9 100%) !important;
    border: none !important; color: #ffffff !important; font-weight: 700 !important;
    border-radius: 12px !important; box-shadow: 0 4px 12px rgba(16,163,74,0.24) !important;
}

/* ---------- Severity banner ---------- */
.severity-badge {
    display: inline-block; padding: 7px 16px; border-radius: 20px;
    font-size: 0.84rem; font-weight: 700; margin-bottom: 14px;
}
.dq-warning {
    background: #fffbeb; border: 1px solid #fde68a; color: #b45309;
    padding: 10px 14px; border-radius: 10px; font-size: 0.84rem; margin-bottom: 16px;
}

/* ---------- Hero summary strip ---------- */
.hero-strip { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 4px; }
.hero-pill {
    background: #f8fafc; border: 1px solid #e6eaf0; border-radius: 12px;
    padding: 8px 14px; font-size: 0.8rem; color: #334155;
}
.hero-pill b { color: #0f172a; }

/* ---------- Comparison table ---------- */
.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.cmp-table th {
    text-align: left; padding: 10px 12px; background: #f8fafc;
    color: #64748b; font-weight: 700; font-size: 0.74rem;
    text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 1px solid #e6eaf0;
}
.cmp-table td {
    padding: 9px 12px; border-bottom: 1px solid #eef1f6; color: #334155;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-winner { font-weight: 700; color: #0f172a; background: #eff6ff; border-radius: 6px; }

/* ---------- Map legend ---------- */
.map-legend-row { display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#334155; margin-bottom:4px;}
.map-legend-swatch { width:12px; height:12px; border-radius:3px; display:inline-block; }

/* ---------- Breadcrumb / action bar ---------- */
.action-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: #ffffff; border: 1px solid #e6eaf0; border-radius: 14px;
    padding: 10px 18px; margin-bottom: 18px;
    box-shadow: 0 2px 4px rgba(15,23,42,0.02);
}
.breadcrumb { display:flex; align-items:center; gap:6px; font-size:0.86rem; color:#64748b; flex-wrap:wrap; }
.breadcrumb .crumb { color:#64748b; }
.breadcrumb .crumb.active { color:#0f172a; font-weight:700; }
.breadcrumb .sep { color:#cbd5e1; }

/* ---------- Landing / empty state ---------- */
.landing-hero {
    background: linear-gradient(135deg, #eff6ff 0%, #f7f9fc 60%, #ffffff 100%);
    border: 1px solid #dbeafe; border-radius: 20px;
    padding: 40px 40px 34px 40px; margin-bottom: 20px; text-align: center;
}
.landing-hero h2 {
    font-size: 1.55rem; font-weight: 800; color: #0f172a; margin: 10px 0 8px 0;
}
.landing-hero p { font-size: 0.94rem; color: #475569; max-width: 620px; margin: 0 auto; line-height:1.55; }
.landing-icon { font-size: 2.4rem; }

.step-row { display:flex; gap:16px; margin-top: 26px; flex-wrap: wrap; justify-content:center; }
.step-card {
    flex: 1 1 220px; max-width: 260px; background:#ffffff; border:1px solid #e6eaf0;
    border-radius:14px; padding:18px 16px; text-align:left;
    box-shadow: 0 2px 4px rgba(15,23,42,0.03);
}
.step-num {
    width:28px; height:28px; border-radius:50%; background:#2563eb; color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.85rem; margin-bottom:10px;
}
.step-card h5 { margin:0 0 4px 0; font-size:0.92rem; color:#0f172a; }
.step-card p { margin:0; font-size:0.8rem; color:#64748b; line-height:1.45; }

.skeleton-label { font-size:0.74rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#cbd5e1; margin: 22px 0 10px 2px; }
.skeleton-tile {
    background: #f1f5f9; border: 1px dashed #e2e8f0; border-radius: 14px;
    min-height: 90px; padding: 14px 16px;
}
.skeleton-bar { height:10px; border-radius:6px; background:#e2e8f0; margin-bottom:8px; }
.skeleton-bar.short { width:55%; }
.skeleton-bar.value { height:16px; width:70%; background:#dbe3ee; }

/* ---------- Responsive breakpoints (tablet / mobile) ---------- */
@media (max-width: 900px) {
    .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    .executive-header { flex-direction: column; align-items: flex-start; gap: 10px; }
    .landing-hero { padding: 26px 18px 22px 18px; }
    .step-row { flex-direction: column; }
    .step-card { max-width: 100%; }
    .action-bar { flex-wrap: wrap; }
    [data-testid="stTabs"] [data-baseweb="tab"] { padding: 0 10px; font-size: 0.8rem; }
    .kpi-value { font-size: 1.2rem; }
    .hero-strip { gap: 6px; }
    .hero-pill { font-size: 0.74rem; padding: 6px 10px; }
}
@media (max-width: 640px) {
    .kpi-tile { min-height: 84px; padding: 10px 12px; }
    .confidence-card { flex-wrap: wrap; }
}
</style>
"""
st.markdown(EXECUTIVE_SAAS_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached backend calls (UNCHANGED logic — presentation layer only)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Google Earth Engine...")
def load_gee() -> bool:
    return init_gee()


@st.cache_data(show_spinner=False, persist="disk")
def cached_parameters(district: str, start_date: str, end_date: str):
    """Cached wrapper around the GEE extraction pipeline.

    persist="disk" means results survive an app restart, so re-running the
    same district/date combo later (or after redeploying) doesn't re-spend
    GEE quota or API calls.

    IMPORTANT: this function must NOT touch any Streamlit UI element,
    directly or via a callback. st.cache_data records every Streamlit
    element created during a cache MISS and "replays" those exact calls on
    a later cache HIT - if a callback here updates a progress bar/placeholder
    that was created outside this function, replay fails with
    CacheReplayClosureError because that placeholder no longer exists at
    replay time. Progress feedback is handled purely with a plain
    st.spinner() around the call site instead (see the Overview tab).
    """
    return get_district_10_parameters(district, start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_ai_recommendations(cache_key: str, data: dict):
    """
    Cached wrapper around the LLM call - keyed on a stable string (district +
    date range + damage%) rather than the raw dict, so identical results don't
    trigger a fresh (paid) API call. Contains no Streamlit UI calls, so it's
    safe to cache (see the CacheReplayClosureError note above).
    """
    return generate_ai_recommendations(data)


def resolve_date_range(analysis_type: str, date_meta: dict) -> tuple:
    if analysis_type == "Daily":
        d = date_meta["date"]
        return d, d
    if analysis_type == "Weekly":
        start = pd.Timestamp(date_meta["week_of"])
        end = start + pd.Timedelta(days=6)
        return str(start.date()), str(end.date())
    if analysis_type == "Monthly":
        y, m = date_meta["year"], date_meta["month"]
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthEnd(1)
        return str(start.date()), str(end.date())
    if analysis_type == "Quarterly":
        y, q = date_meta["year"], date_meta["quarter"]
        q_num = int(q[1])
        start_month = (q_num - 1) * 3 + 1
        start = pd.Timestamp(year=y, month=start_month, day=1)
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        return str(start.date()), str(end.date())
    if analysis_type == "Custom Date Range":
        return date_meta["start_date"], date_meta["end_date"]
    y = date_meta["year"]
    return f"{y}-07-15", f"{y}-09-30"


gee_ready = load_gee()
if not gee_ready:
    st.error(
        "Could not connect to Google Earth Engine. Check GEE_PROJECT_ID and "
        "your service-account credentials, then reload the app."
    )
    st.stop()

# NOTE: PROVINCE_DISTRICT_MAP, DISTRICT_COORDS, and CURRENT_YEAR now live in
# config.py — imported at the top of this file, edit them there.

# ----------------------------------------------------------------------------
# Small presentation helpers
# ----------------------------------------------------------------------------
def fmt(value, suffix="", prefix="", decimals=None):
    if value is None:
        return "N/A"
    if decimals is not None:
        value = round(value, decimals)
    if isinstance(value, (int, float)):
        return f"{prefix}{value:,}{suffix}"
    return f"{prefix}{value}{suffix}"


def severity_hex(severity: str) -> str:
    return SEVERITY_STYLE.get(severity, SEVERITY_STYLE["Unknown"])["hex"]


def kpi_tile_html(icon: str, label: str, value: str, accent: str, sub: str = None) -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-tile" style="--accent:{accent};">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{sub_html}</div>'
    )


def render_kpi_row(columns, tiles):
    """tiles: list of (icon, label, value, accent, sub) matching len(columns)."""
    for col, (icon, label, value, accent, sub) in zip(columns, tiles):
        with col:
            st.markdown(kpi_tile_html(icon, label, value, accent, sub), unsafe_allow_html=True)


def interpret_zscore(z):
    """Plain-language read of the CHIRPS rainfall Z-score vs. the 2000-2020 baseline."""
    if z is None:
        return "No 20-year satellite baseline available for comparison.", "neutral"
    if z >= 2:
        return f"Rainfall is extremely above the 20-year norm ({z:+.2f}σ) — a primary flood driver.", "warn"
    if z >= 1:
        return f"Rainfall is above normal ({z:+.2f}σ) — likely contributed to inundation.", "warn"
    if z > -1:
        return f"Rainfall is within the normal historical range ({z:+.2f}σ).", "good"
    if z > -2:
        return f"Rainfall is below normal ({z:+.2f}σ) — flooding likely driven by other factors (upstream flow, drainage).", "neutral"
    return f"Rainfall is extremely below normal ({z:+.2f}σ) — investigate non-rainfall flood sources.", "neutral"


def confidence_from_method(method: str):
    """Maps the flood-detection method returned by gee_engine to a confidence tier."""
    if not method or method == "Unavailable":
        return "Low Confidence", "#fee2e2", "#b91c1c", "#fca5a5"
    if "Sentinel-1" in method:
        return "High Confidence", "#dcfce7", "#15803d", "#bbf7d0"
    if "MODIS" in method:
        return "Moderate Confidence", "#fef9c3", "#a16207", "#fef08a"
    return "Moderate Confidence", "#fef9c3", "#a16207", "#fef08a"


def insight_box(html: str, kind: str = ""):
    cls = f"insight-callout {kind}".strip()
    st.markdown(f'<div class="{cls}">💡 {html}</div>', unsafe_allow_html=True)


def _chart_to_png_bytes(fig) -> io.BytesIO:
    """Render a matplotlib figure to an in-memory PNG for embedding in the PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _pdf_commodity_bar_chart(commodity_breakdown: list) -> io.BytesIO:
    """Horizontal bar chart of financial loss by commodity, green/blue themed."""
    names = [r["Commodity"] for r in commodity_breakdown]
    values = [r["Financial Loss ($ USD)"] for r in commodity_breakdown]
    order = sorted(range(len(values)), key=lambda i: values[i])
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(names, values, color="#16a34a")
    ax.set_xlabel("Financial Loss ($ USD)", fontsize=9)
    ax.set_title("Financial Loss by Commodity", fontsize=11, fontweight="bold", color="#0f172a")
    ax.tick_params(labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" ${val:,.0f}",
                va="center", fontsize=7.5, color="#0f172a")
    fig.tight_layout()
    return _chart_to_png_bytes(fig)


def _pdf_commodity_pie_chart(commodity_breakdown: list) -> io.BytesIO:
    """Pie chart of each commodity's share of total financial loss, blue/green palette."""
    names = [r["Commodity"] for r in commodity_breakdown]
    values = [r["Financial Loss ($ USD)"] for r in commodity_breakdown]
    palette = ["#0ea5e9", "#16a34a", "#0284c7", "#22c55e", "#0369a1", "#15803d"]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    if sum(values) > 0:
        ax.pie(
            values, labels=names, autopct="%1.0f%%", startangle=90,
            colors=palette[: len(values)], textprops={"fontsize": 7},
        )
    ax.set_title("Loss Share by Commodity", fontsize=11, fontweight="bold", color="#0f172a")
    fig.tight_layout()
    return _chart_to_png_bytes(fig)


def _pdf_location_map(district_name: str, coords: dict) -> io.BytesIO:
    """
    Schematic location map: plots every tracked district as a small dot and
    highlights the report's district in red, with lat/lon gridlines. This is
    intentionally a lightweight coordinate plot (no tile-server dependency),
    since a real basemap screenshot would require a headless-Chrome/Selenium
    stack that most Streamlit deployment targets don't provide reliably.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name, (lat, lon) in coords.items():
        if name == "All Pakistan":
            continue
        if name == district_name:
            continue
        ax.scatter(lon, lat, s=22, color="#0ea5e9", alpha=0.55, zorder=2)
        ax.annotate(name, (lon, lat), fontsize=6, color="#334155", xytext=(3, 3), textcoords="offset points")

    if district_name in coords:
        lat, lon = coords[district_name]
        ax.scatter(lon, lat, s=130, color="#dc2626", edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(f"  {district_name}", (lon, lat), fontsize=8.5, fontweight="bold", color="#dc2626")

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.set_title(f"Relative Location - {district_name}", fontsize=11, fontweight="bold", color="#0f172a")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _chart_to_png_bytes(fig)


def create_pdf_report(data: dict, ai_text: str = None) -> bytes:
    """
    Builds the executive PDF report: parameter summary, visual analytics
    (bar chart, pie chart, location map - all rendered with matplotlib so
    the PDF generator has no browser/Chrome dependency), commodity table,
    and statistical validation metrics.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, f"Pakistan Flood Impact Assessment Report ({data['Year']})", ln=True, align="C")
    pdf.set_font("Arial", "I", 11)
    pdf.cell(190, 7, f"Region: {data['District']} | Source: Google Earth Engine Multi-Satellite Pipeline", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "1. Executive Environmental & Agronomic Parameters", ln=True)
    pdf.set_font("Arial", "", 10)

    params = [
        ("Total Cropland Area", fmt(data["P1_Total_Cropland_SqKm"], suffix=" Sq Km")),
        ("Inundated Cropland Area", fmt(data["P2_Flooded_Cropland_SqKm"], suffix=" Sq Km")),
        ("Crop Damage Proportion", fmt(data["P3_Crop_Damage_Percent"], suffix="%")),
        ("NDVI Vegetation Health Drop", fmt(data["P4_NDVI_Health_Drop"])),
        ("Monsoon Precipitation (CHIRPS)", fmt(data["P5_Rainfall_CHIRPS_mm"], suffix=" mm")),
        ("Soil Moisture Saturation Level", fmt(data["P6_Soil_Moisture_Sat_Percent"], suffix="%")),
        ("Average Elevation (DEM)", fmt(data["P7_Elevation_Avg_Meters"], suffix=" meters")),
        ("Aggregate Yield Loss", fmt(data["P8_Crop_Yield_Loss_Tons"], suffix=" Metric Tons")),
        ("Total Estimated Financial Damage", fmt(data["P9_Financial_Loss_USD"], prefix="$")),
        ("Composite Vulnerability Score", fmt(data["P10_Vulnerability_Score"], suffix=" / 100")),
    ]
    for label, val in params:
        pdf.cell(100, 6, str(label), border=1)
        pdf.cell(90, 6, str(val), border=1, ln=True)
    pdf.ln(6)

    # --- 2. Visual Analytics: bar chart + pie chart side by side -----------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "2. Visual Analytics", ln=True)
    commodities = data.get("Commodity_Breakdown", [])
    if commodities:
        bar_img = _pdf_commodity_bar_chart(commodities)
        pie_img = _pdf_commodity_pie_chart(commodities)
        y_before = pdf.get_y()
        pdf.image(bar_img, x=10, y=y_before, w=112)
        pdf.image(pie_img, x=124, y=y_before, w=76)
        pdf.set_y(y_before + 78)
    else:
        pdf.set_font("Arial", "", 10)
        pdf.cell(190, 6, "No commodity data available for this region/window.", ln=True)
    pdf.ln(2)

    # --- 3. Location Map ------------------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "3. Location Map", ln=True)
    map_img = _pdf_location_map(data["District"], DISTRICT_COORDS)
    pdf.image(map_img, x=45, w=110)
    pdf.ln(4)

    # --- 4. Commodity breakdown table -------------------------------------------------------
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "4. Crop & Export Fruit Disaggregated Loss", ln=True)
    pdf.set_font("Arial", "B", 9)
    pdf.cell(70, 6, "Commodity Name", border=1)
    pdf.cell(40, 6, "Loss (Metric Tons)", border=1, align="R")
    pdf.cell(40, 6, "Price ($/Ton)", border=1, align="R")
    pdf.cell(40, 6, "Financial Loss ($)", border=1, align="R", ln=True)
    pdf.set_font("Arial", "", 9)
    for row in commodities:
        pdf.cell(70, 6, str(row["Commodity"]), border=1)
        pdf.cell(40, 6, f"{row['Yield Loss (Metric Tons)']:,}", border=1, align="R")
        pdf.cell(40, 6, f"${row['Export/Market Price ($/Ton)']}", border=1, align="R")
        pdf.cell(40, 6, f"${row['Financial Loss ($ USD)']:,.2f}", border=1, align="R", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "5. Empirical Statistical Analysis & Validation Metrics", ln=True)
    pdf.set_font("Arial", "", 10)
    for stat_key, stat_val in data["Statistical_Metrics"].items():
        pdf.cell(100, 6, str(stat_key), border=1)
        pdf.cell(90, 6, str(stat_val), border=1, ln=True)

    if data.get("Data_Quality_Warnings"):
        pdf.ln(6)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(190, 7, "6. Data Quality Notes", ln=True)
        pdf.set_font("Arial", "", 9)
        for w in data["Data_Quality_Warnings"]:
            pdf.multi_cell(190, 5, f"- {w}")

    # --- 7. Data Sources & Methodology -------------------------------------------------------
    # Lets the reader trace every number back to an exact GEE collection ID,
    # instead of just trusting an unsourced "satellite data" claim.
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, "7. Data Sources & Methodology", ln=True)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(190, 5, "Every parameter above is traceable to one of the Google Earth Engine "
                           "collections below. The exact flood-detection method actually used for "
                           "this run is noted in the Statistical Metrics table (Flood Detection Method).")
    pdf.ln(2)
    for src in DATA_SOURCES:
        pdf.set_font("Arial", "B", 9.5)
        pdf.cell(190, 5.5, f"{src['name']}  -  GEE ID: {src['collection_id']}", ln=True)
        pdf.set_font("Arial", "", 8.5)
        pdf.cell(6, 5, "", border=0)
        pdf.cell(184, 5, f"Provider: {src['provider']}", ln=True)
        pdf.cell(6, 5, "", border=0)
        pdf.multi_cell(184, 5, f"Used for: {src['purpose']}")
        pdf.ln(1.5)

    # --- 8. AI-Generated Recommendations (only if the user generated one in-app) -------------
    if ai_text:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(190, 8, "8. AI-Generated Recommendations & Precautions", ln=True)
        pdf.set_font("Arial", "I", 8)
        pdf.multi_cell(190, 4.5, f"Generated by {GEMINI_MODEL} from the parameters in this report. "
                                  f"Review before acting on it - this is a decision-support aid, not a substitute "
                                  f"for on-ground assessment.")
        pdf.ln(1)
        pdf.set_font("Arial", "", 9.5)
        pdf.multi_cell(190, 5, ai_text)

    pdf.ln(8)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(190, 5, "Generated by Decision Support System - Climate Resilience Division Pakistan | Prepared by Ali Anus", align="C")
    return bytes(pdf.output())


# ----------------------------------------------------------------------------
# Executive Top Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="executive-header">
        <div>
            <h1>🌾 Agricultural Flood SDSS</h1>
            <p>Research-Grade Multi-Satellite Remote Sensing & Economic Analytics · 2010–2026</p>
        </div>
        <div class="live-chip"><span class="live-dot"></span> GEE Connected</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar Control Panel (logic unchanged, labels lightly restyled)
# ----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.markdown("### 🎛️ Analysis Controls")

    st.sidebar.markdown('<div class="sidebar-step">Step 1 · Region</div>', unsafe_allow_html=True)
    province = st.sidebar.selectbox(
        "Province:", ["All Provinces", "Sindh", "Punjab", "Khyber Pakhtunkhwa", "Balochistan"],
    )

    if province != "All Provinces":
        available_districts = PROVINCE_DISTRICT_MAP.get(province, [])
    else:
        available_districts = list(DISTRICT_COORDS.keys())

    selected_district = st.sidebar.selectbox("District / Territory:", available_districts)

    st.sidebar.markdown('<div class="sidebar-step">Step 2 · Temporal Window</div>', unsafe_allow_html=True)
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Yearly", "Monthly", "Quarterly", "Weekly", "Daily", "Custom Date Range"],
        index=0,
    )

    date_meta = {}
    if analysis_type == "Daily":
        picked = st.sidebar.date_input("Date:", value=pd.Timestamp("2022-08-15"))
        date_meta["date"] = str(picked)
    elif analysis_type == "Weekly":
        picked = st.sidebar.date_input("Target Week Date:", value=pd.Timestamp("2022-08-15"))
        date_meta["week_of"] = str(picked)
    elif analysis_type == "Monthly":
        c_y, c_m = st.sidebar.columns(2)
        date_meta["year"] = c_y.selectbox("Year:", list(range(2010, CURRENT_YEAR + 1)), index=12)
        date_meta["month"] = c_m.selectbox(
            "Month:", list(range(1, 13)), index=7,
            format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%b"),
        )
    elif analysis_type == "Quarterly":
        c_y, c_q = st.sidebar.columns(2)
        date_meta["year"] = c_y.selectbox("Year:", list(range(2010, CURRENT_YEAR + 1)), index=12)
        date_meta["quarter"] = c_q.selectbox("Quarter:", ["Q1", "Q2", "Q3", "Q4"], index=2)
    elif analysis_type == "Custom Date Range":
        c_s, c_e = st.sidebar.columns(2)
        date_meta["start_date"] = str(c_s.date_input("Start:", value=pd.Timestamp("2022-07-15")))
        date_meta["end_date"] = str(c_e.date_input("End:", value=pd.Timestamp("2022-09-30")))
    else:  # Yearly
        date_meta["year"] = st.sidebar.slider("Assessment Year:", min_value=2010, max_value=CURRENT_YEAR, value=2022, step=1)

    start_date, end_date = resolve_date_range(analysis_type, date_meta)
    st.sidebar.caption(f"📅 Range: **{start_date} → {end_date}**")

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("▶ Run Satellite Analysis", type="primary", width="stretch")
    reset_clicked = st.sidebar.button("🔄 Reset Filters", width="stretch")

    if reset_clicked:
        for key in ["params_data", "status", "selected_district", "start_date", "end_date", "trend_df", "trend_dist"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.markdown('<div class="dss-brand">Prepared by <b>Ali Anus</b></div>', unsafe_allow_html=True)

    return {
        "district": selected_district,
        "province": province,
        "analysis_type": analysis_type,
        "start_date": start_date,
        "end_date": end_date,
        "run_clicked": run_clicked,
    }


sidebar_state = render_sidebar()
selected_district = sidebar_state["district"]
start_date = sidebar_state["start_date"]
end_date = sidebar_state["end_date"]
run_btn = sidebar_state["run_clicked"]


# ----------------------------------------------------------------------------
# Persistent breadcrumb + Back-to-Overview action bar
# ----------------------------------------------------------------------------
def render_action_bar(has_data: bool, district: str = None, province: str = None, year: int = None):
    crumb_home = '<span class="crumb active">Home</span>' if not has_data else '<span class="crumb">Home</span>'
    crumbs = [crumb_home]
    if has_data:
        if province and province != "All Provinces":
            crumbs.append(f'<span class="crumb">{province}</span>')
        crumbs.append(f'<span class="crumb">{district}</span>')
        crumbs.append(f'<span class="crumb active">{year} Analysis</span>')

    crumb_html = f' <span class="sep">›</span> '.join(crumbs)

    bar_col1, bar_col2 = st.columns([5, 1])
    with bar_col1:
        st.markdown(f'<div class="action-bar" style="justify-content:flex-start;">'
                    f'<div class="breadcrumb">🧭 {crumb_html}</div></div>', unsafe_allow_html=True)
    with bar_col2:
        if has_data:
            st.markdown('<div class="actionbar-back-marker"></div>', unsafe_allow_html=True)
            if st.button("⬅ Back to Overview", width="stretch", key="back_to_overview_btn"):
                for key in ["params_data", "status", "selected_district", "start_date", "end_date",
                            "trend_df", "trend_dist", "cmp_dA", "cmp_dB", "cmp_status", "cmp_names"]:
                    st.session_state.pop(key, None)
                st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & Commodity Loss",
    "⚔️ District Comparison",
    "📈 Historical Trajectory",
    "📄 Diagnostics & PDF Export",
    "🤖 AI Insights",
])

# ----------------------------------------------------------------------------
# Trigger Analytics — now with a granular, step-by-step progress bar instead
# of a single opaque spinner (backend computation itself is unchanged).
# ----------------------------------------------------------------------------
if run_btn:
    with st.spinner(f"Extracting GEE satellite datasets for {selected_district} ({start_date} \u2192 {end_date})..."):
        data, status = cached_parameters(selected_district, start_date, end_date)
    st.session_state["params_data"] = data
    st.session_state["status"] = status
    st.session_state["selected_district"] = selected_district
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date

data = st.session_state.get("params_data")
status = st.session_state.get("status")
curr_dist = st.session_state.get("selected_district", selected_district)
curr_start = st.session_state.get("start_date", start_date)
curr_end = st.session_state.get("end_date", end_date)
curr_year = int(curr_start[:4])

render_action_bar(
    has_data=bool(data),
    district=curr_dist,
    province=sidebar_state["province"],
    year=curr_year,
)

if data is None and status:
    st.error(friendly_error(status))

if data:
    severity = data.get("Flood_Severity", "Unknown")
    sev = SEVERITY_STYLE.get(severity, SEVERITY_STYLE["Unknown"])

    st.markdown(
        f'<div class="severity-badge" style="{sev["badge"]}">'
        f'🌊 Auto-Classified Severity: <b>{severity}</b> '
        f'({fmt(data["P3_Crop_Damage_Percent"], suffix="%")} Cropland Impact)</div>',
        unsafe_allow_html=True,
    )

    # Hero summary strip — quick-glance headline numbers before the deep dive
    method = data["Statistical_Metrics"].get("Flood Detection Method", "Unavailable")
    conf_label, conf_bg, conf_fg, conf_border = confidence_from_method(method)
    st.markdown(
        f"""
        <div class="hero-strip">
            <div class="hero-pill">📅 <b>{curr_year}</b> · {curr_start} → {curr_end}</div>
            <div class="hero-pill">💰 Est. Loss <b>{fmt(data["P9_Financial_Loss_USD"], prefix="$")}</b></div>
            <div class="hero-pill">⚠️ Vulnerability <b>{fmt(data["P10_Vulnerability_Score"], suffix="/100")}</b></div>
            <div class="hero-pill" style="background:{conf_bg}; border-color:{conf_border}; color:{conf_fg};">
                🛰️ {method} · <b>{conf_label}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    if data.get("Data_Quality_Warnings"):
        friendly_warnings = [friendly_error(w) for w in data["Data_Quality_Warnings"]]
        st.markdown(
            f'<div class="dq-warning">⚠️ Satellite Notes: '
            f'{"; ".join(friendly_warnings)}</div>',
            unsafe_allow_html=True,
        )

    # ==========================================================
# TAB 1: DEEP DIVE & COMMODITY BREAKDOWN
# ==========================================================
with tab1:
    if data:
        st.markdown(f"#### 📌 Core Satellite & Agronomic Metrics — {curr_dist} ({curr_year})")

        # ---- Group 1: Environmental / Physical indicators ----
        st.markdown(
            '<div class="section-label">🌱 Environmental &amp; Physical Indicators</div>',
            unsafe_allow_html=True,
        )
        env_cols = st.columns(5)
        render_kpi_row(env_cols, [
            ("🌾", "Total Cropland", fmt(data["P1_Total_Cropland_SqKm"], suffix=" km²"), ACCENT_ENV, None),
            ("🌊", "Flooded Cropland", fmt(data["P2_Flooded_Cropland_SqKm"], suffix=" km²"), ACCENT_ENV, None),
            ("🍃", "NDVI Health Drop", fmt(data["P4_NDVI_Health_Drop"]), ACCENT_ENV, "vs. pre-event baseline"),
            ("🌧️", "Rainfall (CHIRPS)", fmt(data["P5_Rainfall_CHIRPS_mm"], suffix=" mm"), ACCENT_ENV, None),
            ("⛰️", "Avg Elevation", fmt(data["P7_Elevation_Avg_Meters"], suffix=" m"), ACCENT_ENV, "SRTM DEM"),
        ])
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        env_cols2 = st.columns(5)
        render_kpi_row(env_cols2[:1], [
            ("💧", "Soil Moisture Sat.", fmt(data["P6_Soil_Moisture_Sat_Percent"], suffix="%"), ACCENT_ENV, None),
        ])

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ---- Group 2: Economic impact indicators ----
        st.markdown(
            '<div class="section-label">💰 Economic Impact Indicators</div>',
            unsafe_allow_html=True,
        )
        eco_cols = st.columns(3)
        render_kpi_row(eco_cols, [
            ("📉", "Crop Damage Share", fmt(data["P3_Crop_Damage_Percent"], suffix="%"), severity_hex(severity), "cropland lost to flooding"),
            ("🌾", "Crop Yield Loss", fmt(data["P8_Crop_Yield_Loss_Tons"], suffix=" tons"), ACCENT_ECO, None),
            ("💵", "Financial Loss", fmt(data["P9_Financial_Loss_USD"], prefix="$"), ACCENT_ECO, "farm-gate estimate"),
        ])

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ---- Group 3: Risk & confidence indicators ----
        st.markdown(
            '<div class="section-label">⚠️ Risk &amp; Statistical Confidence</div>',
            unsafe_allow_html=True,
        )
        z = data["Statistical_Metrics"].get("Monsoon Rain Z-Score")
        z_val = z if isinstance(z, (int, float)) else None
        variance_val = data["Statistical_Metrics"].get("Spatial Flood Variance (Pixel Reducer)")

        risk_cols = st.columns(3)
        render_kpi_row(risk_cols, [
            ("🎯", "Vulnerability Score", fmt(data["P10_Vulnerability_Score"], suffix="/100"), severity_hex(severity), "composite risk index"),
            ("📊", "Rainfall Z-Score", f"{z_val:+.2f}σ" if z_val is not None else "N/A", ACCENT_NEUTRAL, "vs. 2000–2020 norm"),
            ("🧮", "Spatial Flood Variance", fmt(variance_val), ACCENT_NEUTRAL, "pixel-level reducer"),
        ])

        z_text, z_kind = interpret_zscore(z_val)
        insight_box(z_text, z_kind)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ---- Confidence card for detection method ----
        st.markdown(
            f"""
            <div class="confidence-card">
                <div style="font-size:1.6rem;">🛰️</div>
                <div>
                    <div style="font-size:0.78rem; color:#64748b; font-weight:600;">Flood Detection Method</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#0f172a;">{method}</div>
                </div>
                <div style="margin-left:auto;">
                    <span class="confidence-pill" style="background:{conf_bg}; color:{conf_fg}; border:1px solid {conf_border};">{conf_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Disaggregated commodity table ----
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("#### 🌾 Commodity &amp; Export Fruit Disaggregated Loss")
        df_comm = pd.DataFrame(data["Commodity_Breakdown"])
        st.dataframe(df_comm, width="stretch", hide_index=True)

        if not df_comm.empty:
            top_row = df_comm.loc[df_comm["Financial Loss ($ USD)"].idxmax()]
            total_fin = df_comm["Financial Loss ($ USD)"].sum()
            share = (top_row["Financial Loss ($ USD)"] / total_fin * 100) if total_fin else 0
            insight_box(
                f"<b>{top_row['Commodity']}</b> is the single largest driver of loss, accounting for "
                f"<b>{share:.1f}%</b> of the ${total_fin:,.0f} total financial impact estimated for {curr_dist}."
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Graphical analytics (side-by-side) ----
        col_hbar, col_bar = st.columns(2)

        with col_hbar:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            fig_hbar = px.bar(
                df_comm,
                x="Financial Loss ($ USD)",
                y="Commodity",
                orientation="h",
                title=f"Financial Loss Ranking ($ USD) — {curr_dist}",
                text_auto=".2s",
                color="Financial Loss ($ USD)",
                color_continuous_scale=["#bbf7d0", "#16a34a", "#14532d"],
            )
            fig_hbar.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Loss ($ USD)", yaxis_title="",
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#334155"},
                title_font={"size": 14, "color": "#0f172a"},
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig_hbar, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_bar:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            fig_bar = px.bar(
                df_comm,
                x="Commodity",
                y="Yield Loss (Metric Tons)",
                color="Commodity",
                title=f"Volume Loss (Metric Tons) — {curr_dist}",
                text_auto=".2s",
                color_discrete_sequence=["#0ea5e9", "#06b6d4", "#22c55e", "#84cc16", "#eab308", "#f97316"],
            )
            fig_bar.update_layout(
                xaxis_title="", yaxis_title="Metric Tons", showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#334155"},
                title_font={"size": 14, "color": "#0f172a"},
                margin=dict(l=10, r=10, t=45, b=10),
            )
            st.plotly_chart(fig_bar, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Geospatial Map Card ----
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        map_header_col, legend_col = st.columns([3, 1])
        with map_header_col:
            st.markdown(f"#### 🗺️ Geospatial Inundation Map ({curr_dist})")
            st.caption("Auto-focused high-resolution spatial extent of agricultural flooding.")
        with legend_col:
            st.markdown(
                """
                <div style="padding-top:6px;">
                    <div class="map-legend-row"><span class="map-legend-swatch" style="background:#2e7d32;"></span> Cropland Mask</div>
                    <div class="map-legend-row"><span class="map-legend-swatch" style="background:#0288d1;"></span> Inundated Zone</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        coords = DISTRICT_COORDS.get(curr_dist, [30.3753, 69.3451])
        zoom_lvl = 6 if curr_dist == "All Pakistan" else 10
        m = folium.Map(location=coords, zoom_start=zoom_lvl, tiles="CartoDB positron")

        if data.get("Cropland_Tile"):
            folium.TileLayer(
                tiles=data["Cropland_Tile"], attr="ESA WorldCover 10m",
                name="🟢 Cropland Mask", overlay=True, opacity=0.6,
            ).add_to(m)
        if data.get("Flood_Tile"):
            folium.TileLayer(
                tiles=data["Flood_Tile"], attr=method,
                name=f"🔵 Inundated Water Zone ({method})", overlay=True, opacity=0.8,
            ).add_to(m)

        damage = data["P3_Crop_Damage_Percent"] or 0
        marker_color = "red" if damage > 25 else "orange" if damage > 10 else "green"
        folium.Marker(
            location=coords,
            popup=folium.Popup(
                f"<div style='font-family:Inter,sans-serif;'>"
                f"<b>{curr_dist} ({curr_year})</b><br>"
                f"Severity: <b>{severity}</b><br>"
                f"Crop Damage: <b>{fmt(data['P3_Crop_Damage_Percent'], suffix='%')}</b><br>"
                f"Flooded Area: <b>{fmt(data['P2_Flooded_Cropland_SqKm'], suffix=' km²')}</b><br>"
                f"Financial Loss: <b>{fmt(data['P9_Financial_Loss_USD'], prefix='$')}</b>"
                f"</div>",
                max_width=260,
            ),
            icon=folium.Icon(color=marker_color),
        ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width="100%", height=500)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================================
    else:
        # ------------------------------------------------------------------
        # Landing state — shown before the first analysis is run.
        # Designed so the page never looks "empty": hero intro, a 3-step
        # guide, and greyed-out skeleton KPI/chart placeholders that mirror
        # the exact layout the real dashboard will render into.
        # ------------------------------------------------------------------
        st.markdown(
            """
            <div class="landing-hero">
                <div class="landing-icon">🛰️🌾</div>
                <h2>Satellite-Powered Flood &amp; Crop Impact Intelligence</h2>
                <p>
                    Select a province and district, choose a time window, then run the analysis
                    to generate 10 satellite-derived impact parameters, commodity-level financial
                    loss estimates, and an exportable executive PDF report.
                </p>
                <div class="step-row">
                    <div class="step-card">
                        <div class="step-num">1</div>
                        <h5>Choose a region</h5>
                        <p>Pick a province and district from the sidebar — or run "All Pakistan" for a national view.</p>
                    </div>
                    <div class="step-card">
                        <div class="step-num">2</div>
                        <h5>Set the time window</h5>
                        <p>Yearly, quarterly, monthly, or a fully custom date range — matched to your monsoon season of interest.</p>
                    </div>
                    <div class="step-card">
                        <div class="step-num">3</div>
                        <h5>Run &amp; export</h5>
                        <p>Click "Run Satellite Analysis" to pull live GEE data, then export a branded PDF report in one click.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="skeleton-label">🌱 Environmental &amp; Physical Indicators — preview</div>', unsafe_allow_html=True)
        sk_cols = st.columns(5)
        for c in sk_cols:
            with c:
                st.markdown(
                    '<div class="skeleton-tile">'
                    '<div class="skeleton-bar short"></div>'
                    '<div class="skeleton-bar value"></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="skeleton-label">💰 Economic Impact Indicators — preview</div>', unsafe_allow_html=True)
        sk_cols2 = st.columns(3)
        for c in sk_cols2:
            with c:
                st.markdown(
                    '<div class="skeleton-tile">'
                    '<div class="skeleton-bar short"></div>'
                    '<div class="skeleton-bar value"></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        insight_box(
            "👈 Configure your region and time window in the sidebar, then click "
            "<b>\"▶ Run Satellite Analysis\"</b> to replace this preview with live results.",
        )
# ==========================================================
# TAB 2: DISTRICT COMPARISON MODE
# ==========================================================
with tab2:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown("#### ⚔️ Side-by-Side District Comparison")
    st.caption("The larger raw value in each row is highlighted for quick visual scanning.")
    all_dists = list(DISTRICT_COORDS.keys())
    c_comp1, c_comp2 = st.columns(2)
    dist_A = c_comp1.selectbox("Select District A:", all_dists, index=0)
    dist_B = c_comp2.selectbox("Select District B:", all_dists, index=1)

    if st.button("🔎 Run Comparative Extraction"):
        with st.spinner("Extracting comparative satellite parameters..."):
            dA, statusA = cached_parameters(dist_A, curr_start, curr_end)
            dB, statusB = cached_parameters(dist_B, curr_start, curr_end)
        st.session_state["cmp_dA"] = dA
        st.session_state["cmp_dB"] = dB
        st.session_state["cmp_status"] = (statusA, statusB)
        st.session_state["cmp_names"] = (dist_A, dist_B)

    dA = st.session_state.get("cmp_dA")
    dB = st.session_state.get("cmp_dB")

    if dA and dB:
        name_A, name_B = st.session_state.get("cmp_names", (dist_A, dist_B))
        rows = [
            ("1. Total Cropland (km²)", "P1_Total_Cropland_SqKm", "", ""),
            ("2. Flooded Cropland (km²)", "P2_Flooded_Cropland_SqKm", "", ""),
            ("3. Crop Damage Share (%)", "P3_Crop_Damage_Percent", "", "%"),
            ("4. NDVI Crop Health Drop", "P4_NDVI_Health_Drop", "", ""),
            ("5. Monsoon Rainfall (mm)", "P5_Rainfall_CHIRPS_mm", "", " mm"),
            ("6. Soil Moisture Sat. (%)", "P6_Soil_Moisture_Sat_Percent", "", "%"),
            ("7. Avg Elevation (m)", "P7_Elevation_Avg_Meters", "", " m"),
            ("8. Yield Loss (Tons)", "P8_Crop_Yield_Loss_Tons", "", " t"),
            ("9. Financial Loss ($ USD)", "P9_Financial_Loss_USD", "$", ""),
            ("10. Vulnerability Score", "P10_Vulnerability_Score", "", ""),
        ]

        table_rows_html = ""
        for label, key, prefix, suffix in rows:
            valA, valB = dA.get(key), dB.get(key)
            a_str = fmt(valA, prefix=prefix, suffix=suffix)
            b_str = fmt(valB, prefix=prefix, suffix=suffix)
            a_cls = b_cls = ""
            if isinstance(valA, (int, float)) and isinstance(valB, (int, float)):
                if valA > valB:
                    a_cls = "cmp-winner"
                elif valB > valA:
                    b_cls = "cmp-winner"
            table_rows_html += (
                f"<tr><td>{label}</td>"
                f"<td class='{a_cls}'>{a_str}</td>"
                f"<td class='{b_cls}'>{b_str}</td></tr>"
            )

        st.markdown(
            f"""
            <table class="cmp-table">
                <thead><tr><th>Parameter</th><th>{name_A}</th><th>{name_B}</th></tr></thead>
                <tbody>{table_rows_html}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

        higher_loss = name_A if (dA.get("P9_Financial_Loss_USD") or 0) > (dB.get("P9_Financial_Loss_USD") or 0) else name_B
        insight_box(
            f"<b>{higher_loss}</b> carries the greater estimated financial exposure between the two "
            f"districts for this time window — prioritize relief and monitoring resources accordingly."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Graphical comparison (bar + pie) ----
        col_cmp_bar, col_cmp_pie = st.columns(2)

        with col_cmp_bar:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            # Only comparable 0-100 scale metrics go in one grouped bar chart;
            # $ / km² / ton values are left out here since their scale would
            # dwarf the percentage metrics and make the chart unreadable.
            bar_metrics = [
                ("Crop Damage (%)", "P3_Crop_Damage_Percent"),
                ("Soil Moisture Sat. (%)", "P6_Soil_Moisture_Sat_Percent"),
                ("Vulnerability Score", "P10_Vulnerability_Score"),
            ]
            df_bar_cmp = pd.DataFrame([
                {"Metric": label, "District": name_A, "Value": dA.get(key)} for label, key in bar_metrics
            ] + [
                {"Metric": label, "District": name_B, "Value": dB.get(key)} for label, key in bar_metrics
            ])
            fig_cmp_bar = px.bar(
                df_bar_cmp, x="Metric", y="Value", color="District", barmode="group",
                title="Comparable Impact Metrics (0-100 scale)",
                color_discrete_sequence=["#0ea5e9", "#16a34a"],
            )
            fig_cmp_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#334155"},
                title_font={"size": 14, "color": "#0f172a"},
                margin=dict(l=10, r=10, t=45, b=10), legend_title="",
            )
            st.plotly_chart(fig_cmp_bar, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_cmp_pie:
            st.markdown('<div class="ui-card">', unsafe_allow_html=True)
            loss_A = dA.get("P9_Financial_Loss_USD") or 0
            loss_B = dB.get("P9_Financial_Loss_USD") or 0
            if (loss_A + loss_B) > 0:
                fig_cmp_pie = px.pie(
                    names=[name_A, name_B], values=[loss_A, loss_B],
                    title="Financial Loss Share ($ USD)", hole=0.45,
                    color_discrete_sequence=["#0ea5e9", "#16a34a"],
                )
                fig_cmp_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "color": "#334155"},
                    title_font={"size": 14, "color": "#0f172a"},
                    margin=dict(l=10, r=10, t=45, b=10),
                )
                fig_cmp_pie.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_cmp_pie, width="stretch")
            else:
                st.info("No financial loss recorded for either district in this window.")
            st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.get("cmp_status"):
        statusA, statusB = st.session_state["cmp_status"]
        st.error(friendly_error(statusA if not dA else statusB))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# TAB 3: HISTORICAL TREND & RECOVERY
# ==========================================================
with tab3:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📈 Historical Trajectory: {curr_dist}")
    yr_range = st.slider("Year range to analyze:", 2015, CURRENT_YEAR, (2020, 2025))
    run_trend = st.button("📊 Build Trend Chart")

    if run_trend:
        years_to_fetch = list(range(yr_range[0], yr_range[1] + 1))
        progress = st.progress(0.0, text="Fetching yearly satellite data...")
        rows = []
        for i, y in enumerate(years_to_fetch):
            d, _ = cached_parameters(curr_dist, f"{y}-07-15", f"{y}-09-30")
            if d:
                rows.append({
                    "Year": y,
                    "Crop Damage Share (%)": d["P3_Crop_Damage_Percent"],
                    "NDVI Health Drop": d["P4_NDVI_Health_Drop"],
                    "Severity": d["Flood_Severity"],
                })
            progress.progress((i + 1) / len(years_to_fetch), text=f"Fetched {y}...")
        progress.empty()

        if rows:
            df_trend = pd.DataFrame(rows)
            st.session_state["trend_df"] = df_trend
            st.session_state["trend_dist"] = curr_dist

    df_trend = st.session_state.get("trend_df")
    if df_trend is not None and st.session_state.get("trend_dist") == curr_dist:
        marker_colors = [severity_hex(s) for s in df_trend["Severity"]]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_trend["Year"], y=df_trend["Crop Damage Share (%)"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=11, color=marker_colors, line=dict(width=1, color="#ffffff")),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
            name="Crop Damage Share (%)",
        ))
        fig_trend.update_layout(
            title=f"Crop Damage Share Over Time ({curr_dist})",
            xaxis_title="Year", yaxis_title="Crop Damage Share (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter", "color": "#334155"},
            title_font={"size": 14, "color": "#0f172a"},
            margin=dict(l=10, r=10, t=45, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, width="stretch")

        worst = df_trend.loc[df_trend["Crop Damage Share (%)"].idxmax()]
        latest = df_trend.iloc[-1]
        if worst["Year"] != latest["Year"]:
            recovered_pts = round(worst["Crop Damage Share (%)"] - latest["Crop Damage Share (%)"], 2)
            kind = "good" if recovered_pts > 0 else "warn"
            verb = "fell" if recovered_pts > 0 else "rose"
            insight_box(
                f"Peak damage was <b>{worst['Crop Damage Share (%)']}%</b> in <b>{int(worst['Year'])}</b>. "
                f"By <b>{int(latest['Year'])}</b> it stood at <b>{latest['Crop Damage Share (%)']}%</b> "
                f"— crop damage share {verb} <b>{abs(recovered_pts):.2f} percentage points</b> since the peak.",
                kind,
            )
        else:
            insight_box(
                f"<b>{int(latest['Year'])}</b> is the most severe year on record in this window, "
                f"with <b>{latest['Crop Damage Share (%)']}%</b> cropland damage.",
                "warn",
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# TAB 4: DIAGNOSTICS & PDF REPORT
# ==========================================================
with tab4:
    if data:
        # ---- Section 1: Empirical Validation (full width, no longer squeezed into a column) ----
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("#### \U0001F4CA Empirical Validation Index")
        df_stats = pd.DataFrame(
            list(data["Statistical_Metrics"].items()),
            columns=["Statistical Metric", "Observed Value"],
        )
        # Statistical_Metrics mixes floats (e.g. Z-scores) and strings (e.g.
        # "N/A", "-3.61 Sigma") in the same column. Force a uniform string
        # dtype so PyArrow's Arrow-conversion for st.table doesn't try to
        # infer a numeric column and choke on the string entries.
        df_stats["Observed Value"] = df_stats["Observed Value"].astype(str)
        st.table(df_stats)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Section 2: Data confidence (its own full-width card, not crammed beside stats) ----
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="confidence-card">
                <div style="font-size:1.6rem;">\U0001F6F0\uFE0F</div>
                <div>
                    <div style="font-size:0.78rem; color:#64748b; font-weight:600;">Overall Data Confidence</div>
                    <div style="font-size:1.0rem; font-weight:700; color:#0f172a;">
                        {"No warnings raised during extraction" if not data.get("Data_Quality_Warnings") else f"{len(data['Data_Quality_Warnings'])} data-quality note(s) — see banner above"}
                    </div>
                </div>
                <div style="margin-left:auto;">
                    <span class="confidence-pill" style="background:{conf_bg}; color:{conf_fg}; border:1px solid {conf_border};">{conf_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Section 3: PDF Export - placed LAST, as the final action on this tab -------------
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("#### \U0001F4C4 Export PDF Executive Summary")
        st.write(
            "The report includes the executive parameter summary, a financial-loss bar chart, "
            "a commodity loss-share pie chart, a district location map, the full commodity "
            "breakdown table, and the statistical validation metrics above."
        )

        cache_key = f"{curr_dist}_{curr_start}_{curr_end}"
        gen_col, dl_col = st.columns([1, 1.4])
        expected_ai_key = f"{curr_dist}_{curr_start}_{curr_end}_{data.get('P3_Crop_Damage_Percent')}"
        if gen_col.button("\U0001F5A8\uFE0F Generate PDF Report", width="stretch"):
            with st.spinner("Rendering charts and building the PDF..."):
                ai_text_for_pdf = (
                    st.session_state.get("ai_text")
                    if st.session_state.get("ai_cache_key") == expected_ai_key
                    else None
                )
                st.session_state["pdf_bytes"] = create_pdf_report(data, ai_text=ai_text_for_pdf)
                st.session_state["pdf_cache_key"] = cache_key

        pdf_ready = st.session_state.get("pdf_bytes") and st.session_state.get("pdf_cache_key") == cache_key
        with dl_col:
            if pdf_ready:
                st.download_button(
                    label=f"\U0001F4E5 Download Official PDF Report ({curr_dist})",
                    data=st.session_state["pdf_bytes"],
                    file_name=f"{curr_dist}_Flood_Assessment_{curr_year}.pdf",
                    mime="application/pdf",
                    type="primary",
                    width="stretch",
                )
            else:
                st.info("Click **Generate PDF Report** to build the file for this district/date range.")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Run the analysis in the **Overview** tab first to unlock diagnostics and the PDF export for that district/year.")

# ==========================================================
# TAB 5: AI INSIGHTS (LLM-generated recommendations & precautions)
# ==========================================================
with tab5:
    if data:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("#### \U0001F916 AI-Generated Recommendations & Precautions")
        st.write(
            "Sends the computed satellite parameters (not raw imagery) to an LLM to generate "
            "a plain-language situation summary, immediate precautions, recovery steps, and "
            "longer-term prevention measures for this district and time window."
        )

        if not ai_is_configured():
            st.markdown(
                '<div class="dq-warning">\u26A0\uFE0F AI Insights is not configured yet. Add '
                '<code>GEMINI_API_KEY</code> (and optionally <code>GEMINI_MODEL</code>) '
                'to your <code>.env</code> file to enable this tab — get a free key (no credit '
                'card needed) at <a href="https://aistudio.google.com/apikey" target="_blank">'
                'aistudio.google.com/apikey</a>. The rest of the dashboard '
                'works fully without it.</div>',
                unsafe_allow_html=True,
            )
        else:
            ai_cache_key = f"{curr_dist}_{curr_start}_{curr_end}_{data.get('P3_Crop_Damage_Percent')}"
            if st.button("\u2728 Generate AI Recommendations", type="primary"):
                with st.spinner("Consulting the AI advisory model..."):
                    ai_text, ai_status = cached_ai_recommendations(ai_cache_key, data)
                st.session_state["ai_text"] = ai_text
                st.session_state["ai_status"] = ai_status
                st.session_state["ai_cache_key"] = ai_cache_key

            ai_ready = (
                st.session_state.get("ai_text")
                and st.session_state.get("ai_cache_key") == ai_cache_key
            )
            if ai_ready:
                st.markdown(
                    f'<div class="insight-callout good">{st.session_state["ai_text"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("This text will be included in the PDF report if you generate it before leaving this district/window.")
            elif st.session_state.get("ai_status") and st.session_state.get("ai_cache_key") == ai_cache_key:
                st.error(st.session_state["ai_status"])
            else:
                st.info("Click **Generate AI Recommendations** to run this for the current district/window.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Run the analysis in the **Overview** tab first to unlock AI Insights for that district/year.")