"""Fraud Risk Intelligence — Streamlit decision-support dashboard (v5).

Run from the repository root:
    python -m streamlit run dashboard/app.py

Read-only over verified reports. No model training, no holdout-based tuning,
no estimated or demo numbers. Every derived metric below is computed directly
from the existing report files.

IMPORTANT: also copy `.streamlit/config.toml` (delivered next to this file) into
the repository root so Streamlit uses the light theme regardless of the browser.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.data_access import (DataIssue, discover_runs, get_run, get_optimization,
                                   read_csv, read_json, scenario_economics,
                                   verify_business_consistency, fnum)

# ───────────────────────────── design tokens ─────────────────────────────
NAVY, INK, TEAL, RED, GOLD = '#12263d', '#1b3048', '#0a8a90', '#d2535b', '#c98a2b'
MUTED, GRID = '#7d8fa5', '#e8eef4'
POLICY_LABEL = {'fraud_score': 'Fraud score', 'score_times_amount': 'Score × amount'}
POLICY_COLOR = {'fraud_score': TEAL, 'score_times_amount': GOLD}

st.set_page_config(page_title='Fraud Risk Intelligence | Analytics Lab', page_icon='🛡️',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown("""<style>
/* v5 presentation system — light content, dark navigation, theme-independent. */
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Segoe UI', Inter, system-ui, Arial, sans-serif;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #f3f6fa !important; color: #1b3048 !important; }
[data-testid="stHeader"] { background: #f3f6fa !important; box-shadow: none; }
.block-container { max-width: 1480px; padding: 2.6rem 2.6rem 3rem; }
@media (max-width: 900px) { .block-container { padding: 2.2rem 1rem 2.5rem !important; } }

/* ---- base typography ---- */
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4 {
    color: #12263d !important; letter-spacing: -.018em; }
[data-testid="stMain"] h1 { font-size: 2.15rem !important; font-weight: 800 !important; margin-bottom: .1rem; padding-top: .2rem; }
[data-testid="stMain"] h4 { font-size: 1.05rem !important; }
[data-testid="stMain"] p, [data-testid="stMain"] label, [data-testid="stMain"] li,
[data-testid="stMain"] [data-testid="stWidgetLabel"], [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] { color: #24405b; }
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
    color: #55697e !important; font-size: .84rem; }

/* ---- tabs ---- */
[data-testid="stMain"] .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #d3dfe9; margin-bottom: .6rem; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"] { min-height: 46px; padding: 0 18px; border-radius: 10px 10px 0 0;
    background: transparent !important; font-weight: 650; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"], [data-testid="stMain"] .stTabs [data-baseweb="tab"] * { color: #3a536c !important; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"]:hover { background: #e6eef5 !important; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #fff !important; box-shadow: inset 0 3px 0 #0a8a90; }
[data-testid="stMain"] .stTabs [data-baseweb="tab"][aria-selected="true"],
[data-testid="stMain"] .stTabs [data-baseweb="tab"][aria-selected="true"] * { color: #07767c !important; font-weight: 800; }
[data-testid="stMain"] .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

/* ---- controls forced to light ---- */
[data-testid="stMain"] [data-baseweb="select"] > div, [data-testid="stMain"] [data-baseweb="input"] > div,
[data-testid="stMain"] [data-baseweb="base-input"] { background: #fff !important; border-color: #c7d6e3 !important; color: #1b3048 !important; }
[data-testid="stMain"] [data-baseweb="input"] input, [data-testid="stMain"] [data-baseweb="select"] * {
    color: #1b3048 !important; -webkit-text-fill-color: #1b3048; }
[data-testid="stMain"] [data-testid="stNumberInput"] button { background: #eef3f8 !important; color: #1b3048 !important; border-color: #c7d6e3 !important; }
[data-testid="stMain"] [data-testid="stNumberInput"] button * { color: #1b3048 !important; }
[data-testid="stMain"] [data-testid="stSlider"] [role="slider"],
[data-testid="stMain"] [data-testid="stSelectSlider"] [role="slider"] { background: #0a8a90 !important; border-color: #0a8a90 !important; }
[data-testid="stMain"] [data-testid="stSliderThumbValue"], [data-testid="stMain"] [data-testid="stSliderThumbValue"] *,
[data-testid="stMain"] [data-testid="stTickBar"] *, [data-testid="stMain"] [data-testid="stTickBarMin"],
[data-testid="stMain"] [data-testid="stTickBarMax"] { color: #07767c !important; font-weight: 700; }
[data-testid="stMain"] [data-testid="stRadio"] label * { color: #1e3952 !important; }
[data-testid="stMain"] [data-testid="stDownloadButton"] button, [data-testid="stMain"] [data-testid="stButton"] button {
    background: #fff !important; border: 1px solid #b6cadb !important; border-radius: 10px; font-weight: 650; padding: .35rem 1rem; }
[data-testid="stMain"] [data-testid="stDownloadButton"] button *, [data-testid="stMain"] [data-testid="stButton"] button * { color: #14405a !important; }
[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover { border-color: #0a8a90 !important; background: #eaf6f6 !important; }
[data-testid="stMain"] [data-testid="stExpander"] { background: #fff !important; border: 1px solid #d3dfe9 !important; border-radius: 12px; overflow: hidden; }
[data-testid="stMain"] [data-testid="stExpander"] details, [data-testid="stMain"] [data-testid="stExpander"] summary {
    background: #fff !important; }
[data-testid="stMain"] [data-testid="stExpander"] summary, [data-testid="stMain"] [data-testid="stExpander"] summary * {
    color: #1b3a55 !important; font-weight: 650; }

/* ---- sidebar ---- */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: linear-gradient(180deg,#0f2036 0%,#112a42 100%) !important; }
[data-testid="stSidebar"] { border-right: 1px solid #27455d; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #bdd0e3 !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"], [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #e8f2fb !important; font-weight: 650; font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stSidebar"] hr { border-color: #27455d !important; margin: .9rem 0; }
/* nav: hide radio bullet, make pill buttons */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 10px 14px !important; border-radius: 10px !important;
    border: 1px solid transparent; transition: background .15s ease; width: 100%; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] label * { color: #dbe8f5 !important; font-weight: 560; font-size: .95rem; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #1d3b55 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #14505a !important; border-color: #2f9ea3 !important;
    box-shadow: inset 4px 0 0 #5ddcd6; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color: #fff !important; font-weight: 780; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #0a182b !important; border: 1px solid #3a5671 !important; border-radius: 10px !important; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #f5faff !important; text-transform: none; letter-spacing: 0; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stButton"] button { background: #1c3a52 !important; border: 1px solid #44647c !important; border-radius: 10px; width: 100%; }
[data-testid="stSidebar"] [data-testid="stButton"] button * { color: #fff !important; text-transform: none; letter-spacing: 0; }
.sb-brand { padding: 4px 2px 6px; }
.sb-brand .t { color: #fff; font-size: 1.2rem; font-weight: 850; letter-spacing: .03em; }
.sb-brand .k { color: #7fe0d9; text-transform: uppercase; font-size: .66rem; font-weight: 750; letter-spacing: .12em; margin-top: 4px; }

/* ---- components ---- */
.eyebrow { display: inline-block; color: #07767c !important; background: #dff1f0; border-radius: 999px; padding: 4px 12px;
    font-size: .72rem; font-weight: 800; letter-spacing: .1em; }
.intro { color: #4f647a !important; font-size: 1.02rem; line-height: 1.55; margin: .1rem 0 1.3rem; max-width: 900px; }
.hero { background: radial-gradient(900px 260px at 100% 0%, rgba(93,220,214,.28), transparent 70%), linear-gradient(115deg,#0f2339,#175064);
    border-radius: 18px; padding: 28px 32px; margin: 4px 0 20px; box-shadow: 0 10px 28px rgba(15,35,57,.16); }
.hero .strap { color: #8fece4 !important; font-size: .74rem; letter-spacing: .13em; font-weight: 800; }
.hero .headline { color: #fff !important; font-weight: 800; font-size: clamp(1.35rem,2.1vw,1.95rem); line-height: 1.28; margin: 9px 0 8px; }
.hero .detail { color: #dbeaf2 !important; font-size: .98rem; line-height: 1.6; max-width: 980px; }
.hero .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.hero .chip { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2); color: #fff !important;
    border-radius: 999px; padding: 5px 13px; font-size: .8rem; font-weight: 650; }
.kpi { background: #fff; border: 1px solid #d7e2ec; border-radius: 14px; padding: 16px 18px 15px; min-height: 134px;
    border-top: 4px solid var(--accent,#0a8a90); box-shadow: 0 4px 14px rgba(16,42,62,.04); }
.kpi .lab { color: #4a6278 !important; font-size: .78rem; font-weight: 750; text-transform: uppercase; letter-spacing: .05em; }
.kpi .val { color: #12263d !important; font-size: clamp(1.35rem,1.75vw,1.9rem); font-weight: 820; margin: 9px 0 5px;
    overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.kpi .sub { color: #55697e !important; font-size: .8rem; line-height: 1.45; }
.insight-card { background: #fff; border: 1px solid #d7e2ec; border-left: 4px solid var(--accent,#0a8a90); border-radius: 14px;
    padding: 15px 18px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(16,42,62,.03); }
.insight-card .tiny { color: #4f677c !important; font-size: .74rem; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }
.insight-card .big { color: #12263d !important; font-size: 1.75rem; font-weight: 830; margin: 5px 0 3px; }
.insight-card .desc { color: #455e73 !important; font-size: .88rem; line-height: 1.5; }
.callout { border-radius: 12px; padding: 13px 17px; margin: 10px 0 18px; line-height: 1.6; font-size: .94rem;
    border: 1px solid var(--bd); background: var(--bg); border-left: 4px solid var(--ac); }
.callout, .callout * { color: var(--tx) !important; }
.callout.info { --bg:#e5f3f3; --bd:#cbe5e5; --ac:#0a8a90; --tx:#173d46; }
.callout.warn { --bg:#fdf3df; --bd:#f1dfb5; --ac:#c98a2b; --tx:#58410f; }
.callout.error { --bg:#fde9ea; --bd:#f3c9cc; --ac:#d2535b; --tx:#6c1f26; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 800; letter-spacing: .02em; }
.pill.ok { background: #dff3ea; color: #0b6d4c !important; } .pill.bad { background: #fde3e5; color: #a42a35 !important; }
.pill.sel { background: #d9f0ef; color: #06656b !important; }
.pcard { background: #fff; border: 1px solid #d7e2ec; border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 14px rgba(16,42,62,.04);
    border-top: 5px solid var(--accent,#0a8a90); }
.pcard.sel { box-shadow: 0 0 0 2px var(--accent), 0 8px 22px rgba(16,42,62,.08); }
.pcard .pt { color: #12263d !important; font-weight: 800; font-size: 1.1rem; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.pcard .pg { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px 18px; margin-top: 14px; }
.pcard .pl { color: #50667b !important; font-size: .74rem; font-weight: 750; text-transform: uppercase; letter-spacing: .04em; }
.pcard .pv { color: #12263d !important; font-size: 1.5rem; font-weight: 820; font-variant-numeric: tabular-nums; }
.tbl-wrap { background: #fff; border: 1px solid #d7e2ec; border-radius: 13px; overflow: auto; margin: .3rem 0 .9rem; box-shadow: 0 3px 10px rgba(16,42,62,.03); }
.tbl { border-collapse: collapse; width: 100%; font-size: .88rem; }
.tbl th { position: sticky; top: 0; background: #eaf0f6; color: #28435d !important; text-align: left; padding: 10px 14px;
    font-weight: 750; font-size: .76rem; text-transform: uppercase; letter-spacing: .04em; border-bottom: 1px solid #d3dfe9; white-space: nowrap; }
.tbl td { padding: 9px 14px; color: #1e3550 !important; border-bottom: 1px solid #edf2f7; white-space: nowrap; }
.tbl tr:nth-child(even) td { background: #f8fafc; }
.tbl tr:hover td { background: #eef6f6; }
.tbl td.num, .tbl th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tbl td.win { background: #dff3f1 !important; color: #06565b !important; font-weight: 800; }
.tbl td.first { font-weight: 700; color: #12263d !important; }
.footer-note { color: #5a6e83 !important; font-size: .8rem; border-top: 1px solid #d7e2ec; padding-top: 15px; margin-top: 34px; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #d9e4ee; border-radius: 14px; overflow: hidden;
    box-shadow: 0 4px 14px rgba(16,42,62,.03); }
</style>""", unsafe_allow_html=True)


# ───────────────────────────── helpers ─────────────────────────────
def stretch(fn, *args, **kwargs):
    """Full-width rendering that works on old and new Streamlit versions."""
    try:
        return fn(*args, width='stretch', **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def kpi(col, label, value, sub='', accent=TEAL):
    label, value, sub = (html.escape(str(x)) for x in (label, value, sub))
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(label, big, desc, accent=TEAL):
    label, big, desc = (html.escape(str(x)) for x in (label, big, desc))
    st.markdown(f'<div class="insight-card" style="--accent:{accent}"><div class="tiny">{label}</div>'
                f'<div class="big">{big}</div><div class="desc">{desc}</div></div>', unsafe_allow_html=True)


def callout(body, kind='info'):
    """Theme-independent replacement for st.info / st.warning / st.error. `body` may contain trusted HTML."""
    st.markdown(f'<div class="callout {kind}">{body}</div>', unsafe_allow_html=True)


def note(body):
    callout(body, 'info')


def hero(strap, headline, detail, chips=()):
    strap, headline, detail = (html.escape(str(x)) for x in (strap, headline, detail))
    chip_html = ''.join(f'<span class="chip">{html.escape(str(c))}</span>' for c in chips)
    st.markdown(f'<div class="hero"><div class="strap">{strap}</div><div class="headline">{headline}</div>'
                f'<div class="detail">{detail}</div>' + (f'<div class="chips">{chip_html}</div>' if chip_html else '') + '</div>',
                unsafe_allow_html=True)


def panel(fig, height=420):
    """Single accessible Plotly style across native interactive charts."""
    fig.update_layout(
        template='plotly_white', height=height, autosize=True,
        margin=dict(l=28, r=26, t=92, b=62),
        font=dict(family='Segoe UI, Inter, Arial, sans-serif', color=INK, size=14),
        title_font=dict(size=18, color=NAVY),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        hoverlabel=dict(bgcolor='white', font_color=NAVY, font_size=13, bordercolor='#cbd8e4'),
        hovermode='closest',
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='#cfdbe6', tickfont=dict(color='#31465f', size=12),
                     title_font=dict(color='#31465f', size=13), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color='#31465f', size=12),
                     title_font=dict(color='#31465f', size=13), automargin=True)
    if fig.layout.showlegend is not False and any(t.showlegend is not False for t in fig.data) and len(fig.data) > 1:
        fig.update_layout(legend=dict(orientation='h', x=0, y=1.07, yanchor='bottom', title_text='',
                                      font=dict(size=12, color=INK)))
    stretch(st.plotly_chart, fig, config={'displaylogo': False, 'responsive': True, 'displayModeBar': 'hover',
                                           'modeBarButtonsToRemove': ['lasso2d', 'select2d']})


def _cell(v, col, fmt, raw):
    if col in raw:
        return str(v), ''
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return '—', 'num'
    f = fmt.get(col)
    if f is not None:
        try:
            return html.escape(f.format(v) if isinstance(f, str) else f(v)), 'num'
        except Exception:
            pass
    if isinstance(v, (bool, np.bool_)):
        return str(v), ''
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}', 'num'
    if isinstance(v, (float, np.floating)):
        return (f'{v:,.2f}' if abs(v) >= 1000 else f'{v:.4g}'), 'num'
    return html.escape(str(v)), ''


def table(df, fmt=None, height=None, raw=()):
    """Light, theme-independent HTML table (the native canvas grid turns black in dark browsers)."""
    fmt, raw = fmt or {}, set(raw)
    num_cols = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in raw}
    head = ''.join(f'<th class="{"num" if c in num_cols else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body = []
    for _, r in df.iterrows():
        tds = []
        for j, c in enumerate(df.columns):
            s, cls = _cell(r[c], c, fmt, raw)
            if j == 0 and c not in num_cols:
                cls += ' first'
            tds.append(f'<td class="{cls.strip()}">{s}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    style = f' style="max-height:{height}px"' if height else ''
    st.markdown(f'<div class="tbl-wrap"{style}><table class="tbl"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(body)}</tbody></table></div>', unsafe_allow_html=True)


def compare_table(left_name, right_name, rows):
    """rows = (label, left_value, right_value, formatter, higher_is_better|None). Shades the more favourable cell."""
    out = []
    for label, a, b, f, hib in rows:
        ca = cb = ''
        if hib is not None and a != b:
            a_wins = (a > b) if hib else (a < b)
            ca, cb = ('win', '') if a_wins else ('', 'win')
        out.append(f'<tr><td class="first">{html.escape(label)}</td><td class="num {ca}">{f(a)}</td><td class="num {cb}">{f(b)}</td></tr>')
    st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Measure</th><th class="num">{html.escape(left_name)}</th>'
                f'<th class="num">{html.escape(right_name)}</th></tr></thead><tbody>{"".join(out)}</tbody></table></div>',
                unsafe_allow_html=True)


def download(df, name, label='⬇ Download CSV'):
    st.download_button(label, df.to_csv(index=False).encode('utf-8'), file_name=name, mime='text/csv')


def failure(exc: Exception):
    callout(html.escape(str(exc)), 'error')
    callout('Regenerate the corresponding reports with the existing src module. '
            'The dashboard never substitutes estimated or demo numbers.', 'info')


def signed(v, digits=2):
    return f'{v:+,.{digits}f}'


def decile_frame(band: pd.DataFrame) -> pd.DataFrame:
    d = band.sort_values('amount_band').copy()
    d['decile'] = 'Q' + (d.amount_band.astype(int) + 1).astype(str).str.zfill(2)
    d['cum_cases'] = d.share_of_fraud_cases_pct.cumsum()
    d['cum_value'] = d.share_of_fraud_value_pct.cumsum()
    d['gap_pp'] = d.share_of_fraud_value_pct - d.share_of_fraud_cases_pct
    return d


@st.cache_data(show_spinner=False, ttl=60)
def cached_table(rel: str):
    return read_csv(ROOT, rel)


@st.cache_data(show_spinner=False, ttl=60)
def cached_run(run: str):
    return get_run(ROOT, run)


@st.cache_data(show_spinner=False, ttl=60)
def cached_opt(run: str):
    return get_optimization(ROOT, run)


# ───────────────────────────── sidebar ─────────────────────────────
runs = discover_runs(ROOT)
PAGES = ['Overview', 'Risk & Statistics', 'Model Lab', 'Investigation Studio', 'Data & Methodology']
ICONS = {'Overview': '◧', 'Risk & Statistics': '∑', 'Model Lab': '◉', 'Investigation Studio': '⚖', 'Data & Methodology': '☰'}
with st.sidebar:
    st.markdown('<div class="sb-brand"><div class="t">🛡️ FRAUD RISK LAB</div>'
                '<div class="k">Fraud intelligence · Research portfolio</div></div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio('Workspace', PAGES, format_func=lambda p: f'{ICONS[p]}   {p}', label_visibility='collapsed')
    st.divider()
    selected_run = None
    if runs:
        preferred = next((r for r in runs if r.startswith('chronological_')), runs[0])
        selected_run = st.selectbox('Experiment', runs, index=runs.index(preferred),
                                    format_func=lambda r: ('Chronological' if r.startswith('chronological_') else 'Random') + ' · ' + r.rsplit('_', 1)[-1])
    else:
        st.caption('No complete ML experiment found; model pages will explain required inputs.')
    st.caption('Original dataset → descriptive reports\n\nML → deduplicated holdout')
    st.divider()
    if st.button('↻ Refresh reports', help='Reload regenerated CSV and JSON reports'):
        st.cache_data.clear()
        st.rerun()
    st.caption('ULB dataset · Offline analytics · No live transactions')

TITLES = {'Overview': 'Fraud Risk Intelligence', 'Risk & Statistics': 'Risk & Statistical Evidence',
          'Model Lab': 'Model Evaluation Lab', 'Investigation Studio': 'Investigation Strategy Studio',
          'Data & Methodology': 'Data Provenance & Methodology'}
INTROS = {
    'Overview': 'Where fraud concentrates — by case count, by recorded value, and across elapsed time.',
    'Risk & Statistics': 'Segment risk and exploratory hypothesis tests, with every caveat kept visible.',
    'Model Lab': 'Locked-holdout evaluation of the selected model, at the threshold chosen on training data.',
    'Investigation Studio': 'Compare batch review-queue rankings under a fixed analyst capacity, then stress-test the economics.',
    'Data & Methodology': 'Where every number comes from, and what this analysis can and cannot claim.',
}
st.markdown(f'<span class="eyebrow">FRAUD ANALYTICS / {html.escape(page.upper())}</span>', unsafe_allow_html=True)
st.title(TITLES[page])
st.markdown(f'<div class="intro">{html.escape(INTROS[page])}</div>', unsafe_allow_html=True)

# ═════════════════════════════ OVERVIEW ═════════════════════════════
if page == 'Overview':
    try:
        k = cached_table('reports/tables/business/executive_kpis.csv').iloc[0]
        band = cached_table('reports/tables/business/amount_segment_exposure.csv')
        verify_business_consistency(k, band)
    except (DataIssue, KeyError, ValueError) as e:
        failure(e)
        st.stop()

    d = decile_frame(band)
    low, high = d.iloc[0], d.iloc[-1]
    top3_value = d.share_of_fraud_value_pct.iloc[-3:].sum()
    top3_cases = d.share_of_fraud_cases_pct.iloc[-3:].sum()
    widest = d.loc[d.gap_pp.abs().idxmax()]

    hero('RESEARCH FINDING / 01', 'Finding frauds and protecting transaction value are not the same goal.',
         f'The lowest-amount decile holds {low.share_of_fraud_cases_pct:.1f}% of fraud cases but only '
         f'{low.share_of_fraud_value_pct:.2f}% of recorded fraudulent value; the highest decile holds '
         f'{high.share_of_fraud_value_pct:.1f}% of the value in {int(high.fraud_cases)} cases.',
         chips=['Historical data', 'Duplicates retained', 'Not verified bank loss'])
    c = st.columns(4, gap='medium')
    kpi(c[0], 'Transactions', fnum(k.transactions), 'Original dataset, duplicates retained', NAVY)
    kpi(c[1], 'Confirmed fraud cases', fnum(k.fraud_cases), 'Class = 1 in original dataset', RED)
    kpi(c[2], 'Fraud prevalence', f'{k.fraud_rate_pct:.3f}%',
        f'≈ 1 in {int(round(100 / k.fraud_rate_pct)):,} transactions' if k.fraud_rate_pct > 0 else 'No fraud cases observed', TEAL)
    kpi(c[3], 'Recorded fraudulent value', fnum(k.recorded_fraud_value, 2), 'Dataset amount units; not proven bank loss', GOLD)
    st.write('')

    t1, t2 = st.tabs(['Transaction value concentration', 'Elapsed-time patterns'])
    with t1:
        left, right = st.columns([1.65, 1], gap='large')
        with left:
            fig = go.Figure()
            fig.add_bar(x=d.decile, y=d.share_of_fraud_cases_pct, name='Share of fraud cases', marker_color=TEAL,
                        text=[f'{v:.1f}%' if v >= 5 else '' for v in d.share_of_fraud_cases_pct], textposition='outside', cliponaxis=False,
                        customdata=d[['fraud_cases', 'transactions']].to_numpy(),
                        hovertemplate='%{x}<br>Fraud case share %{y:.2f}%<br>Cases %{customdata[0]:,}<br>Transactions %{customdata[1]:,}<extra></extra>')
            fig.add_bar(x=d.decile, y=d.share_of_fraud_value_pct, name='Share of fraudulent value', marker_color=GOLD,
                        text=[f'{v:.1f}%' if v >= 5 else '' for v in d.share_of_fraud_value_pct], textposition='outside', cliponaxis=False,
                        customdata=d[['fraud_amount']].to_numpy(),
                        hovertemplate='%{x}<br>Fraud value share %{y:.2f}%<br>Recorded value %{customdata[0]:,.2f}<extra></extra>')
            fig.update_layout(title=dict(text='Fraud cases vs. fraudulent value by amount decile', x=.02), barmode='group',
                              yaxis_title='Share of total fraud (%)', xaxis_title='Transaction amount decile (Q01 = lowest)', bargap=.22)
            panel(fig, 450)
        with right:
            insight('Lowest amount decile', f'{low.share_of_fraud_cases_pct:.1f}% of cases',
                    f'but only {low.share_of_fraud_value_pct:.2f}% of recorded fraudulent value.', TEAL)
            insight('Highest amount decile', f'{high.share_of_fraud_value_pct:.1f}% of value',
                    f'across {int(high.fraud_cases)} fraud cases ({high.share_of_fraud_cases_pct:.1f}% of cases).', GOLD)
            insight('Top three deciles', f'{top3_value:.1f}% of value',
                    f'hold {top3_cases:.1f}% of fraud cases. Largest case/value gap: {widest.decile} ({signed(widest.gap_pp, 1)} pp).', NAVY)

        l2, r2 = st.columns(2, gap='large')
        with l2:
            xs = ['Q00'] + d.decile.tolist()
            fig = go.Figure()
            fig.add_scatter(x=xs, y=np.linspace(0, 100, len(xs)), mode='lines', name='Equal share',
                            line=dict(color=MUTED, width=2, dash='dot'), hoverinfo='skip')
            fig.add_scatter(x=xs, y=[0] + d.cum_cases.tolist(), mode='lines+markers', name='Fraud cases',
                            line=dict(color=TEAL, width=3), marker=dict(size=7))
            fig.add_scatter(x=xs, y=[0] + d.cum_value.tolist(), mode='lines+markers', name='Fraudulent value',
                            line=dict(color=GOLD, width=3), marker=dict(size=7), fill='tonexty', fillcolor='rgba(201,138,43,.08)')
            fig.update_layout(title=dict(text='Cumulative concentration curve', x=.02),
                              yaxis_title='Cumulative share of fraud (%)', xaxis_title='Amount decile (low → high)')
            panel(fig, 380)
        with r2:
            fig = go.Figure(go.Bar(x=d.decile, y=d.gap_pp, marker_color=[GOLD if v >= 0 else TEAL for v in d.gap_pp],
                                   hovertemplate='%{x}<br>Value share − case share: %{y:+.2f} pp<extra></extra>'))
            fig.add_hline(y=0, line_color=MUTED)
            fig.update_layout(title=dict(text='Where value outruns case count (percentage points)', x=.02),
                              yaxis_title='Value share − case share (pp)', xaxis_title='Amount decile', showlegend=False)
            panel(fig, 380)
        note('<b>Implication:</b> review policies need both fraud-case recall and fraudulent-value recall. '
             'These figures describe historical amounts, not prevented losses.')
        with st.expander('Underlying decile data'):
            table(d.drop(columns=['decile']).round(4))
            download(d, 'amount_decile_exposure.csv')

    with t2:
        try:
            tm = cached_table('reports/tables/business/time_segment_exposure.csv')
            colors = [RED if v == tm.fraud_rate_pct.max() else TEAL for v in tm.fraud_rate_pct]
            fig = go.Figure(go.Bar(x=tm.time_segment, y=tm.fraud_rate_pct, marker_color=colors,
                                   customdata=tm[['transactions', 'fraud_cases']].to_numpy(),
                                   hovertemplate='%{x}<br>Fraud rate %{y:.3f}%<br>Transactions %{customdata[0]:,}<br>Fraud cases %{customdata[1]:,}<extra></extra>'))
            fig.add_hline(y=float(k.fraud_rate_pct), line_dash='dash', line_color=GOLD,
                          annotation_text=f'Overall {k.fraud_rate_pct:.3f}%', annotation_position='top left')
            fig.update_layout(title=dict(text='Observed fraud rate by 4-hour data interval', x=.02),
                              xaxis_title='Elapsed time segment', yaxis_title='Fraud rate (%)', showlegend=False)
            panel(fig, 410)
            st.caption('Elapsed dataset time is not local time of day. The ≈ two-day sample does not establish recurring daily patterns.')
        except (DataIssue, KeyError) as e:
            callout(html.escape(str(e)), 'warn')

# ═════════════════════════════ RISK & STATISTICS ═════════════════════════════
elif page == 'Risk & Statistics':
    try:
        risk = cached_table('reports/tables/statistics/segment_risk.csv')
        tests = cached_table('reports/tables/statistics/hypothesis_tests.csv')
        band = cached_table('reports/tables/business/amount_segment_exposure.csv')
    except DataIssue as e:
        failure(e)
        st.stop()

    x = risk.loc[risk.comparison == 'zero_vs_positive']
    if not x.empty:
        rr = x.iloc[0]
        c = st.columns(3, gap='medium')
        kpi(c[0], 'Zero-amount fraud rate', f'{rr.fraud_rate_pct:.2f}%', 'Records with zero recorded value', RED)
        kpi(c[1], 'Positive-amount fraud rate', f'{rr.reference_rate_pct:.3f}%', 'Reference group', TEAL)
        kpi(c[2], 'Observed relative risk', f'{rr.risk_ratio:.2f}×',
            f'Approx. 95% interval {rr.risk_ratio_ci_low:.2f}–{rr.risk_ratio_ci_high:.2f}', GOLD)
    note('<b>Association is not causation.</b> Zero-amount records have zero recorded transaction value, '
         'and their underlying transaction purpose is unknown.')

    d = decile_frame(band)
    t1, t2, t3 = st.tabs(['Amount exposure', 'Segment risk', 'Hypothesis tests'])
    with t1:
        l, r = st.columns(2, gap='large')
        with l:
            overall = d.fraud_cases.sum() / d.transactions.sum() * 100
            fig = px.bar(d, x='decile', y='fraud_rate_pct', color_discrete_sequence=[TEAL], title='Fraud rate by amount decile',
                         hover_data=['transactions', 'fraud_cases'])
            fig.add_hline(y=overall, line_dash='dash', line_color=GOLD, annotation_text=f'Average {overall:.3f}%',
                          annotation_position='top right')
            fig.update_layout(yaxis_title='Fraud rate (%)', xaxis_title='Amount decile', title_x=.02)
            panel(fig, 420)
        with r:
            fig = px.bar(d, x='decile', y='share_of_fraud_value_pct', color_discrete_sequence=[GOLD],
                         title='Share of fraudulent value', hover_data=['fraud_amount'])
            fig.update_layout(yaxis_title='Fraud value share (%)', xaxis_title='Amount decile', title_x=.02)
            panel(fig, 420)
    with t2:
        need = {'comparison', 'risk_ratio', 'risk_ratio_ci_low', 'risk_ratio_ci_high'}
        if need.issubset(risk.columns):
            rk = risk.dropna(subset=['risk_ratio']).copy()
            fig = go.Figure()
            fig.add_scatter(x=rk.risk_ratio, y=rk.comparison, mode='markers', marker=dict(size=12, color=TEAL),
                            error_x=dict(type='data', symmetric=False, array=rk.risk_ratio_ci_high - rk.risk_ratio,
                                         arrayminus=rk.risk_ratio - rk.risk_ratio_ci_low, color=MUTED),
                            hovertemplate='%{y}<br>RR %{x:.2f}<extra></extra>')
            fig.add_vline(x=1, line_dash='dash', line_color=RED, annotation_text='No difference (RR = 1)')
            fig.update_layout(title=dict(text='Relative risk with approximate 95% intervals', x=.02),
                              xaxis_title='Risk ratio (log scale)', xaxis_type='log', showlegend=False)
            panel(fig, 170 + 60 * len(rk))
        table(risk)
        download(risk, 'segment_risk.csv')
    with t3:
        if {'feature', 'p_value_holm'}.issubset(tests.columns):
            tt = tests.copy()
            tt['neg_log10_p'] = -np.log10(tt.p_value_holm.clip(lower=1e-300))
            tt = tt.sort_values('neg_log10_p').tail(20)
            fig = go.Figure(go.Bar(x=tt.neg_log10_p, y=tt.feature, orientation='h',
                                   marker_color=[TEAL if p < .05 else MUTED for p in tt.p_value_holm],
                                   hovertemplate='%{y}<br>−log10(p) %{x:.1f}<extra></extra>'))
            fig.add_vline(x=-np.log10(.05), line_dash='dash', line_color=RED, annotation_text='Holm p = 0.05')
            fig.update_layout(title=dict(text='Strongest exploratory signals (top 20)', x=.02),
                              xaxis_title='−log10(Holm-adjusted p)', showlegend=False)
            panel(fig, 170 + 24 * len(tt))
        cols = [c for c in ['feature', 'test', 'p_value_holm', 'effect_size_name', 'effect_size'] if c in tests.columns]
        formats = {}
        if 'p_value_holm' in cols:
            formats['p_value_holm'] = '{:.3e}'
        if 'effect_size' in cols:
            formats['effect_size'] = '{:.4f}'
        st.caption('P-values are displayed in scientific notation; values remain unchanged in downloaded CSV.')
        table(tests[cols], fmt=formats, height=460)
        download(tests[cols], 'hypothesis_tests.csv')
        st.caption('Holm-adjusted p-values from exploratory tests; post-selection PCA tests are not confirmatory. '
                   'Statistical significance is not operational usefulness.')

# ═════════════════════════════ MODEL LAB ═════════════════════════════
elif page == 'Model Lab':
    if not selected_run:
        callout('No ML run found. Run <code>python -m src.ml_models</code> first.', 'warn')
        st.stop()
    try:
        met, meta = cached_run(selected_run)
    except DataIssue as e:
        failure(e)
        st.stop()

    tn, fp, fn, tp = (int(met.tn), int(met.fp), int(met.fn), int(met.tp))
    n = tn + fp + fn + tp
    alerts = tp + fp
    specificity = tn / (tn + fp) if tn + fp else float('nan')
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')
    alert_rate = alerts / n * 10000 if n else float('nan')
    alerts_per_fraud = alerts / tp if tp else float('nan')

    note(f'<b>{html.escape(str(met.selected_model))}</b> · {html.escape(str(met.split))} holdout · threshold {met.threshold:.6f} '
         'chosen on training OOF. This is a historical evaluation, not live model performance.')
    c = st.columns(4, gap='medium')
    kpi(c[0], 'Average precision', f'{met.pr_auc:.3f}', 'PR-AUC on locked holdout', NAVY)
    kpi(c[1], 'Precision', f'{met.precision:.1%}', f'{alerts_per_fraud:.1f} alerts per fraud caught', GOLD)
    kpi(c[2], 'Fraud case recall', f'{met.recall:.1%}', f'{tp} caught · {fn} missed', TEAL)
    kpi(c[3], 'Fraud value recall', f'{met.fraud_value_recall:.1%}', 'Share of recorded fraud value flagged', RED)
    st.write('')

    t1, t2, t3, t4 = st.tabs(['Performance', 'Operating profile', 'Experiments', 'Diagnostic figures'])
    with t1:
        mode = st.radio('Confusion matrix view', ['Counts', 'Row %'], horizontal=True)
        left, right = st.columns([.95, 1.05], gap='large')
        with left:
            m = np.array([[tn, fp], [fn, tp]], dtype=float)
            if mode == 'Row %':
                z = m / m.sum(axis=1, keepdims=True) * 100
                text = [[f'{v:.2f}%' for v in row] for row in z]
            else:
                z = np.log10(m + 1)
                text = [[f'{int(v):,}' for v in row] for row in m]
            fig = go.Figure(go.Heatmap(z=z, x=['Predicted legitimate', 'Predicted fraud'], y=['Actual legitimate', 'Actual fraud'],
                                       text=text, texttemplate='%{text}', textfont=dict(size=18),
                                       colorscale=[[0, '#eef3fa'], [1, '#087e8b']], showscale=False, xgap=3, ygap=3,
                                       hoverinfo='skip'))
            fig.update_yaxes(autorange='reversed', showgrid=False)
            fig.update_layout(title=dict(text='Locked holdout confusion matrix', x=.02))
            panel(fig, 390)
        with right:
            vals = {'Fraud detected': tp, 'Fraud missed': fn, 'Legitimate false alerts': fp}
            fig = go.Figure(go.Bar(x=list(vals), y=list(vals.values()), marker_color=[TEAL, RED, GOLD],
                                   text=[f'{v:,}' for v in vals.values()], textposition='outside', cliponaxis=False))
            fig.update_layout(title=dict(text='Classification outcomes (counts)', x=.02), yaxis_title='Transactions', showlegend=False)
            panel(fig, 390)
        st.caption('Confusion counts refer only to the selected holdout population, not the full raw dataset.')
    with t2:
        c = st.columns(4, gap='medium')
        kpi(c[0], 'Alert load', f'{alert_rate:,.1f} / 10k', 'Transactions flagged per 10,000', NAVY)
        kpi(c[1], 'Specificity', f'{specificity:.3%}', 'Legitimate transactions left alone', TEAL)
        kpi(c[2], 'F1 score', f'{f1:.3f}', 'Harmonic mean of precision & recall', GOLD)
        kpi(c[3], 'Missed fraud value', f'{1 - met.fraud_value_recall:.1%}', 'Recorded value not flagged', RED)
        st.write('')
        gl, gr = st.columns(2, gap='large')
        with gl:
            fig = go.Figure()
            fig.add_bar(y=['Fraud cases', 'Fraud value'], x=[met.recall * 100, met.fraud_value_recall * 100], orientation='h',
                        marker_color=[TEAL, GOLD], text=[f'{met.recall:.1%}', f'{met.fraud_value_recall:.1%}'], textposition='inside',
                        textfont=dict(color='#fff', size=15))
            fig.update_layout(title=dict(text='Coverage achieved at the chosen threshold', x=.02),
                              xaxis=dict(range=[0, 100], title='Coverage (%)'), showlegend=False)
            panel(fig, 300)
        with gr:
            fig = go.Figure()
            fig.add_bar(y=['Of all fraud cases', 'Of all alerts'], x=[tp, tp], orientation='h', name='Fraud caught / true alert', marker_color=TEAL,
                        text=[f'{tp:,}', f'{tp:,}'], textposition='inside', textfont=dict(color='#fff', size=14))
            fig.add_bar(y=['Of all fraud cases', 'Of all alerts'], x=[fn, 0], orientation='h', name='Fraud missed', marker_color=RED,
                        text=[f'{fn:,}' if fn else '', ''], textposition='inside', textfont=dict(color='#fff', size=14))
            fig.add_bar(y=['Of all fraud cases', 'Of all alerts'], x=[0, fp], orientation='h', name='False alert', marker_color=GOLD,
                        text=['', f'{fp:,}' if fp else ''], textposition='inside', textfont=dict(color='#fff', size=14))
            fig.update_layout(barmode='stack', barnorm='percent', title=dict(text='What the review queue contains', x=.02),
                              xaxis_title='Share (%)')
            panel(fig, 300)
        st.caption(f'{alerts:,} alerts from {n:,} holdout transactions; {tp:,} confirmed frauds in the review queue.')
    with t3:
        if len(runs) > 1:
            rows = []
            smoke_runs = []
            for run in runs:
                try:
                    mm, run_meta = cached_run(run)
                    # Only full 3 × 3 candidate searches belong in headline comparison.
                    if len(run_meta.get('model_choices', [])) < 3 or len(run_meta.get('strategies', [])) < 3:
                        smoke_runs.append(run)
                        continue
                    rows.append({'Experiment': run, 'Split': str(mm.split), 'Selected model': str(mm.selected_model),
                                 'PR-AUC': float(mm.pr_auc), 'Precision': float(mm.precision), 'Recall': float(mm.recall),
                                 'Value recall': float(mm.fraud_value_recall), 'Fraud cases': int(mm.test_fraud)})
                except DataIssue:
                    pass
            if rows:
                cmp_df = pd.DataFrame(rows)
                fig = px.bar(cmp_df.melt(id_vars='Experiment', value_vars=['PR-AUC', 'Precision', 'Recall', 'Value recall']),
                             x='Experiment', y='value', color='variable', barmode='group',
                             color_discrete_sequence=[NAVY, GOLD, TEAL, RED], title='Metrics across experiment settings')
                fig.update_layout(yaxis_title='Score', yaxis_range=[0, 1], title_x=.02)
                panel(fig, 400)
                table(cmp_df, {'PR-AUC': '{:.3f}', 'Precision': '{:.1%}', 'Recall': '{:.1%}', 'Value recall': '{:.1%}'})
                download(cmp_df, 'experiment_comparison.csv')
                st.caption('Full nine-candidate runs only. Different holdout populations: descriptive comparison, not a paired experiment.')
                if smoke_runs:
                    with st.expander(f'{len(smoke_runs)} smoke/partial run(s) excluded from headline comparison'):
                        st.write(smoke_runs)
        else:
            callout('Only one saved experiment is available; nothing to compare yet.', 'info')
    with t4:
        folder = ROOT / 'reports/figures/ml' / selected_run
        figure_options = {
            'Precision–Recall and ROC curves': '02_ranking_curves.png',
            'Score reliability and distributions': '03_descriptive_reliability.png',
            'False positives and missed fraud': '04_error_characteristics.png',
            'Detected versus missed fraud value': '05_fraud_value.png',
            'Training cross-validation evidence': '06_training_cv_evidence.png',
            'Holdout temporal segments': '07_holdout_temporal_segments.png',
        }
        figure_name = st.selectbox('Diagnostic figure', list(figure_options), key='figure_choice')
        figure_path = folder / figure_options[figure_name]
        if figure_path.is_file():
            stretch(st.image, str(figure_path), caption=f'{figure_name} · {selected_run}')
        else:
            callout(f'Missing diagnostic figure: <code>{html.escape(figure_path.name)}</code>. Run src.model_evaluation with --format png.', 'warn')
        st.caption('These fixed holdout visualizations are descriptive. No thresholds or probability calibration '
                   'parameters are estimated on the test set.')

# ═════════════════════════════ INVESTIGATION STUDIO ═════════════════════════════
elif page == 'Investigation Studio':
    if not selected_run:
        callout('No ML experiment is available.', 'warn')
        st.stop()
    try:
        df, choices = cached_opt(selected_run)
    except DataIssue as e:
        failure(e)
        st.stop()

    note('<b>Experiment design:</b> Both predeclared ranking strategies are shown on the same historical holdout. '
         'Training OOF alone selected the operating policy at each capacity. This is <b>batch queue ranking</b>, '
         'not live streaming or automatic transaction blocking.')

    capacities = sorted(df.capacity_per_10000.unique())
    if not capacities:
        callout('Optimization report contains no review capacities.', 'error')
        st.stop()
    cap = st.select_slider('Review allowance / 10,000 transactions', options=capacities,
                           value=capacities[min(2, len(capacities) - 1)], format_func=lambda v: f'{v:g} reviews / 10k')
    shown = df[df.capacity_per_10000 == cap].copy().sort_values('policy')
    chosen = str(choices.loc[choices.capacity_per_10000 == cap, 'selected_policy'].iloc[0])
    st.caption(f'Training-selected ranking at this budget: **{POLICY_LABEL[chosen]}**. '
               'Changing the display budget does not retrain or reselect a policy.')

    policy_order = ['fraud_score', 'score_times_amount']
    by_policy = {key: shown.loc[shown.policy.eq(key)].iloc[0] for key in policy_order}
    base_row, value_row = by_policy['fraud_score'], by_policy['score_times_amount']
    delta_value = float(value_row.value_recall_pct) - float(base_row.value_recall_pct)
    delta_cases = int(value_row.tp) - int(base_row.tp)
    delta_fp = int(value_row.fp) - int(base_row.fp)
    delta_legit = float(value_row.legitimate_value_flagged) - float(base_row.legitimate_value_flagged)

    st.markdown('##### Score × amount **minus** Fraud score, at this review budget')
    dc = st.columns(4, gap='medium')
    kpi(dc[0], 'Fraud-value coverage', f'{signed(delta_value)} pp', 'Recorded fraudulent value', GOLD)
    kpi(dc[1], 'Fraud cases detected', f'{delta_cases:+,d}', 'Case count difference', TEAL)
    kpi(dc[2], 'Legitimate false alerts', f'{delta_fp:+,d}', 'Count difference', RED)
    kpi(dc[3], 'Legitimate value flagged', signed(delta_legit), 'Dataset amount units', NAVY)
    callout('Historical comparison on the holdout — <b>not grounds to change the training-selected policy.</b>', 'warn')

    t1, t2, t3 = st.tabs(['Policy comparison', 'Trade-off frontier', 'Financial what-if'])

    with t1:
        pc = st.columns(2, gap='large')
        for col, key in zip(pc, policy_order):
            row = by_policy[key]
            prec = row.tp / row.alerts * 100 if row.alerts else float('nan')
            sel = key == chosen
            pill = '<span class="pill sel">✓ TRAINING SELECTED</span>' if sel else ''
            col.markdown(
                f'<div class="pcard{" sel" if sel else ""}" style="--accent:{POLICY_COLOR[key]}">'
                f'<div class="pt"><span>{POLICY_LABEL[key]}</span>{pill}</div><div class="pg">'
                f'<div><div class="pl">Fraud cases detected</div><div class="pv">{int(row.tp):,}</div></div>'
                f'<div><div class="pl">Fraud-case recall</div><div class="pv">{row.case_recall_pct:.1f}%</div></div>'
                f'<div><div class="pl">Fraud-value recall</div><div class="pv">{row.value_recall_pct:.1f}%</div></div>'
                f'<div><div class="pl">Queue precision</div><div class="pv">{prec:.1f}%</div></div>'
                f'</div></div>', unsafe_allow_html=True)
        st.write('')
        st.markdown('#### Same review budget · side-by-side policy outcomes')
        st.caption(f'Reviews per 10,000 transactions: {cap:g}. Shaded cell = more favourable on that single measure; '
                   'descriptive only — the selected policy is not changed by viewing holdout results.')
        compare_table('Fraud score', 'Score × amount', [
            ('Review slots used', int(base_row.alerts), int(value_row.alerts), lambda v: f'{v:,}', None),
            ('Fraud cases detected', int(base_row.tp), int(value_row.tp), lambda v: f'{v:,}', True),
            ('Fraud-case recall', float(base_row.case_recall_pct), float(value_row.case_recall_pct), lambda v: f'{v:.2f}%', True),
            ('Recorded fraud-value recall', float(base_row.value_recall_pct), float(value_row.value_recall_pct), lambda v: f'{v:.2f}%', True),
            ('Legitimate false alerts', int(base_row.fp), int(value_row.fp), lambda v: f'{v:,}', False),
            ('Legitimate transaction value flagged', float(base_row.legitimate_value_flagged),
             float(value_row.legitimate_value_flagged), lambda v: f'{v:,.2f}', False),
        ])
        download(shown, 'holdout_policy_comparison_at_selected_capacity.csv')
        st.write('')
        l, r = st.columns(2, gap='large')
        for col, value, title in [(l, 'case_recall_pct', 'Fraud-case recall by review budget'),
                                  (r, 'value_recall_pct', 'Recorded fraudulent-value coverage')]:
            with col:
                plot_df = df.assign(Policy=df.policy.map(POLICY_LABEL))
                fig = px.line(plot_df, x='capacity_per_10000', y=value, color='Policy', markers=True,
                              color_discrete_map={POLICY_LABEL[k]: v for k, v in POLICY_COLOR.items()}, title=title,
                              labels={'capacity_per_10000': 'Reviews per 10,000 transactions', value: 'Coverage (%)'})
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                fig.add_vline(x=cap, line_dash='dot', line_color=MUTED, annotation_text='displayed budget')
                fig.update_layout(yaxis_range=[0, 100], title_x=.02)
                panel(fig, 400)

    with t2:
        l, r = st.columns(2, gap='large')
        with l:
            fig = go.Figure()
            for pol, g in df.sort_values('capacity_per_10000').groupby('policy'):
                fig.add_scatter(x=g.fp, y=g.value_recall_pct, mode='lines+markers+text', name=POLICY_LABEL[pol],
                                text=[f'{c:g}' for c in g.capacity_per_10000], textposition='top center',
                                line=dict(color=POLICY_COLOR[pol], width=3), marker=dict(size=8))
            fig.update_layout(title=dict(text='Value coverage vs. false-alert burden', x=.02),
                              xaxis_title='Legitimate transactions flagged (count; labels = reviews / 10k)',
                              yaxis_title='Value recall (%)')
            panel(fig, 420)
        with r:
            pv = df.assign(precision_pct=np.where(df.alerts > 0, df.tp / df.alerts.replace(0, np.nan) * 100, np.nan),
                           Policy=df.policy.map(POLICY_LABEL))
            fig = px.line(pv, x='capacity_per_10000', y='precision_pct', color='Policy', markers=True,
                          color_discrete_map={POLICY_LABEL[k]: v for k, v in POLICY_COLOR.items()},
                          title='Precision of the review queue',
                          labels={'capacity_per_10000': 'Reviews per 10,000 transactions', 'precision_pct': 'Precision (%)'})
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            fig.update_layout(title_x=.02)
            panel(fig, 420)
        inc = []
        for pol, g in df.sort_values('capacity_per_10000').groupby('policy'):
            g = g.reset_index(drop=True)
            for i in range(1, len(g)):
                dr = g.alerts[i] - g.alerts[i - 1]
                inc.append({'Policy': POLICY_LABEL[pol], 'From → to (reviews/10k)': f'{g.capacity_per_10000[i-1]:g} → {g.capacity_per_10000[i]:g}',
                            'Extra reviews': int(dr), 'Extra fraud caught': int(g.tp[i] - g.tp[i - 1]),
                            'Extra value recall (pp)': float(g.value_recall_pct[i] - g.value_recall_pct[i - 1])})
        if inc:
            st.markdown('#### Marginal return of extra review capacity')
            table(pd.DataFrame(inc), {'Extra value recall (pp)': '{:+.2f}'}, height=380)

    with t3:
        st.caption('Adjust accounting assumptions for the same fixed historical review sets. These controls do not optimize or select any policy.')
        aa, bb, cc = st.columns(3, gap='large')
        rc = aa.number_input('Cost per reviewed transaction (dataset units)', min_value=0., max_value=100000., value=5., step=1.)
        eff = bb.slider('Intervention effectiveness (%)', 0, 100, 50, 5) / 100
        fc = cc.number_input('Additional cost per legitimate alert', min_value=0., max_value=100000., value=0., step=1.)

        display = []
        for _, row in shown.iterrows():
            econ = scenario_economics(row, review_cost=rc, effectiveness=eff, friction_cost=fc)
            display.append({'Policy': POLICY_LABEL[row.policy], 'Reviews': int(row.alerts),
                            'Recorded fraud value flagged': float(row.fraud_value_flagged),
                            'Gross hypothetical benefit': econ['gross'], 'Review + friction costs': econ['cost'],
                            'Hypothetical net value': econ['net'], 'Break-even effectiveness (%)': econ['breakeven_effectiveness'] * 100})
        disp = pd.DataFrame(display)
        table(disp, {'Recorded fraud value flagged': '{:,.2f}', 'Gross hypothetical benefit': '{:,.2f}',
                     'Review + friction costs': '{:,.2f}', 'Hypothetical net value': '{:,.2f}',
                     'Break-even effectiveness (%)': '{:,.1f}'})
        download(disp, 'scenario_economics.csv')

        l, r = st.columns(2, gap='large')
        with l:
            grid = np.arange(0, 101, 5)
            fig = go.Figure()
            for _, row in shown.iterrows():
                net = [scenario_economics(row, review_cost=rc, effectiveness=e / 100, friction_cost=fc)['net'] for e in grid]
                fig.add_scatter(x=grid, y=net, mode='lines', name=POLICY_LABEL[row.policy],
                                line=dict(color=POLICY_COLOR[row.policy], width=3))
            fig.add_hline(y=0, line_color=MUTED, line_dash='dash')
            fig.add_vline(x=eff * 100, line_dash='dot', line_color=NAVY, annotation_text='chosen')
            fig.update_layout(title=dict(text='Hypothetical net value vs. effectiveness', x=.02),
                              xaxis_title='Intervention effectiveness (%)', yaxis_title='Net value (dataset units)')
            panel(fig, 400)
        with r:
            fig = go.Figure()
            for pol, g in df.sort_values('capacity_per_10000').groupby('policy'):
                net = [scenario_economics(row, review_cost=rc, effectiveness=eff, friction_cost=fc)['net'] for _, row in g.iterrows()]
                fig.add_scatter(x=g.capacity_per_10000, y=net, mode='lines+markers', name=POLICY_LABEL[pol],
                                line=dict(color=POLICY_COLOR[pol], width=3), marker=dict(size=8))
            fig.add_hline(y=0, line_color=MUTED, line_dash='dash')
            fig.update_layout(title=dict(text='Hypothetical net value across review budgets', x=.02),
                              xaxis_title='Reviews per 10,000 transactions', yaxis_title='Net value (dataset units)')
            panel(fig, 400)
        callout('Source transaction amounts are not verified bank losses. Effectiveness and costs are hypothetical. '
                'Flagging a legitimate transaction is not equivalent to blocking it. '
                'Do not use holdout outcomes to select a new operating strategy.', 'warn')

# ═════════════════════════════ DATA & METHODOLOGY ═════════════════════════════
else:
    st.subheader('Evidence lineage')
    lineage = pd.DataFrame([
        ['Business KPIs and segments', 'Original 284,807 transactions', 'src.business_analysis', 'Descriptive, duplicates retained'],
        ['Statistical tests', 'Original 284,807 transactions', 'src.statistical_analysis', 'Exploratory hypothesis tests'],
        ['SQL exports', 'Original transactions', 'src.sql_pipeline', 'Independent reconciliation'],
        ['Model evaluations', 'Deduplicated train/test split', 'src.ml_models + src.model_evaluation', 'Locked historical holdouts'],
        ['Investigation policies', 'Training OOF + model holdout', 'src.investigation_optimization',
         'Batch review scenarios, selected on training data'],
    ], columns=['Data family', 'Population', 'Origin', 'Interpretation'])
    st.markdown('This application is **read-only** over the files generated by the completed analysis pipeline.')
    table(lineage)

    st.subheader('Data readiness')
    checks = {'Business KPIs': 'reports/tables/business/executive_kpis.csv',
              'Amount segmentation': 'reports/tables/business/amount_segment_exposure.csv',
              'Time segmentation': 'reports/tables/business/time_segment_exposure.csv',
              'Statistical risk': 'reports/tables/statistics/segment_risk.csv',
              'Hypothesis tests': 'reports/tables/statistics/hypothesis_tests.csv',
              'SQL audit': 'reports/tables/sql/sql_audit.json'}
    checks.update({f'ML run — {r}': f'reports/tables/ml/{r}/run_metadata.json' for r in runs})
    ready = pd.DataFrame([{'Artifact': n, 'Status': 'Present' if (ROOT / p).is_file() else 'Missing', 'Path': p}
                          for n, p in checks.items()])
    present = int((ready.Status == 'Present').sum())
    c = st.columns(3, gap='medium')
    kpi(c[0], 'Artifacts present', f'{present} / {len(ready)}', 'Required report files', TEAL if present == len(ready) else RED)
    kpi(c[1], 'ML experiments', f'{len(runs)}', 'Complete runs discovered', NAVY)
    kpi(c[2], 'Pipeline mode', 'Read-only', 'No training · no demo numbers', GOLD)
    st.write('')
    ready['Status'] = ready.Status.map({'Present': '<span class="pill ok">PRESENT</span>', 'Missing': '<span class="pill bad">MISSING</span>'})
    table(ready, raw=['Status'])

    st.subheader('Methodological boundaries')
    callout('Anonymized historical data covering roughly two days; no cardholder identity, verified currency, real response '
            'outcomes or customer-friction measurements. Score × amount is a heuristic, not calibrated expected savings. '
            'Interventions require independent operational validation.', 'info')
    sql = read_json(ROOT, 'reports/tables/sql/sql_audit.json', optional=True)
    if sql:
        st.caption(f"SQL audit: {sql.get('source_rows', '?'):,} original transactions; {sql.get('fraud_cases', '?')} fraud cases; "
                   f"{len(sql.get('queries', []))} exported SQL queries.")

st.markdown('<div class="footer-note">FRAUD RISK INTELLIGENCE · Research portfolio / historical ULB data · '
            'No real-time bank decisions or verified financial savings · Reports remain the source of truth</div>',
            unsafe_allow_html=True)