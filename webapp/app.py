# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
MOUSE Streamlit Web App
Microreactor Online Unified Simulation Engine — cost estimation without OpenMC.

Run from the MOUSE repo root:
    streamlit run webapp/app.py
"""

# ---------------------------------------------------------------------------
# Ensure the MOUSE repo root is in sys.path so all MOUSE modules resolve.
# ---------------------------------------------------------------------------
import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# IMPORTANT: Stub openmc and watts BEFORE any MOUSE import.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


class _MaterialStub:
    def __init__(self, name=None, temperature=None):
        self.name = name
        self.temperature = temperature
        self.density = 0.0

    def set_density(self, units, value):
        self.density = value

    def add_nuclide(self, *args, **kwargs):
        pass

    def add_element(self, *args, **kwargs):
        pass

    def add_s_alpha_beta(self, *args, **kwargs):
        pass

    @staticmethod
    def mix_materials(materials, fractions, method, name=None):
        result = _MaterialStub(name=name)
        try:
            result.density = sum(m.density * f for m, f in zip(materials, fractions))
        except Exception:
            result.density = 0.0
        return result


class _MaterialsStub:
    def append(self, mat):
        pass

    def extend(self, mats):
        pass


class _OpenMCStub(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Material = _MaterialStub
        self.Materials = _MaterialsStub


_openmc_stub = _OpenMCStub()
_openmc_stub.Material = _MaterialStub
_openmc_stub.Materials = _MaterialsStub

for _mod in ['openmc', 'openmc.deplete', 'openmc.mgxs']:
    sys.modules[_mod] = _openmc_stub

sys.modules['watts'] = MagicMock()

# ---------------------------------------------------------------------------
# Standard imports (after stubs are in place)
# ---------------------------------------------------------------------------
import io
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

from reactor_config import build_params, SubcriticalError
from cost.cost_estimation import bottom_up_cost_estimate, transform_dataframe
from cost.cost_drivers import cost_drivers_estimate, is_double_digit_excluding_multiples_of_10

# ---------------------------------------------------------------------------
# Performance patches: cache Excel reads that would otherwise repeat on every run.
#
# Bottleneck 1 — calculate_inflation_multiplier calls pd.read_excel once per
#   cost-database row.  Fixed with lru_cache (keyed on file + year + type).
#
# Bottleneck 2 — escalate_cost_database calls pd.read_excel twice (Cost Database
#   sheet + Economics Parameters sheet) on every bottom_up_cost_estimate call.
#   We patch pd.read_excel itself with an lru_cache so the same file+sheet
#   combination is only parsed once per process lifetime.
# ---------------------------------------------------------------------------
import functools
import cost.cost_escalation as _ce

_orig_inflation = _ce.calculate_inflation_multiplier

@functools.lru_cache(maxsize=512)
def _cached_inflation_multiplier(file_path, base_dollar_year, cost_type, escalation_year):
    return _orig_inflation(file_path, base_dollar_year, cost_type, escalation_year)

_ce.calculate_inflation_multiplier = _cached_inflation_multiplier

# Cache pd.read_excel for the cost database (file content never changes at runtime).
import pandas as _pd_orig
_orig_read_excel = _pd_orig.read_excel

@functools.lru_cache(maxsize=32)
def _cached_read_excel(file_path, sheet_name):
    return _orig_read_excel(file_path, sheet_name=sheet_name)

def _patched_read_excel(file_path, sheet_name=0, **kwargs):
    # Only cache reads of the Cost_Database.xlsx; pass everything else through.
    if isinstance(file_path, str) and file_path.endswith('Cost_Database.xlsx') and not kwargs:
        return _cached_read_excel(file_path, sheet_name).copy()
    return _orig_read_excel(file_path, sheet_name=sheet_name, **kwargs)

_pd_orig.read_excel = _patched_read_excel
_ce.pd.read_excel = _patched_read_excel

# ---------------------------------------------------------------------------
# Reactor metadata: full names and design images
# ---------------------------------------------------------------------------
_REACTOR_LABELS = {
    'LTMR': 'Liquid Metal Microreactor (LTMR)',
    'GCMR': 'Gas Cooled Microreactor (GCMR)',
    'HPMR': 'Heat Pipe Microreactor (HPMR)',
}
_LABEL_TO_KEY = {v: k for k, v in _REACTOR_LABELS.items()}

_ASSETS = os.path.join(_repo_root, 'assets', 'Ref_openmc_2d_designs')

_REACTOR_IMAGES = {
    'LTMR': {
        'main': (
            os.path.join(_ASSETS, 'LTMR_core.png'),
            'LTMR core cross-section — hexagonal arrangement of TRIGA-type U-ZrH fuel '
            'pins and ZrH moderator pins cooled by NaK liquid metal, surrounded by a '
            'graphite radial reflector with control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'LTMR_fuel_pin_universe.png'),
                'LTMR fuel pin cross-section — from center outward: zirconium cladding, '
                'gap, U-ZrH fuel meat, gap, and SS304 outer cladding.',
            ),
            (
                os.path.join(_ASSETS, 'LTMR_moderator_pin_universe.png'),
                'LTMR moderator pin cross-section — ZrH hydrogen moderator encased in '
                'SS304 cladding, interspersed between fuel pins to thermalize neutrons.',
            ),
        ],
    },
    'GCMR': {
        'main': (
            os.path.join(_ASSETS, 'GCMR_Core.png'),
            'GCMR core cross-section — hexagonal fuel assemblies containing TRISO fuel '
            'compacts arranged in a honeycomb pattern, cooled by helium gas flowing '
            'through dedicated coolant channels, with graphite reflector and control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'GCMR_Core (zoomed in).png'),
                'GCMR core zoomed — detailed view of the hexagonal assembly arrangement '
                'showing the compact-to-assembly packing and inter-assembly helium flow paths.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_Fuel Assembly.png'),
                'GCMR fuel assembly cross-section — TRISO fuel compacts, helium coolant '
                'channels, and ZrH moderator booster pins embedded in a graphite matrix.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_Fuel Assembly (zoomed in).png'),
                'GCMR fuel assembly zoomed — individual TRISO particles visible within '
                'the graphite fuel compact at the target packing fraction.',
            ),
            (
                os.path.join(_ASSETS, 'GCMR_TRISO_Particle.png'),
                'TRISO fuel particle — multi-layer design with UN fuel kernel, buffer '
                'graphite, inner PyC, SiC pressure vessel, and outer PyC coating that '
                'retains fission products up to ~1600 °C.',
            ),
        ],
    },
    'HPMR': {
        'main': (
            os.path.join(_ASSETS, 'HPMR_core.png'),
            'HPMR core cross-section — monolithic graphite/metal core with hexagonal '
            'fuel assemblies and embedded alkali-metal heat pipes that passively transfer '
            'heat to the secondary side, with graphite reflector and control drums.',
        ),
        'details': [
            (
                os.path.join(_ASSETS, 'HPMR_fuel_assembly.png'),
                'HPMR fuel assembly cross-section — TRISO fuel pins and heat pipes '
                'arranged in a hexagonal pattern within the graphite monolith block.',
            ),
            (
                os.path.join(_ASSETS, 'HPMR_fuel_pin_universe.png'),
                'HPMR fuel pin cross-section — homogenized TRISO fuel region surrounded '
                'by a thin helium gap within the monolith.',
            ),
            (
                os.path.join(_ASSETS, 'HPMR_heatpipe_universe.png'),
                'HPMR heat pipe cross-section — working fluid region and outer cladding '
                'that passively carries heat from the core to the power conversion system '
                'with no moving parts.',
            ),
        ],
    },
}

# ---------------------------------------------------------------------------
# Cached cost estimate (module-level so cache persists across reruns)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _run_estimate(reactor_type, power_mwt, enrichment, interest_rate,
                  construction_duration, debt_to_equity, operation_mode,
                  emergency_shutdowns, startup_duration, startup_duration_refueling,
                  tax_credit_type, tax_credit_value):
    overrides = {
        'Interest Rate': interest_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': 2024,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
        'Startup Duration after Refueling': startup_duration_refueling,
    }
    if tax_credit_type == 'PTC':
        overrides['PTC credit value'] = tax_credit_value
        overrides['PTC credit period'] = 10
        overrides['Tax Rate'] = 0.21
    elif tax_credit_type == 'ITC':
        overrides['ITC credit level'] = tax_credit_value

    p = build_params(reactor_type, power_mwt, enrichment, overrides)
    raw_df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)
    enriched_df = cost_drivers_estimate(raw_df, p)
    return transform_dataframe(enriched_df), enriched_df, p


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _get_mean_std(df, account, which='FOAK'):
    row = df[df['Account'] == account]
    if row.empty:
        return float('nan'), float('nan')
    mean_prefix = 'FOAK Estimated Cost (' if which == 'FOAK' else 'NOAK Estimated Cost ('
    std_prefix  = 'FOAK Estimated Cost std (' if which == 'FOAK' else 'NOAK Estimated Cost std ('
    mean_cols = [c for c in df.columns if c.startswith(mean_prefix)]
    std_cols  = [c for c in df.columns if c.startswith(std_prefix)]
    try:
        mean = float(row[mean_cols[0]].iloc[0]) if mean_cols else float('nan')
    except (TypeError, ValueError):
        mean = float('nan')
    try:
        std = float(row[std_cols[0]].iloc[0]) if std_cols else float('nan')
    except (TypeError, ValueError):
        std = float('nan')
    return mean, std


def _fmt_cost(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = round(mean / 1e6)
    if math.isnan(std) or std == 0:
        return f'${m}M'
    lo = round((mean - std) / 1e6)
    hi = round((mean + std) / 1e6)
    return f'${lo}M – ${hi}M'


def _fmt_lcoe(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = int(round(mean))
    if math.isnan(std) or std == 0:
        return f'${m}/MWh'
    lo = int(round(mean - std))
    hi = int(round(mean + std))
    return f'${lo} – ${hi}/MWh'


def _fmt_lcoh(mean, std):
    if math.isnan(mean):
        return 'N/A'
    m = int(round(mean))
    if math.isnan(std) or std == 0:
        return f'${m}/MWth'
    lo = int(round(mean - std))
    hi = int(round(mean + std))
    return f'${lo} – ${hi}/MWth'


def _get_lcof(df, which='FOAK'):
    mean_col = 'FOAK LCOE' if which == 'FOAK' else 'NOAK LCOE'
    std_col  = 'FOAK LCOE_std' if which == 'FOAK' else 'NOAK LCOE_std'
    if mean_col not in df.columns:
        return float('nan'), float('nan')
    mask = df['Account'].isin([25, 80])
    mean_vals = pd.to_numeric(df.loc[mask, mean_col], errors='coerce').dropna()
    mean = float(mean_vals.sum()) if len(mean_vals) > 0 else float('nan')
    if std_col not in df.columns:
        return mean, float('nan')
    std_vals = pd.to_numeric(df.loc[mask, std_col], errors='coerce').fillna(0)
    std = float(np.sqrt((std_vals ** 2).sum()))
    return mean, std


def _fmt_table_val(x):
    if x == '-' or x is None or x == '':
        return x
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v == 0:
        return '0'
    sign = '-' if v < 0 else ''
    a = abs(v)
    if a >= 1e7:
        return f'{sign}{round(a / 1e6):.0f}M'
    elif a >= 1e6:
        return f'{sign}{round(a / 1e6, 1)}M'
    elif a >= 1e5:
        return f'{sign}{round(a / 1e4) * 10:.0f}K'
    elif a >= 1e4:
        return f'{sign}{round(a / 1e3):.0f}K'
    elif a >= 1e3:
        return f'{sign}{round(a / 1e3, 1)}K'
    else:
        return f'{sign}{int(round(a))}'


# ---------------------------------------------------------------------------
# HTML/CSS card helpers
# ---------------------------------------------------------------------------
_CARD_COLORS = {
    'occ':  '#7c3aed',
    'tci':  '#0369a1',
    'lcoe': '#0891b2',
    'lcoh': '#059669',
    'lcof': '#d97706',
}

def _kpi_card(col, title, foak_val, noak_val, color='#1B4F8C'):
    col.markdown(
        f'''<div style="background:white;border-radius:14px;padding:1.1rem 1.25rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.07);border-top:3px solid {color};
                        min-height:110px;">
              <div style="font-size:0.68rem;font-weight:700;color:#9ca3af;
                          text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.65rem;">{title}</div>
              <div style="display:flex;align-items:baseline;gap:0.45rem;margin-bottom:0.45rem;">
                <span style="background:#fff3ed;color:#c84b1e;font-size:0.6rem;font-weight:800;
                             padding:0.12rem 0.38rem;border-radius:4px;letter-spacing:0.05em;
                             flex-shrink:0;">FOAK</span>
                <span style="font-size:1rem;font-weight:700;color:#111827;">{foak_val}</span>
              </div>
              <div style="height:1px;background:#f3f4f6;margin:0.3rem 0;"></div>
              <div style="display:flex;align-items:baseline;gap:0.45rem;margin-top:0.45rem;">
                <span style="background:#eff6ff;color:#1d4ed8;font-size:0.6rem;font-weight:800;
                             padding:0.12rem 0.38rem;border-radius:4px;letter-spacing:0.05em;
                             flex-shrink:0;">NOAK</span>
                <span style="font-size:1rem;font-weight:700;color:#111827;">{noak_val}</span>
              </div>
            </div>''',
        unsafe_allow_html=True,
    )


def _info_card(col, title, value, subtitle='', accent='#16a34a', bg='#f0fdf4', border='#bbf7d0'):
    sub_html = f'<div style="font-size:0.7rem;color:#6b7280;margin-top:0.2rem;">{subtitle}</div>' if subtitle else ''
    col.markdown(
        f'''<div style="background:{bg};border:1px solid {border};border-radius:14px;
                        padding:1.1rem 1.25rem;box-shadow:0 2px 8px rgba(0,0,0,0.04);min-height:80px;">
              <div style="font-size:0.68rem;font-weight:700;color:{accent};
                          text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.35rem;">{title}</div>
              <div style="font-size:1.35rem;font-weight:800;color:#111827;line-height:1.2;">{value}</div>
              {sub_html}
            </div>''',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── App background ── */
.stApp { background: #eef2f8 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(175deg, #0b1f3a 0%, #1a3d66 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #93c5fd !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] .stCaption p {
    color: #475569 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.6rem 0 !important;
}
section[data-testid="stSidebar"] button[kind="icon"],
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"],
section[data-testid="stSidebar"] .stTooltipIcon,
section[data-testid="stSidebar"] svg[data-icon="question-circle"] {
    color: #93c5fd !important;
    opacity: 1 !important;
    fill: #93c5fd !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #e05c2b 0%, #b84520 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(224,92,43,0.35) !important;
    letter-spacing: 0.02em !important;
    width: 100%;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #f06a35 0%, #c9531e 100%) !important;
    box-shadow: 0 6px 20px rgba(224,92,43,0.5) !important;
    transform: translateY(-1px);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.25rem;
    border-bottom: 2px solid #dde3ee;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    color: #64748b;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0.55rem 1.2rem;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1B4F8C !important;
    border-top: 1px solid #dde3ee !important;
    border-left: 1px solid #dde3ee !important;
    border-right: 1px solid #dde3ee !important;
    border-bottom: 2px solid white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: white;
    border-radius: 0 12px 12px 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: white;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1B4F8C, #1565c0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 10px rgba(27,79,140,0.3) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    box-shadow: 0 5px 15px rgba(27,79,140,0.45) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #1B4F8C !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
"""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='MOUSE — Microreactor Cost Estimator',
    page_icon='⚛',
    layout='wide',
)
st.markdown(f'<style>{_CSS}</style>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
        '<span style="font-size:2.2rem;">⚛</span>'
        '<div style="color:white;font-weight:800;font-size:1.1rem;letter-spacing:0.05em;margin-top:0.2rem;">MOUSE</div>'
        '<div style="color:#64748b;font-size:0.65rem;letter-spacing:0.06em;text-transform:uppercase;line-height:1.4;">'
        'Microreactor Optimization<br>Using Simulation &amp; Economics</div>'
        '<div style="margin-top:0.5rem;">'
        '<a href="https://github.com/IdahoLabResearch/MOUSE" target="_blank" '
        'style="color:#93c5fd;font-size:0.65rem;text-decoration:none;">⬡ GitHub Repo</a>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('**A — Reactor Design**')

    reactor_label = st.selectbox(
        'Reactor Type',
        options=list(_REACTOR_LABELS.values()),
        help='Select a microreactor design to estimate costs for.',
    )
    reactor_type = _LABEL_TO_KEY[reactor_label]

    enrichment = st.slider(
        'Enrichment (U-235 fraction)',
        min_value=0.05,
        max_value=0.1975,
        value=0.1975,
        step=0.0025,
        format='%.4f',
        help='U-235 enrichment fraction. Affects uranium masses and fuel lifetime via interpolation.',
    )
    st.caption(f'{enrichment * 100:.2f}% enriched')

    _power_defaults = {'LTMR': 20, 'GCMR': 15, 'HPMR': 7}
    _power_max = {'LTMR': 20, 'GCMR': 20, 'HPMR': 7}

    power_mwt = st.slider(
        'Thermal Power (MWt)',
        min_value=1,
        max_value=_power_max[reactor_type],
        value=_power_defaults[reactor_type],
        step=1,
        key=f'power_{reactor_type}',
        help='Thermal power output. Affects power-dependent params and fuel lifetime via interpolation.',
    )

    st.divider()
    st.markdown('**B — Operation Parameters**')

    _OPERATION_MODE_LABELS = {
        'Remotely Monitored': 'Autonomous',
        'On-Site Staffed':    'Non-Autonomous',
    }
    operation_mode_label = st.selectbox(
        'Operation Mode',
        options=list(_OPERATION_MODE_LABELS.keys()),
        help=(
            '**Remotely Monitored:** Operators monitor the reactor remotely and are '
            'required on-site only for emergencies or shutdown.\n\n'
            '**On-Site Staffed:** Operators must be physically present in the control room 24/7.'
        ),
    )
    operation_mode = _OPERATION_MODE_LABELS[operation_mode_label]
    emergency_shutdowns = st.number_input(
        'Emergency Shutdowns per Year',
        min_value=0.0, max_value=10.0, value=0.2, step=0.1, format='%.1f',
        help='Expected number of unplanned emergency shutdowns per year.',
    )
    startup_duration = st.number_input(
        'Startup Duration after Emergency Shutdown (days)',
        min_value=1, max_value=365, value=14, step=1,
        help='Days required to restart the reactor after an emergency shutdown.',
    )
    startup_duration_refueling = st.number_input(
        'Startup Duration after Refueling (days)',
        min_value=1, max_value=365, value=2, step=1,
        help='Days required to restart the reactor after a scheduled refueling outage.',
    )

    st.divider()
    st.markdown('**C — Economic Parameters**')

    interest_rate = st.number_input(
        'Interest Rate (%)',
        min_value=0.0, max_value=30.0, value=7.0, step=0.5, format='%.1f',
        help='Annual interest rate used in financing cost calculations.',
    )
    construction_duration = st.number_input(
        'Construction Duration (months)',
        min_value=1, max_value=120, value=12, step=1,
        help='Number of months from ground-break to commercial operation.',
    )
    debt_to_equity = st.slider(
        'Debt-to-Equity Ratio',
        min_value=0.0, max_value=5.0, value=1.0, step=0.1, format='%.1f',
        help='Ratio of debt to equity financing (e.g. 1.0 = equal debt and equity).',
    )

    st.markdown('**Government Subsidy (IRA Tax Credits)**')
    tax_credit_type = st.selectbox(
        'Tax Credit',
        options=['None', 'PTC', 'ITC'],
        index=0,
        help='None: no credit. PTC: Production Tax Credit (reduces LCOE). ITC: Investment Tax Credit (reduces OCC).',
    )
    tax_credit_value = None
    if tax_credit_type == 'PTC':
        tax_credit_value = st.selectbox(
            'PTC Credit Value ($/MWh)',
            options=[3.0, 3.3, 3.6, 15.0, 16.5, 18.0],
            index=3,
            format_func=lambda x: f'${x:.1f}/MWh',
            help='Total PTC value including any applicable IRA bonus multipliers.',
        )
    elif tax_credit_type == 'ITC':
        tax_credit_value = st.selectbox(
            'ITC Credit Level',
            options=[0.06, 0.30, 0.40, 0.50],
            index=1,
            format_func=lambda x: f'{x*100:.0f}%',
            help='ITC as a fraction of overnight capital cost (OCC).',
        )

    st.divider()
    run_button = st.button('⚡  Run Cost Estimate', type='primary', use_container_width=True)

    st.divider()
    st.markdown('**💬 Feedback**')
    st.caption('Help us improve MOUSE by sharing your thoughts.')
    st.link_button(
        '📝  Give Feedback',
        'https://qualtricsxm69xy9s7vm.qualtrics.com/jfe/form/SV_4Pb0vub9xCcsVV4',
        use_container_width=True,
    )
# ---------------------------------------------------------------------------
# Welcome banner (shown only before first run)
# ---------------------------------------------------------------------------
if not run_button:
    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        '''<div style="background:linear-gradient(135deg,#0b1f3a 0%,#1B4F8C 55%,#1e6fa8 100%);
                       border-radius:18px;padding:3rem 3rem 2.8rem;color:white;
                       margin-bottom:1.5rem;position:relative;overflow:hidden;">
             <div style="position:absolute;top:-60px;right:-60px;width:300px;height:300px;
                         border-radius:50%;background:rgba(255,255,255,0.04);"></div>
             <div style="position:absolute;right:3rem;top:50%;transform:translateY(-50%);
                         font-size:7rem;opacity:0.1;line-height:1;">⚛</div>
             <div style="font-size:0.75rem;opacity:0.65;margin-bottom:0.7rem;">
               Based on the open-source
               <a href="https://github.com/IdahoLabResearch/MOUSE" target="_blank"
                  style="color:#93c5fd;font-weight:600;text-decoration:none;">
                 IdahoLabResearch/MOUSE
               </a>
               repository on GitHub
             </div>
             <h1 style="font-size:2.4rem;font-weight:800;margin:0 0 0.3rem;color:white;line-height:1.15;">
               MOUSE
             </h1>
             <div style="font-size:1rem;font-weight:600;opacity:0.85;margin-bottom:0.9rem;color:white;">
               Microreactor Optimization Using Simulation and Economics
             </div>
             <p style="font-size:0.92rem;opacity:0.75;margin:0 0 1.8rem;max-width:620px;color:white;line-height:1.6;">
               MOUSE bridges nuclear microreactor design and economics by integrating core physics
               simulations (OpenMC), simplified balance-of-plant calculations, and detailed bottom-up
               cost estimation — enabling parametric optimization studies and uncertainty analysis
               for both <strong style="color:white;">First-of-a-Kind (FOAK)</strong> and
               <strong style="color:white;">Nth-of-a-Kind (NOAK)</strong> deployments.
               Cost estimation correlations derive from the MARVEL project and supplementary literature.
               All costs in <strong style="color:white;">2024 USD</strong>.
             </p>
             <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
               <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                           border-radius:10px;padding:0.7rem 1.2rem;">
                 <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                             opacity:0.6;margin-bottom:0.2rem;">Reactor Types</div>
                 <div style="font-weight:700;font-size:0.88rem;">LTMR · GCMR · HPMR</div>
               </div>
               <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                           border-radius:10px;padding:0.7rem 1.2rem;">
                 <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                             opacity:0.6;margin-bottom:0.2rem;">Power Range</div>
                 <div style="font-weight:700;font-size:0.88rem;">1 – 20 MWt</div>
               </div>
               <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                           border-radius:10px;padding:0.7rem 1.2rem;">
                 <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                             opacity:0.6;margin-bottom:0.2rem;">Enrichment</div>
                 <div style="font-weight:700;font-size:0.88rem;">5 – 19.75% LEU+</div>
               </div>
               <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                           border-radius:10px;padding:0.7rem 1.2rem;">
                 <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                             opacity:0.6;margin-bottom:0.2rem;">Outputs</div>
                 <div style="font-weight:700;font-size:0.88rem;">OCC · TCI · LCOE · LCOH · LCOF</div>
               </div>
               <div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
                           border-radius:10px;padding:0.7rem 1.2rem;">
                 <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                             opacity:0.6;margin-bottom:0.2rem;">Analysis</div>
                 <div style="font-weight:700;font-size:0.88rem;">Parametric · Uncertainty · IRA Credits</div>
               </div>
             </div>
           </div>''',
        unsafe_allow_html=True,
    )

    # ── Caveats box ──────────────────────────────────────────────────────────
    st.markdown(
        '''<div style="background:#fffbeb;border:1.5px solid #f59e0b;border-radius:12px;
                       padding:1.2rem 1.5rem;margin-bottom:1rem;">
             <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
               <span style="font-size:1.2rem;">⚠️</span>
               <span style="font-size:0.82rem;font-weight:800;color:#92400e;
                            text-transform:uppercase;letter-spacing:0.07em;">
                 Important Caveats — Please Read Before Use
               </span>
             </div>
             <ul style="margin:0;padding-left:1.2rem;color:#78350f;font-size:0.84rem;line-height:1.75;">
               <li><strong>Work in progress:</strong> This application is still under active development.
                   Features, results, and interfaces may change without notice.</li>
               <li><strong>Pre-conceptual designs:</strong> The three reactor designs (LTMR, GCMR, HPMR)
                   are pre-conceptual. They have not been fully optimized and do not represent
                   final or licensed configurations.</li>
               <li><strong>Incomplete information:</strong> Cost estimates were developed with incomplete
                   engineering and procurement data. Significant uncertainties exist across all
                   accounts, particularly for novel or first-of-a-kind components.</li>
               <li><strong>Not for investment decisions:</strong> These estimates must <em>not</em> be
                   used as the basis for financial, investment, or procurement decisions. They are
                   intended solely for research, screening, and comparative analysis.</li>
             </ul>
           </div>''',
        unsafe_allow_html=True,
    )

    st.info('Configure your reactor in the sidebar, then click **⚡ Run Cost Estimate** to begin.')
    st.stop()

# ---------------------------------------------------------------------------
# Run cost estimate
# ---------------------------------------------------------------------------
with st.spinner('Running cost estimate…'):
    try:
        display_df, enriched_df, params = _run_estimate(
            reactor_type, power_mwt, enrichment,
            interest_rate / 100.0, construction_duration, debt_to_equity,
            operation_mode, emergency_shutdowns, startup_duration, startup_duration_refueling,
            tax_credit_type, tax_credit_value,
        )
    except SubcriticalError as exc:
        st.error('### ⚠ Reactor is Subcritical')
        st.warning(str(exc))
        ca, cb, cc = st.columns(3)
        _info_card(ca, 'Fuel Lifetime', '0 days', accent='#dc2626', bg='#fef2f2', border='#fecaca')
        _info_card(cb, 'Thermal Power', f'{power_mwt} MWt',  accent='#9a3412', bg='#fff7ed', border='#fed7aa')
        _info_card(cc, 'Enrichment',    f'{enrichment*100:.2f}%', accent='#9a3412', bg='#fff7ed', border='#fed7aa')
        st.info('No cost estimate is available for a subcritical operating point. '
                'Try reducing the power or increasing the enrichment.')
        st.stop()
    except Exception as exc:
        st.error(f'Cost estimation failed: {exc}')
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ---------------------------------------------------------------------------
# Result hero banner
# ---------------------------------------------------------------------------
_credit_badge = ''
if tax_credit_type == 'PTC':
    _credit_badge = f'<span style="background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.35);border-radius:999px;font-size:0.68rem;font-weight:700;padding:0.2rem 0.7rem;letter-spacing:0.06em;margin-left:0.6rem;">PTC ${tax_credit_value}/MWh</span>'
elif tax_credit_type == 'ITC':
    _credit_badge = f'<span style="background:rgba(255,255,255,0.2);border:1px solid rgba(255,255,255,0.35);border-radius:999px;font-size:0.68rem;font-weight:700;padding:0.2rem 0.7rem;letter-spacing:0.06em;margin-left:0.6rem;">ITC {int(tax_credit_value*100)}%</span>'

st.markdown(
    f'''<div style="background:linear-gradient(135deg,#0b1f3a 0%,#1B4F8C 55%,#1e6fa8 100%);
                   border-radius:16px;padding:1.8rem 2.2rem;color:white;
                   margin-bottom:1.5rem;position:relative;overflow:hidden;">
         <div style="position:absolute;right:2.5rem;top:50%;transform:translateY(-50%);
                     font-size:5.5rem;opacity:0.1;line-height:1;">⚛</div>
         <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                     text-transform:uppercase;opacity:0.55;margin-bottom:0.5rem;">
           Cost Estimate Result
         </div>
         <h2 style="font-size:1.65rem;font-weight:800;margin:0 0 0.9rem;color:white;line-height:1.2;">
           {reactor_label} {_credit_badge}
         </h2>
         <div style="display:flex;gap:2rem;flex-wrap:wrap;">
           <div>
             <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Thermal Power</div>
             <div style="font-weight:700;font-size:0.95rem;">{power_mwt} MWt</div>
           </div>
           <div>
             <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Enrichment</div>
             <div style="font-weight:700;font-size:0.95rem;">{enrichment*100:.2f}%</div>
           </div>
           <div>
             <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Interest Rate</div>
             <div style="font-weight:700;font-size:0.95rem;">{interest_rate:.1f}%</div>
           </div>
           <div>
             <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">Operation Mode</div>
             <div style="font-weight:700;font-size:0.95rem;">{operation_mode}</div>
           </div>
           <div>
             <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;opacity:0.55;">All costs in</div>
             <div style="font-weight:700;font-size:0.95rem;">2024 USD</div>
           </div>
         </div>
       </div>''',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Collect all summary values
# ---------------------------------------------------------------------------
fuel_lifetime = params.get('Fuel Lifetime', float('nan'))
power_mwe     = params.get('Power MWe', float('nan'))

if tax_credit_type == 'ITC':
    occ_account  = 'OCC (ITC-adjusted)'
    tci_account  = 'TCI (ITC-adjusted)'
    lcoe_account = 'LCOE (ITC-adjusted)'
elif tax_credit_type == 'PTC':
    occ_account  = 'OCC'
    tci_account  = 'TCI'
    lcoe_account = 'LCOE with PTC'
else:
    occ_account  = 'OCC'
    tci_account  = 'TCI'
    lcoe_account = 'LCOE'

occ_f,  occ_f_std  = _get_mean_std(display_df, occ_account,  'FOAK')
tci_f,  tci_f_std  = _get_mean_std(display_df, tci_account,  'FOAK')
lcoe_f, lcoe_f_std = _get_mean_std(display_df, lcoe_account, 'FOAK')
lcoh_f, lcoh_f_std = _get_mean_std(display_df, 'LCOH',       'FOAK')
lcof_f, lcof_f_std = _get_lcof(enriched_df, 'FOAK')

occ_n,  occ_n_std  = _get_mean_std(display_df, occ_account,  'NOAK')
tci_n,  tci_n_std  = _get_mean_std(display_df, tci_account,  'NOAK')
lcoe_n, lcoe_n_std = _get_mean_std(display_df, lcoe_account, 'NOAK')
lcoh_n, lcoh_n_std = _get_mean_std(display_df, 'LCOH',       'NOAK')
lcof_n, lcof_n_std = _get_lcof(enriched_df, 'NOAK')

_fl_str = f'{fuel_lifetime / 365:.1f} yrs' if not math.isnan(float(fuel_lifetime)) else 'N/A'
_fl_days = f'{int(fuel_lifetime):,} days' if not math.isnan(float(fuel_lifetime)) else ''

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
# ── Persistent caveat reminder (compact) ────────────────────────────────────
st.markdown(
    '''<div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
                   padding:0.65rem 1rem;margin-bottom:1.2rem;
                   display:flex;align-items:flex-start;gap:0.6rem;">
         <span style="font-size:1rem;flex-shrink:0;margin-top:0.05rem;">⚠️</span>
         <span style="font-size:0.78rem;color:#92400e;line-height:1.55;">
           <strong>Caveats:</strong> This app is under active development. Reactor designs are
           pre-conceptual and not fully optimized. Cost estimates were produced with incomplete
           information and <strong>must not be used for investment or procurement decisions</strong>.
           Results are intended for research and comparative screening only.
         </span>
       </div>''',
    unsafe_allow_html=True,
)

tab_summary, tab_drivers, tab_table = st.tabs([
    '📊  Summary',
    '📈  Cost Drivers',
    '📋  Full Breakdown',
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ═══════════════════════════════════════════════════════════════
with tab_summary:

    # ── Reactor image ─────────────────────────────────────────
    img_col, info_col = st.columns([1, 1], gap='large')

    with img_col:
        main_img, main_caption = _REACTOR_IMAGES[reactor_type]['main']
        st.image(main_img, use_container_width=True)
        st.caption(main_caption)
        with st.expander('View detailed cross-sections'):
            for img_path, img_caption in _REACTOR_IMAGES[reactor_type]['details']:
                st.image(img_path, use_container_width=True)
                st.caption(img_caption)

    with info_col:
        # ── Info cards: Power & Fuel Lifetime ──────────────────
        ic1, ic2 = st.columns(2)
        _info_card(ic1, 'Electric Power Output', f'{power_mwe:.1f} MWe',
                   subtitle=f'Thermal input: {power_mwt} MWt')
        _info_card(ic2, 'Fuel Lifetime', _fl_str, subtitle=_fl_days)

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        # ── OCC ─────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:0.6rem;">Capital Costs</div>',
            unsafe_allow_html=True,
        )
        cc1, cc2 = st.columns(2)
        _kpi_card(cc1, 'Overnight Capital Cost (OCC)',
                  _fmt_cost(occ_f, occ_f_std), _fmt_cost(occ_n, occ_n_std),
                  color=_CARD_COLORS['occ'])
        _kpi_card(cc2, 'Total Capital Investment (TCI)',
                  _fmt_cost(tci_f, tci_f_std), _fmt_cost(tci_n, tci_n_std),
                  color=_CARD_COLORS['tci'])

        st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)

        # ── Levelized costs ──────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
            'letter-spacing:0.09em;margin-bottom:0.6rem;">Levelized Costs</div>',
            unsafe_allow_html=True,
        )
        lc1, lc2, lc3 = st.columns(3)
        _kpi_card(lc1, 'LCOE ($/MWh)',
                  _fmt_lcoe(lcoe_f, lcoe_f_std), _fmt_lcoe(lcoe_n, lcoe_n_std),
                  color=_CARD_COLORS['lcoe'])
        _kpi_card(lc2, 'LCOH ($/MWth)',
                  _fmt_lcoh(lcoh_f, lcoh_f_std), _fmt_lcoh(lcoh_n, lcoh_n_std),
                  color=_CARD_COLORS['lcoh'])
        _kpi_card(lc3, 'LCOF — Fuel Only ($/MWh)',
                  _fmt_lcoe(lcof_f, lcof_f_std), _fmt_lcoe(lcof_n, lcof_n_std),
                  color=_CARD_COLORS['lcof'])

    # ── FOAK / NOAK legend note ────────────────────────────────
    st.markdown(
        '<div style="margin-top:1.2rem;padding:0.7rem 1rem;background:#f8fafc;'
        'border-radius:8px;border:1px solid #e2e8f0;font-size:0.78rem;color:#64748b;">'
        '<span style="background:#fff3ed;color:#c84b1e;font-size:0.62rem;font-weight:800;'
        'padding:0.1rem 0.4rem;border-radius:4px;margin-right:0.4rem;">FOAK</span>'
        'First-of-a-Kind — includes learning curve and contingency premiums for an initial deployment.&emsp;'
        '<span style="background:#eff6ff;color:#1d4ed8;font-size:0.62rem;font-weight:800;'
        'padding:0.1rem 0.4rem;border-radius:4px;margin-right:0.4rem;">NOAK</span>'
        'Nth-of-a-Kind — reflects mature, serial production cost reductions at scale.'
        '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — COST DRIVERS
# ═══════════════════════════════════════════════════════════════
with tab_drivers:
    _drv = enriched_df[enriched_df['Account'].apply(
        is_double_digit_excluding_multiples_of_10)].copy()
    _drv = _drv.sort_values('FOAK LCOE', ascending=False)
    _drv = _drv[_drv['FOAK LCOE'] >= 5]

    if _drv.empty:
        st.info('No accounts with FOAK LCOE ≥ 5 $/MWh found.')
    else:
        st.markdown(
            '<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
            'Per-account LCOE contributions ($/MWh) sorted by FOAK impact. '
            'Error bars show ±1 standard deviation across Monte Carlo samples.</p>',
            unsafe_allow_html=True,
        )

        bar_width = 0.38
        r1 = np.arange(len(_drv))
        r2 = r1 + bar_width

        fig, ax = plt.subplots(figsize=(max(11, len(_drv) * 1.5), 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        foak_err = _drv['FOAK LCOE_std'] if 'FOAK LCOE_std' in _drv.columns else None
        noak_err = _drv['NOAK LCOE_std'] if 'NOAK LCOE_std' in _drv.columns else None

        bars_f = ax.bar(r1, _drv['FOAK LCOE'], width=bar_width,
                        color='#E05C2B', edgecolor='white', linewidth=0.6,
                        label='FOAK', zorder=3,
                        yerr=foak_err, capsize=5,
                        error_kw=dict(elinewidth=1.5, ecolor='#b84520', capthick=1.5))
        bars_n = ax.bar(r2, _drv['NOAK LCOE'], width=bar_width,
                        color='#1B7FBD', edgecolor='white', linewidth=0.6,
                        label='NOAK', zorder=3,
                        yerr=noak_err, capsize=5,
                        error_kw=dict(elinewidth=1.5, ecolor='#1155aa', capthick=1.5))

        ax.set_xticks(r1 + bar_width / 2)
        ax.set_xticklabels(_drv['Account Title'], rotation=40, ha='right',
                           fontsize=10, color='#374151')
        ax.set_ylabel('LCOE Contribution ($/MWh)', fontsize=11, color='#374151')
        ax.yaxis.set_tick_params(labelcolor='#374151', labelsize=10)
        ax.set_xlim(-0.3, len(_drv) - 0.2)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='#d1d5db', zorder=0)
        ax.set_axisbelow(True)

        legend = ax.legend(fontsize=11, frameon=True, framealpha=1,
                           edgecolor='#e5e7eb', facecolor='white',
                           loc='upper right', handlelength=1.5)
        legend.get_frame().set_linewidth(0.8)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — FULL BREAKDOWN
# ═══════════════════════════════════════════════════════════════
with tab_table:
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;margin-bottom:1rem;">'
        'Full cost breakdown by Code of Accounts. Highlighted rows are parent accounts. '
        'All dollar values in 2024 USD.</p>',
        unsafe_allow_html=True,
    )

    # Only show cost columns in the table (drop per-account LCOE contribution cols)
    _display_cols = [c for c in display_df.columns
                     if c in ('Account', 'Account Title') or 'Estimated Cost' in str(c)]
    table_df = display_df[_display_cols].copy()

    def _highlight_parents(row):
        acct = row.get('Account', None)
        try:
            if float(acct) == int(float(acct)) and float(acct) % 10 == 0:
                return ['background-color:#f0f4fa;font-weight:600'] * len(row)
        except (TypeError, ValueError):
            pass
        return [''] * len(row)

    _cost_cols = [c for c in table_df.columns if 'Estimated Cost' in str(c)]
    styled = (
        table_df.style
        .apply(_highlight_parents, axis=1)
        .format(_fmt_table_val, subset=_cost_cols, na_rep='-')
    )
    st.dataframe(styled, use_container_width=True, height=580)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # Download (full df including LCOE contribution cols)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        display_df.to_excel(writer, index=False, sheet_name='Cost Estimate')
    buffer.seek(0)

    dl_col, _ = st.columns([1, 3])
    dl_col.download_button(
        label='⬇  Download Full Excel',
        data=buffer.getvalue(),
        file_name=f'MOUSE_cost_estimate_{reactor_type}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True,
    )
