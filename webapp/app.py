# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
MOUSE Streamlit Web App
Microreactor Online Unified Simulation Engine — cost estimation without OpenMC.

Run from the MOUSE repo root:
    streamlit run webapp/app.py
"""

# ---------------------------------------------------------------------------
# Ensure the MOUSE repo root is in sys.path so all MOUSE modules resolve.
# This is needed when Streamlit adds only the script directory (webapp/) to
# sys.path but core_design/, cost/, etc., live one level up.
# ---------------------------------------------------------------------------
import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# IMPORTANT: Stub openmc and watts BEFORE any MOUSE import.
# core_design/utils.py, drums.py, and openmc_materials_database.py all import
# openmc at the top level. We replace them with lightweight stubs so the
# pure-math functions work without an actual OpenMC installation.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock


class _MaterialStub:
    """Minimal stub for openmc.Material that stores density correctly."""

    def __init__(self, name=None, temperature=None):
        self.name = name
        self.temperature = temperature
        self.density = 0.0  # default; overwritten by set_density

    def set_density(self, units, value):
        # Store the raw value. For g/cm3 materials this is the true density.
        # For atom/b-cm materials (homog_TRISO, heatpipe) the value is never
        # used in mass calculations so correctness is not needed.
        self.density = value

    def add_nuclide(self, *args, **kwargs):
        pass

    def add_element(self, *args, **kwargs):
        pass

    def add_s_alpha_beta(self, *args, **kwargs):
        pass

    @staticmethod
    def mix_materials(materials, fractions, method, name=None):
        """Return a stub material with a weighted-average density."""
        result = _MaterialStub(name=name)
        try:
            result.density = sum(m.density * f for m, f in zip(materials, fractions))
        except Exception:
            result.density = 0.0
        return result


class _MaterialsStub:
    """Minimal stub for openmc.Materials."""

    def append(self, mat):
        pass

    def extend(self, mats):
        pass


class _OpenMCStub(MagicMock):
    """openmc module stub — Material and Materials have real implementations."""

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
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import streamlit as st

from reactor_config import build_params, SubcriticalError
from cost.cost_estimation import bottom_up_cost_estimate, transform_dataframe

# ---------------------------------------------------------------------------
# Performance patch: cache the per-row Excel read inside calculate_inflation_multiplier.
# Without this, pd.read_excel is called once per cost-database row on every run.
# ---------------------------------------------------------------------------
import functools
import cost.cost_escalation as _ce

_orig_inflation = _ce.calculate_inflation_multiplier

@functools.lru_cache(maxsize=512)
def _cached_inflation_multiplier(file_path, base_dollar_year, cost_type, escalation_year):
    return _orig_inflation(file_path, base_dollar_year, cost_type, escalation_year)

_ce.calculate_inflation_multiplier = _cached_inflation_multiplier


@st.cache_data(show_spinner=False)
def _run_estimate(reactor_type, power_mwt, enrichment, interest_rate,
                  construction_duration, debt_to_equity, operation_mode,
                  emergency_shutdowns, startup_duration):
    overrides = {
        'Interest Rate': interest_rate,
        'Construction Duration': construction_duration,
        'Debt To Equity Ratio': debt_to_equity,
        'Escalation Year': 2024,
        'Operation Mode': operation_mode,
        'Emergency Shutdowns Per Year': emergency_shutdowns,
        'Startup Duration after Emergency Shutdown': startup_duration,
    }
    p = build_params(reactor_type, power_mwt, enrichment, overrides)
    df = bottom_up_cost_estimate('cost/Cost_Database.xlsx', p)
    return transform_dataframe(df), p

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='MOUSE — Microreactor Cost Estimator',
    page_icon='⚛',
    layout='wide',
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title('MOUSE — Microreactor Cost Estimator')
st.caption(
    'Microreactor Online Unified Simulation Engine · '
    'Bottom-up capital & levelized cost estimates for LTMR, GCMR, and HPMR designs.'
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header('Inputs')

    reactor_type = st.selectbox(
        'Reactor Type',
        options=['LTMR', 'GCMR', 'HPMR'],
        help='LTMR = Liquid Metal Thermal, GCMR = Gas Cooled (Design A), HPMR = Heat Pipe',
    )

    st.markdown('**Core Design Parameters**')

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

    # Default power per reactor type; unique key resets slider when reactor changes
    _power_defaults = {'LTMR': 20, 'GCMR': 15, 'HPMR': 7}
    power_mwt = st.slider(
        'Thermal Power (MWt)',
        min_value=1,
        max_value=20,
        value=_power_defaults[reactor_type],
        step=1,
        key=f'power_{reactor_type}',
        help='Thermal power output. Affects power-dependent params and fuel lifetime via interpolation.',
    )

    st.divider()
    st.markdown('**Economic Parameters**')

    interest_rate = st.number_input(
        'Interest Rate (%)',
        min_value=0.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
        format='%.1f',
        help='Annual interest rate used in financing cost calculations.',
    )

    construction_duration = st.number_input(
        'Construction Duration (months)',
        min_value=1,
        max_value=120,
        value=12,
        step=1,
        help='Number of months from ground-break to commercial operation.',
    )

    debt_to_equity = st.slider(
        'Debt-to-Equity Ratio',
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format='%.2f',
        help='Fraction of capital funded by debt (0 = all equity, 1 = all debt).',
    )

    operation_mode = st.selectbox(
        'Operation Mode',
        options=['Autonomous', 'Non-Autonomous'],
        help='Autonomous = minimal on-site operators; Non-Autonomous = full staffing.',
    )

    emergency_shutdowns = st.number_input(
        'Emergency Shutdowns per Year',
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1,
        format='%.1f',
        help='Expected number of unplanned emergency shutdowns per year.',
    )

    startup_duration = st.number_input(
        'Startup Duration after Emergency Shutdown (days)',
        min_value=1,
        max_value=365,
        value=14,
        step=1,
        help='Days required to restart the reactor after an emergency shutdown.',
    )

    run_button = st.button('Run Cost Estimate', type='primary', use_container_width=True)

# ---------------------------------------------------------------------------
# Placeholder for results
# ---------------------------------------------------------------------------
if not run_button:
    st.info('Configure inputs in the sidebar and click **Run Cost Estimate** to begin.')
    st.stop()

# ---------------------------------------------------------------------------
# Build params and run cost estimate
# ---------------------------------------------------------------------------
with st.spinner('Running cost estimate…'):
    try:
        display_df, params = _run_estimate(
            reactor_type, power_mwt, enrichment,
            interest_rate / 100.0, construction_duration, debt_to_equity,
            operation_mode, emergency_shutdowns, startup_duration,
        )
    except SubcriticalError as exc:
        st.warning('### Reactor is Subcritical')
        st.error(str(exc))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric('Fuel Lifetime', '0 days')
        col_b.metric('Thermal Power', f'{power_mwt} MWt')
        col_c.metric('Enrichment', f'{enrichment * 100:.2f}%')
        st.info(
            'No cost estimate is available for a subcritical operating point. '
            'Try reducing the power or increasing the enrichment.'
        )
        st.stop()
    except Exception as exc:
        st.error(f'Cost estimation failed: {exc}')
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
st.subheader('Summary')

# Find key summary rows in the result dataframe.
# OCC, TCI, and LCOE are appended as string accounts by cost_estimation.py.
def _get_value(df, account):
    """Return the FOAK cost for a given account label, or NaN."""
    row = df[df['Account'] == account]
    if row.empty:
        return float('nan')
    foak_cols = [c for c in df.columns if 'FOAK' in c and 'std' not in c]
    if not foak_cols:
        return float('nan')
    val = row[foak_cols[0]].iloc[0]
    try:
        return float(val)
    except (TypeError, ValueError):
        return float('nan')


occ_val = _get_value(display_df, 'OCC')
tci_val = _get_value(display_df, 'TCI')
lcoe_val = _get_value(display_df, 'LCOE')
power_mwe = params.get('Power MWe', float('nan'))

col1, col2, col3, col4 = st.columns(4)
col1.metric('OCC (Overnight Capital Cost)', f'${occ_val:,.0f}' if not pd.isna(occ_val) else 'N/A')
col2.metric('TCI (Total Capital Investment)', f'${tci_val:,.0f}' if not pd.isna(tci_val) else 'N/A')
col3.metric('LCOE', f'${lcoe_val:,.1f} /MWh' if not pd.isna(lcoe_val) else 'N/A')
col4.metric('Power Output', f'{power_mwe:.2f} MWe')

# ---------------------------------------------------------------------------
# Full cost table
# ---------------------------------------------------------------------------
st.subheader('Detailed Cost Breakdown')

# Highlight parent-level accounts (Level 0 accounts are integers like 10, 20, …)
def _highlight_parents(row):
    acct = row.get('Account', None)
    try:
        if float(acct) == int(float(acct)) and float(acct) % 10 == 0:
            return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
    except (TypeError, ValueError):
        pass
    return [''] * len(row)


styled = display_df.style.apply(_highlight_parents, axis=1)
st.dataframe(styled, use_container_width=True, height=600)

# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------
st.subheader('Export Results')

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    display_df.to_excel(writer, index=False, sheet_name='Cost Estimate')
buffer.seek(0)

st.download_button(
    label='Download Excel',
    data=buffer.getvalue(),
    file_name=f'MOUSE_cost_estimate_{reactor_type}.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
)
