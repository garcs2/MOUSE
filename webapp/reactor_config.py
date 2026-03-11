# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
reactor_config.py — Builds a fully-populated params dict for LTMR, GCMR, or HPMR
without running OpenMC. OpenMC outputs (Fuel Lifetime, Mass U235, Mass U238) are
interpolated from a pre-computed parametric study CSV keyed by reactor type,
enrichment, and thermal power.
"""

import os
import numpy as np
import pandas as pd

# openmc and watts must already be stubbed in sys.modules before this module is imported.
from core_design.utils import (
    calculate_lattice_radius,
    calculate_pins_in_assembly,
    calculate_heat_flux,
    calculate_heat_flux_TRISO,
    calculate_number_fuel_elements_hpmr,
    number_of_heatpipes_hmpr,
)
from core_design.drums import (
    calculate_drums_volumes_and_masses,
    calculate_reflector_mass_LTMR,
    calculate_reflector_mass_GCMR,
    calculate_moderator_mass_GCMR,
    calculate_reflector_and_moderator_mass_HPMR,
    calculate_moderator_mass,  # used by LTMR
)
from core_design.pins_arrangement import LTMR_pins_arrangement
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
# BOP.py re-exports everything from tools.py and adds its own functions
from reactor_engineering_evaluation.BOP import (
    calculate_heat_exchanger_mass,
    calculate_primary_pump_mechanical_power,
)
from reactor_engineering_evaluation.tools import (
    mass_flow_rate,
    compressor_power,
    GCMR_integrated_heat_transfer_vessel,
    calculate_shielding_masses,
)
from reactor_engineering_evaluation.vessels_calcs import vessels_specs

# ---------------------------------------------------------------------------
# CSV-based OpenMC results — interpolated from parametric study table
# ---------------------------------------------------------------------------
_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets', 'Ref_Results', 'parametric_study_power_and_enrichment.csv',
)
_lookup_df = None


class SubcriticalError(Exception):
    """Raised when the interpolated fuel lifetime is zero (subcritical core)."""
    pass


def _load_lookup():
    global _lookup_df
    if _lookup_df is None:
        df = pd.read_csv(_CSV_PATH)
        df.columns = df.columns.str.strip()
        df['reactor type'] = df['reactor type'].str.strip()
        _lookup_df = df.dropna(subset=['reactor type']).copy()
    return _lookup_df


def interpolate_openmc_results(reactor_type, power_mwt, enrichment):
    """
    Interpolate OpenMC results for a given (reactor_type, power_mwt, enrichment).

    - Mass U235 / Mass U238 depend only on enrichment → 1-D linear interpolation.
    - Fuel Lifetime depends on both enrichment and power → 2-D linear interpolation
      over the triangulated convex hull of available data points, with a
      nearest-neighbour fallback for points outside that hull.
    - Fuel Lifetime is clamped to >= 0.

    Returns a dict with keys 'Fuel Lifetime', 'Mass U235', 'Mass U238'.
    Raises SubcriticalError if Fuel Lifetime == 0.
    """
    from scipy.interpolate import griddata

    df = _load_lookup()
    rdf = df[df['reactor type'] == reactor_type].copy()

    # --- 1-D interpolation for uranium masses (enrichment only) ---
    enr_unique = np.sort(rdf['Enrichment'].unique())
    u235_vals = [rdf[rdf['Enrichment'] == e]['Mass U235'].mean() for e in enr_unique]
    u238_vals = [rdf[rdf['Enrichment'] == e]['Mass U238'].mean() for e in enr_unique]
    mass_u235 = float(np.interp(enrichment, enr_unique, u235_vals))
    mass_u238 = float(np.interp(enrichment, enr_unique, u238_vals))

    # --- 2-D interpolation for fuel lifetime (enrichment × power) ---
    points = rdf[['Enrichment', 'Power MWt']].values
    fl_vals = rdf['Fuel Lifetime'].values.astype(float)
    query = np.array([[enrichment, power_mwt]])

    fl = griddata(points, fl_vals, query, method='linear')[0]
    if np.isnan(fl):
        # Outside the convex hull — nearest-neighbour fallback
        fl = griddata(points, fl_vals, query, method='nearest')[0]
    fl = max(0.0, float(fl))

    return {
        'Fuel Lifetime': int(round(fl)),
        'Mass U235': max(0, int(round(mass_u235))),
        'Mass U238': max(0, int(round(mass_u238))),
    }


def _build_ltmr(params):
    """Populate params for LTMR (Liquid Metal Thermal Microreactor)."""

    # Sec 1: Materials
    # Note: 'Enrichment' and 'Power MWt' are pre-set by build_params from user input.
    params.update({
        'reactor type': 'LTMR',
        'TRISO Fueled': 'No',
        'Fuel': 'TRIGA_fuel',
        'H_Zr_ratio': 1.6,
        'U_met_wo': 0.3,
        'Coolant': 'NaK',
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Moderator': 'ZrH',
        'Control Drum Absorber': 'B4C_enriched',
        'Control Drum Reflector': 'Graphite',
        'Common Temperature': 600,
        'HX Material': 'SS316',
    })

    # Sec 2: Geometry
    params.update({
        'Fuel Pin Materials': ['Zr', None, params['Fuel'], None, 'SS304'],
        'Fuel Pin Radii': [0.28575, 0.3175, 1.5113, 1.5367, 1.5875],
        'Moderator Pin Materials': ['ZrH', 'SS304'],
        'Moderator Pin Inner Radius': 1.5367,
        'Moderator Pin Radii': [1.5367, 1.5875],
        'Pin Gap Distance': 0.1,
        'Pins Arrangement': LTMR_pins_arrangement,
        'Number of Rings per Assembly': 12,
        'Radial Reflector Thickness': 14,
    })
    params['Lattice Radius'] = calculate_lattice_radius(params)
    params['Active Height'] = 78.4
    params['Axial Reflector Thickness'] = params['Radial Reflector Thickness']
    params['Fuel Pin Count'] = calculate_pins_in_assembly(params, 'FUEL')
    params['Moderator Pin Count'] = calculate_pins_in_assembly(params, 'MODERATOR')
    params['Moderator Mass'] = calculate_moderator_mass(params)
    params['Core Radius'] = params['Lattice Radius'] + params['Radial Reflector Thickness']

    # Sec 3: Control drums
    params.update({
        'Drum Radius': 9.016,
        'Drum Absorber Thickness': 1,
        'Drum Height': params['Active Height'] + 2 * params['Axial Reflector Thickness'],
    })
    calculate_drums_volumes_and_masses(params)
    calculate_reflector_mass_LTMR(params)

    # Sec 4: Overall system
    # 'Power MWt' is pre-set by build_params; do not override it here.
    params.update({
        'Thermal Efficiency': 0.31,
        'Heat Flux Criteria': 0.9,
        'Burnup Steps': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
                         30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 140.0],
    })
    params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
    params['Heat Flux'] = calculate_heat_flux(params)

    # Sec 5: Interpolate OpenMC results from parametric study table
    _omc = interpolate_openmc_results('LTMR', params['Power MWt'], params['Enrichment'])
    if _omc['Fuel Lifetime'] == 0:
        raise SubcriticalError(
            f"Fuel lifetime is zero for LTMR at Power={params['Power MWt']} MWt, "
            f"Enrichment={params['Enrichment']:.4f}. The reactor is subcritical."
        )
    params.update(_omc)
    params['Uranium Mass'] = (params['Mass U235'] + params['Mass U238']) / 1000  # kg
    fuel_calculations(params)

    # Sec 6: Primary Loop + BoP
    params.update({
        'Secondary HX Mass': 0,
        'Primary Pump': 'Yes',
        'Secondary Pump': 'No',
        'Pump Isentropic Efficiency': 0.8,
        'Primary Loop Inlet Temperature': 430 + 273.15,
        'Primary Loop Outlet Temperature': 520 + 273.15,
        'Secondary Loop Inlet Temperature': 395 + 273.15,
        'Secondary Loop Outlet Temperature': 495 + 273.15,
    })
    params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)
    params.update({
        'BoP Count': 2,
        'BoP per loop load fraction': 0.5,
    })
    params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']
    mass_flow_rate(params)
    calculate_primary_pump_mechanical_power(params)

    # Sec 7: Shielding
    params.update({
        'In Vessel Shield Thickness': 10.16,
        'In Vessel Shield Inner Radius': params['Core Radius'],
        'In Vessel Shield Material': 'B4C_natural',
        'Out Of Vessel Shield Thickness': 39.37,
        'Out Of Vessel Shield Material': 'WEP',
        'Out Of Vessel Shield Effective Density Factor': 0.5,
    })
    params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

    # Sec 8: Vessels
    params.update({
        'Vessel Radius': params['Core Radius'] + params['In Vessel Shield Thickness'],
        'Vessel Thickness': 1,
        'Vessel Lower Plenum Height': 42.848 - 40,
        'Vessel Upper Plenum Height': 47.152,
        'Vessel Upper Gas Gap': 0,
        'Vessel Bottom Depth': 32.129,
        'Vessel Material': 'stainless_steel',
        'Gap Between Vessel And Guard Vessel': 2,
        'Guard Vessel Thickness': 0.5,
        'Guard Vessel Material': 'stainless_steel',
        'Gap Between Guard Vessel And Cooling Vessel': 5,
        'Cooling Vessel Thickness': 0.5,
        'Cooling Vessel Material': 'stainless_steel',
        'Gap Between Cooling Vessel And Intake Vessel': 3,
        'Intake Vessel Thickness': 0.5,
        'Intake Vessel Material': 'stainless_steel',
    })
    vessels_specs(params)
    calculate_shielding_masses(params)

    # Sec 9: Operation
    params.update({
        'Number of Operators': 2,
        'Levelization Period': 60,
        'Refueling Period': 7,
        'Startup Duration after Refueling': 2,
        'Reactors Monitored Per Operator': 10,
        'Security Staff Per Shift': 1,
    })
    params['Onsite Coolant Inventory'] = 1 * 855 * 8.2402
    params['Replacement Coolant Inventory'] = 0

    total_refueling_period = (params['Fuel Lifetime'] + params['Refueling Period']
                              + params['Startup Duration after Refueling'])
    total_refueling_period_yr = total_refueling_period / 365
    params['A75: Vessel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Core Barrel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Reflector Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Drum Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['Mainenance to Direct Cost Ratio'] = 0.015
    params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

    # Sec 10: Buildings & Economic params
    params.update({
        'Land Area': 18,
        'Excavation Volume': 412.605,
        'Reactor Building Slab Roof Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Basement Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Exterior Walls Volume': ((2 * 9750 * 3500 * 1500) + (3502.4 * 3500 * (1500 + 750))) / 1e9,
        'Reactor Building Superstructure Area': ((2 * 3500 * 3500) + (2 * 7500 * 3500)) / 1e6,
        'Integrated Heat Exchanger Building Slab Roof Volume': 0,
        'Integrated Heat Exchanger Building Basement Volume': 0,
        'Integrated Heat Exchanger Building Exterior Walls Volume': 0,
        'Integrated Heat Exchanger Building Superstructure Area': 0,
        'Turbine Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Control Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Manipulator Building Slab Roof Volume': (4876.8 * 2438.4 * 400) / 1e9,
        'Manipulator Building Basement Volume': (4876.8 * 2438.4 * 1500) / 1e9,
        'Manipulator Building Exterior Walls Volume': ((4876.8 * 4445 * 400) + (2038.4 * 4445 * 400 * 2)) / 1e9,
        'Refueling Building Slab Roof Volume': 0,
        'Refueling Building Basement Volume': 0,
        'Refueling Building Exterior Walls Volume': 0,
        'Spent Fuel Building Slab Roof Volume': 0,
        'Spent Fuel Building Basement Volume': 0,
        'Spent Fuel Building Exterior Walls Volume': 0,
        'Emergency Building Slab Roof Volume': 0,
        'Emergency Building Basement Volume': 0,
        'Emergency Building Exterior Walls Volume': 0,
        'Storage Building Slab Roof Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Basement Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Exterior Walls Volume': ((8400 * 2700 * 400) + (3100 * 2700 * 400 * 2)) / 1e9,
        'Radwaste Building Slab Roof Volume': 0,
        'Radwaste Building Basement Volume': 0,
        'Radwaste Building Exterior Walls Volume': 0,
        'Annual Return': 0.0475,
        'NOAK Unit Number': 100,
        'Escalation Year': 2024,
        'Interest Rate': 0.07,
        'Construction Duration': 12,
        'Debt To Equity Ratio': 0.5,
    })
    params['ITC credit level'] = 0.30


def _build_gcmr(params):
    """Populate params for GCMR Design A (Gas Cooled Microreactor)."""

    # Sec 1: Materials
    # Note: 'Enrichment' and 'Power MWt' are pre-set by build_params from user input.
    params.update({
        'reactor type': 'GCMR',
        'TRISO Fueled': 'Yes',
        'Fuel': 'UN',
        'UO2 atom fraction': 0.7,
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Matrix Material': 'Graphite',
        'Moderator': 'Graphite',
        'Moderator Booster': 'ZrH',
        'Coolant': 'Helium',
        'Common Temperature': 850,
        'Control Drum Absorber': 'B4C_enriched',
        'Control Drum Reflector': 'Graphite',
        'HX Material': 'SS316',
    })

    # Sec 2: Geometry
    params.update({
        'Fuel Pin Materials': ['UN', 'buffer_graphite', 'PyC', 'SiC', 'PyC'],
        'Fuel Pin Radii': [0.025, 0.035, 0.039, 0.0425, 0.047],
        'Compact Fuel Radius': 0.6225,
        'Packing Fraction': 0.3,
        'Coolant Channel Radius': 0.35,
        'Moderator Booster Radius': 0.55,
        'Lattice Pitch': 2.25,
        'Assembly Rings': 6,
        'Core Rings': 5,
    })
    params['Assembly FTF'] = params['Lattice Pitch'] * (params['Assembly Rings'] - 1) * np.sqrt(3)
    params['Radial Reflector Thickness'] = 27.393
    params['Axial Reflector Thickness'] = params['Radial Reflector Thickness']
    params['Core Radius'] = (params['Assembly FTF'] * params['Core Rings']
                             + params['Radial Reflector Thickness'])
    params['Active Height'] = 250

    # Sec 3: Control drums
    params.update({
        'Drum Radius': 9,
        'Drum Absorber Thickness': 1,
        'Drum Height': params['Active Height'] + 2 * params['Axial Reflector Thickness'],
    })
    calculate_drums_volumes_and_masses(params)
    calculate_reflector_mass_GCMR(params)
    calculate_moderator_mass_GCMR(params)

    # Sec 4: Overall system
    # 'Power MWt' is pre-set by build_params; do not override it here.
    params.update({
        'Thermal Efficiency': 0.4,
        'Heat Flux Criteria': 0.9,
        'Burnup Steps': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
                         30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0],
    })
    params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
    params['Heat Flux'] = calculate_heat_flux_TRISO(params)

    # Sec 5: Interpolate OpenMC results from parametric study table
    _omc = interpolate_openmc_results('GCMR', params['Power MWt'], params['Enrichment'])
    if _omc['Fuel Lifetime'] == 0:
        raise SubcriticalError(
            f"Fuel lifetime is zero for GCMR at Power={params['Power MWt']} MWt, "
            f"Enrichment={params['Enrichment']:.4f}. The reactor is subcritical."
        )
    params.update(_omc)
    params['Uranium Mass'] = (params['Mass U235'] + params['Mass U238']) / 1000  # kg
    fuel_calculations(params)

    # Sec 6: Primary Loop + BoP
    params.update({
        'Primary Loop Purification': True,
        'Secondary HX Mass': 0,
        'Compressor Pressure Ratio': 4,
        'Compressor Isentropic Efficiency': 0.8,
        'Primary Loop Count': 2,
        'Primary Loop per loop load fraction': 0.5,
        'Primary Loop Inlet Temperature': 300 + 273.15,
        'Primary Loop Outlet Temperature': 550 + 273.15,
        'Secondary Loop Inlet Temperature': 290 + 273.15,
        'Secondary Loop Outlet Temperature': 500 + 273.15,
        'Primary Loop Pressure Drop': 50e3,
    })
    params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)
    mass_flow_rate(params)
    compressor_power(params)
    params.update({
        'BoP Count': 2,
        'BoP per loop load fraction': 0.5,
    })
    params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']
    params.update({
        'Integrated Heat Transfer Vessel Thickness': 0,
        'Integrated Heat Transfer Vessel Material': 'SA508',
    })
    GCMR_integrated_heat_transfer_vessel(params)

    # Sec 7: Shielding
    params.update({
        'In Vessel Shield Thickness': 0,
        'In Vessel Shield Inner Radius': params['Core Radius'],
        'In Vessel Shield Material': 'B4C_natural',
        'Out Of Vessel Shield Thickness': 39.37,
        'Out Of Vessel Shield Material': 'WEP',
        'Out Of Vessel Shield Effective Density Factor': 0.5,
    })
    params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

    # Sec 8: Vessels
    params.update({
        'Vessel Radius': params['Core Radius'] + params['In Vessel Shield Thickness'],
        'Vessel Thickness': 1,
        'Vessel Lower Plenum Height': 42.848 - 40,
        'Vessel Upper Plenum Height': 47.152,
        'Vessel Upper Gas Gap': 0,
        'Vessel Bottom Depth': 32.129,
        'Vessel Material': 'stainless_steel',
        'Gap Between Vessel And Guard Vessel': 0,
        'Guard Vessel Thickness': 0,
        'Guard Vessel Material': 'low_alloy_steel',
        'Gap Between Guard Vessel And Cooling Vessel': 5,
        'Cooling Vessel Thickness': 0.5,
        'Cooling Vessel Material': 'stainless_steel',
        'Gap Between Cooling Vessel And Intake Vessel': 4,
        'Intake Vessel Thickness': 0.5,
        'Intake Vessel Material': 'stainless_steel',
    })
    vessels_specs(params)
    calculate_shielding_masses(params)

    # Sec 9: Operation
    params.update({
        'Number of Operators': 2,
        'Levelization Period': 60,
        'Refueling Period': 7,
        'Startup Duration after Refueling': 2,
        'Reactors Monitored Per Operator': 10,
        'Security Staff Per Shift': 1,
    })
    params['Onsite Coolant Inventory'] = 10 * 24.417 * 8.2402
    params['Replacement Coolant Inventory'] = params['Onsite Coolant Inventory'] / 4
    params['Annual Coolant Supply Frequency'] = 1 if params['Primary Loop Purification'] else 6

    total_refueling_period = (params['Fuel Lifetime'] + params['Refueling Period']
                              + params['Startup Duration after Refueling'])
    total_refueling_period_yr = total_refueling_period / 365
    params['A75: Vessel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Core Barrel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Reflector Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Drum Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['Mainenance to Direct Cost Ratio'] = 0.015
    params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

    # Sec 10: Buildings & Economic params
    params.update({
        'Land Area': 18,
        'Excavation Volume': 412.605,
        'Reactor Building Slab Roof Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Basement Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Exterior Walls Volume': ((2 * 9750 * 3500 * 1500) + (3502.4 * 3500 * (1500 + 750))) / 1e9,
        'Reactor Building Superstructure Area': ((2 * 3500 * 3500) + (2 * 7500 * 3500)) / 1e6,
        'Integrated Heat Exchanger Building Slab Roof Volume': 0,
        'Integrated Heat Exchanger Building Basement Volume': 0,
        'Integrated Heat Exchanger Building Exterior Walls Volume': 0,
        'Integrated Heat Exchanger Building Superstructure Area': 0,
        'Turbine Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Control Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Manipulator Building Slab Roof Volume': (4876.8 * 2438.4 * 400) / 1e9,
        'Manipulator Building Basement Volume': (4876.8 * 2438.4 * 1500) / 1e9,
        'Manipulator Building Exterior Walls Volume': ((4876.8 * 4445 * 400) + (2038.4 * 4445 * 400 * 2)) / 1e9,
        'Refueling Building Slab Roof Volume': 0,
        'Refueling Building Basement Volume': 0,
        'Refueling Building Exterior Walls Volume': 0,
        'Spent Fuel Building Slab Roof Volume': 0,
        'Spent Fuel Building Basement Volume': 0,
        'Spent Fuel Building Exterior Walls Volume': 0,
        'Emergency Building Slab Roof Volume': 0,
        'Emergency Building Basement Volume': 0,
        'Emergency Building Exterior Walls Volume': 0,
        'Storage Building Slab Roof Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Basement Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Exterior Walls Volume': ((8400 * 2700 * 400) + (3100 * 2700 * 400 * 2)) / 1e9,
        'Radwaste Building Slab Roof Volume': 0,
        'Radwaste Building Basement Volume': 0,
        'Radwaste Building Exterior Walls Volume': 0,
        'Annual Return': 0.0475,
        'NOAK Unit Number': 100,
        'Escalation Year': 2024,
        'Interest Rate': 0.07,
        'Construction Duration': 12,
        'Debt To Equity Ratio': 0.5,
    })
    params['PTC credit value'] = 15.0
    params['PTC credit period'] = 10
    params['domestic_content_bonus'] = 0.10
    params['energy_community_bonus'] = 0.10
    params['Tax Rate'] = 0.21


def _build_hpmr(params):
    """Populate params for HPMR (Heat Pipe Microreactor)."""

    # Sec 1: Materials
    # Note: 'Enrichment' and 'Power MWt' are pre-set by build_params from user input.
    params.update({
        'reactor type': 'HPMR',
        'TRISO Fueled': 'Yes',
        'Fuel': 'homog_TRISO',
        'Radial Reflector': 'Graphite',
        'Axial Reflector': 'Graphite',
        'Moderator': 'monolith_graphite',
        'Secondary Coolant': 'Helium',
        'Control Drum Absorber': 'B4C_natural',
        'Control Drum Reflector': 'Graphite',
        'Cooling Device': 'heatpipe',
        'Common Temperature': 1000,
        'HX Material': 'SS316',
    })

    # Sec 2: Geometry
    params.update({
        'Fuel Pin Materials': ['homog_TRISO', 'Helium'],
        'Fuel Pin Radii': [1.00, 1.05],
        'Heat Pipe Materials': ['heatpipe', 'Helium'],
        'Heat Pipe Radii': [1.10, 1.15],
        'Number of Rings per Assembly': 6,
        'Number of Rings per Core': 3,
        'Lattice Pitch': 3.4,
    })
    params['Assembly FTF'] = ((params['Lattice Pitch'] * (params['Number of Rings per Assembly'] - 1)
                               + 1.4 * params['Fuel Pin Radii'][-1]) * np.sqrt(3))
    params['hexagonal Core Edge Length'] = ((params['Assembly FTF'] * (params['Number of Rings per Core'] - 1))
                                            + (params['Assembly FTF'] / 2) + 6.6)
    params['Radial Reflector Thickness'] = 50
    params['Core Radius'] = (0.5 * np.sqrt(3) * params['hexagonal Core Edge Length']
                             + params['Radial Reflector Thickness'])
    params['Active Height'] = 2 * params['Core Radius']
    params['Axial Reflector Thickness'] = params['Radial Reflector Thickness']
    params['Fuel Pin Count per Assembly'] = calculate_number_fuel_elements_hpmr(params['Number of Rings per Assembly'])
    params['Fuel Assemblies Count'] = ((3 * params['Number of Rings per Core'] ** 2)
                                       - (3 * params['Number of Rings per Core']))
    params['Fuel Pin Count'] = params['Fuel Assemblies Count'] * params['Fuel Pin Count per Assembly']
    number_of_heatpipes_hmpr(params)

    # Sec 3: Control drums
    params.update({
        'Drum Radius': 0.4 * params['Radial Reflector Thickness'],
        'Drum Absorber Thickness': 1,
        'Drum Height': params['Active Height'],
    })
    calculate_drums_volumes_and_masses(params)
    calculate_reflector_and_moderator_mass_HPMR(params)

    # Sec 4: Overall system
    # 'Power MWt' is pre-set by build_params; do not override it here.
    params.update({
        'Thermal Efficiency': 0.36,
        'Heat Flux Criteria': 0.9,
        'Time Steps': [t * 86400 for t in [0.01, 0.99, 3, 6, 20, 70, 100, 165,
                                            365, 365, 365, 365, 365, 365, 365.00]],
    })
    params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
    params['Heat Flux'] = calculate_heat_flux(params)

    # Sec 5: Interpolate OpenMC results from parametric study table
    _omc = interpolate_openmc_results('HPMR', params['Power MWt'], params['Enrichment'])
    if _omc['Fuel Lifetime'] == 0:
        raise SubcriticalError(
            f"Fuel lifetime is zero for HPMR at Power={params['Power MWt']} MWt, "
            f"Enrichment={params['Enrichment']:.4f}. The reactor is subcritical."
        )
    params.update(_omc)
    params['Uranium Mass'] = (params['Mass U235'] + params['Mass U238']) / 1000  # kg
    fuel_calculations(params)

    # Sec 6: Primary Loop + BoP
    params.update({
        'Primary Loop Purification': True,
        'Secondary HX Mass': 0,
        'Primary Loop Count': 2,
        'Primary Loop Inlet Temperature': 900 + 273.15,
        'Primary Loop Outlet Temperature': 650 + 273.15,
        'Secondary Loop Inlet Temperature': 300 + 273.15,
        'Secondary Loop Outlet Temperature': 630 + 273.15,
    })
    params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)
    params.update({
        'BoP Count': 2,
        'BoP per loop load fraction': 0.5,
    })
    params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']
    # Note: HPMR uses heat pipes — no primary pump or mass flow rate call

    # Sec 7: Shielding
    params.update({
        'In Vessel Shield Thickness': 0,
        'In Vessel Shield Inner Radius': params['Core Radius'],
        'In Vessel Shield Material': 'B4C_natural',
        'Out Of Vessel Shield Thickness': 39.37,
        'Out Of Vessel Shield Material': 'WEP',
        'Out Of Vessel Shield Effective Density Factor': 0.5,
    })
    params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

    # Sec 8: Vessels
    params.update({
        'Vessel Radius': params['Core Radius'] + params['In Vessel Shield Thickness'],
        'Vessel Thickness': 1,
        'Vessel Lower Plenum Height': 42.848 - 40,
        'Vessel Upper Plenum Height': 47.152,
        'Vessel Upper Gas Gap': 0,
        'Vessel Bottom Depth': 32.129,
        'Vessel Material': 'stainless_steel',
        'Gap Between Vessel And Guard Vessel': 0,
        'Guard Vessel Thickness': 0,
        'Guard Vessel Material': 'low_alloy_steel',
        'Gap Between Guard Vessel And Cooling Vessel': 5,
        'Cooling Vessel Thickness': 0.5,
        'Cooling Vessel Material': 'stainless_steel',
        'Gap Between Cooling Vessel And Intake Vessel': 4,
        'Intake Vessel Thickness': 0.5,
        'Intake Vessel Material': 'stainless_steel',
    })
    vessels_specs(params)
    calculate_shielding_masses(params)

    # Sec 9: Operation
    params.update({
        'Number of Operators': 2,
        'Levelization Period': 60,
        'Refueling Period': 7,
        'Startup Duration after Refueling': 2,
        'Reactors Monitored Per Operator': 10,
        'Security Staff Per Shift': 1,
    })
    params['Onsite Coolant Inventory'] = 1 * 24.417 * 8.2402
    params['Replacement Coolant Inventory'] = params['Onsite Coolant Inventory'] / 4
    params['Annual Coolant Supply Frequency'] = 1 if params['Primary Loop Purification'] else 6

    total_refueling_period = (params['Fuel Lifetime'] + params['Refueling Period']
                              + params['Startup Duration after Refueling'])
    total_refueling_period_yr = total_refueling_period / 365
    params['A75: Vessel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Core Barrel Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Reflector Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['A75: Drum Replacement Period (cycles)'] = np.floor(10 / total_refueling_period_yr)
    params['Mainenance to Direct Cost Ratio'] = 0.015
    params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

    # Sec 10: Buildings & Economic params
    params.update({
        'Land Area': 18,
        'Excavation Volume': 412.605,
        'Reactor Building Slab Roof Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Basement Volume': (9750 * 6502.4 * 1500) / 1e9,
        'Reactor Building Exterior Walls Volume': ((2 * 9750 * 3500 * 1500) + (3502.4 * 3500 * (1500 + 750))) / 1e9,
        'Reactor Building Superstructure Area': ((2 * 3500 * 3500) + (2 * 7500 * 3500)) / 1e6,
        'Integrated Heat Exchanger Building Slab Roof Volume': 0,
        'Integrated Heat Exchanger Building Basement Volume': 0,
        'Integrated Heat Exchanger Building Exterior Walls Volume': 0,
        'Integrated Heat Exchanger Building Superstructure Area': 0,
        'Turbine Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Turbine Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Control Building Slab Roof Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Basement Volume': (12192 * 2438 * 200) / 1e9,
        'Control Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,
        'Manipulator Building Slab Roof Volume': (4876.8 * 2438.4 * 400) / 1e9,
        'Manipulator Building Basement Volume': (4876.8 * 2438.4 * 1500) / 1e9,
        'Manipulator Building Exterior Walls Volume': ((4876.8 * 4445 * 400) + (2038.4 * 4445 * 400 * 2)) / 1e9,
        'Refueling Building Slab Roof Volume': 0,
        'Refueling Building Basement Volume': 0,
        'Refueling Building Exterior Walls Volume': 0,
        'Spent Fuel Building Slab Roof Volume': 0,
        'Spent Fuel Building Basement Volume': 0,
        'Spent Fuel Building Exterior Walls Volume': 0,
        'Emergency Building Slab Roof Volume': 0,
        'Emergency Building Basement Volume': 0,
        'Emergency Building Exterior Walls Volume': 0,
        'Storage Building Slab Roof Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Basement Volume': (8400 * 3500 * 400) / 1e9,
        'Storage Building Exterior Walls Volume': ((8400 * 2700 * 400) + (3100 * 2700 * 400 * 2)) / 1e9,
        'Radwaste Building Slab Roof Volume': 0,
        'Radwaste Building Basement Volume': 0,
        'Radwaste Building Exterior Walls Volume': 0,
        'Annual Return': 0.0475,
        'NOAK Unit Number': 100,
        'Escalation Year': 2024,
        'Interest Rate': 0.07,
        'Construction Duration': 12,
        'Debt To Equity Ratio': 0.5,
    })
    # No ITC/PTC credits for HPMR by default


def build_params(reactor_type, power_mwt, enrichment, user_overrides):
    """
    Build a fully-populated params dict for the given reactor type,
    then apply user_overrides on top.

    Parameters
    ----------
    reactor_type : str
        One of 'LTMR', 'GCMR', 'HPMR'.
    power_mwt : float
        Thermal power in MWt (1–20). Controls all power-dependent params and
        is used to interpolate Fuel Lifetime from the parametric study table.
    enrichment : float
        U-235 enrichment fraction (0.05–0.1975). Controls all enrichment-dependent
        params (Uranium masses, Fuel Lifetime) via the parametric study table.
    user_overrides : dict
        User-supplied values that override defaults (e.g. Interest Rate).

    Returns
    -------
    dict
        Fully-populated params dict ready for bottom_up_cost_estimate().

    Raises
    ------
    SubcriticalError
        If the interpolated Fuel Lifetime is zero at the requested operating point.
    """
    params = {}
    # Set power and enrichment FIRST so all builder sections can use them.
    params['Power MWt'] = float(power_mwt)
    params['Enrichment'] = float(enrichment)

    builders = {'LTMR': _build_ltmr, 'GCMR': _build_gcmr, 'HPMR': _build_hpmr}
    if reactor_type not in builders:
        raise ValueError(f"Unknown reactor type: {reactor_type!r}. Choose LTMR, GCMR, or HPMR.")

    builders[reactor_type](params)

    # Apply user overrides (these come last so they win)
    params.update(user_overrides)

    # Number of Samples = 1: use deterministic class-3 estimates (no Monte Carlo sampling).
    # With Number of Samples > 1, the lognormal sampler is called with NaN bounds for
    # many database rows (those lacking low/high-end estimates), which propagates NaN
    # through the entire cost calculation. Number of Samples = 1 uses unit_cost_0 /
    # fixed_cost_0 directly and avoids this issue entirely.
    params['Number of Samples'] = 1

    return params
