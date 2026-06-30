# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
watts_exec_LTMR_3D.py

Genuine-3D LTMR core run that evaluates the temperature / reflector / coolant
reactivity coefficients with automatic thermal geometric expansion, then applies
the quasi-static A/B/C inherent-safety screen (ABC Analysis).

Expansion rules (core_design_3D/core_thermal_geometry.py):
  1. Fuel expands axially only.
  2. All other components expand axially at the fuel's rate.
  3. Only reflecting components expand radially.
"""
import os
import sys
sys.path.insert(0, '/home/garcsamu/OpenMC/TEMA')   # repo root on the INL HPC
import numpy as np
import watts

from core_design_3D.openmc_template_LTMR_3D import build_openmc_model_LTMR_3D
from core_design_3D.pins_arrangement_3D import LTMR_pins_arrangement
from core_design_3D.utils_3D import *          # run_openmc, monitor_heat_flux,
                                               # calculate_hex_apothem,
                                               # calculate_core_radius_from_hex,
                                               # calculate_pins_in_assembly,
                                               # calculate_heat_flux, ...
from core_design_3D.drums_3D import *          # calculate_drums_volumes_and_masses,
                                               # calculate_moderator_mass,
                                               # calculate_reflector_mass_LTMR, ...
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
from reactor_engineering_evaluation.tools import *

import warnings
warnings.filterwarnings("ignore")

import time
time_start = time.time()

params = watts.Parameters()

def update_params(updates):
    params.update(updates)

# **************************************************************************************
#                                   Sec. 0: Settings
# **************************************************************************************
update_params({
    'plotting': "N",
    'cross_sections_xml_location': '/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml',
    'simplified_chain_thermal_xml': '/home/garcsamu/OpenMC/TEMA/data/chain_casl_pwr.xml',
    'XS_type': 'endf8.0',          # materials dispatcher in openmc_materials_database_3D
})

# **************************************************************************************
#                                   Sec. 1: Materials
# **************************************************************************************
update_params({
    'reactor type': "LTMR",
    'Fuel': 'UO2',
    'Enrichment': 0.1975,
    'TRISO Fueled': "No",
    'H_Zr_ratio': 1.6,
    'U_met_wo': 0.3,
    'er_wo': 0.0,                  # required by the UZrH builder; 0 = no erbium
    'Coolant': 'NaK',
    'Radial Reflector': 'BeO',
    'Axial Reflector': 'BeO',
    'Moderator': 'ZrH',
    'Control Drum Absorber': 'B4C_enriched',
    'Control Drum Reflector': 'BeO',
    'Common Temperature': 600,     # operating temperature (K)
    'UO2 atom fraction': 0.75,
    'HX Material': 'SS316',

    # --- thermal-expansion controls (ABC mode) ---
    # Build materials cold and let the geometry builder own the (mass-conserving)
    # solid-density scaling; temperatures and coolant EOS density are still applied
    # by the materials module (after the materials-module edit).
    'Thermal Expansion': True,
    'Reference Temperature': 600,     # or set = Common Temperature to keep base
                                         # geometry unperturbed
    'Per-Region Temperatures': True,     # isolate fuel/reflector/coolant Doppler
                                         # (requires the materials-module edit)
    'Geometric Expansion': True
})

# **************************************************************************************
#                       Sec. 2: Geometry (Pins, Lattice, Reflector)
# **************************************************************************************
update_params({
    'Fuel Pin Materials': ['Zr', None, params['Fuel'], None, 'SS304'],
    'Fuel Pin Radii': [0.28575, 0.3175, 1.5113, 1.5367, 1.5875],   # cm
    'Moderator Pin Materials': ['ZrH', 'SS304'],
    'Moderator Pin Inner Radius': 1.5367,
    'Moderator Pin Radii': [1.5367, 1.5875],
    "Pin Gap Distance": 0.1,
    'Pins Arrangement': LTMR_pins_arrangement,
    'Number of Rings per Assembly': 12,
    'Radial Reflector Thickness': 14,     # cm
})

params['Lattice Apothem'] = calculate_hex_apothem(params)
params['Lattice Radius'] = params['Lattice Apothem']
params['Assembly FTF'] = 2 * params['Lattice Apothem']
params['Active Height'] = 78.4
params['Axial Reflector Thickness'] = params['Radial Reflector Thickness']
params['Fuel Pin Count'] = calculate_pins_in_assembly(params, "FUEL")
params['Moderator Pin Count'] = calculate_pins_in_assembly(params, "MODERATOR")
params['Moderator Mass'] = calculate_moderator_mass(params)
params['Core Radius'] = calculate_core_radius_from_hex(params)

# **************************************************************************************
#                                   Sec. 3: Control Drums
# **************************************************************************************
update_params({
    'Number of Drums': 12,
    # 'Drum Radius': 9.016,        # optional; auto-sized if omitted
    'Drum Absorber Thickness': 1,
    'Drum Absorber Arc Degrees': 120,
    'Drum Height': params['Active Height'] + 2 * params['Axial Reflector Thickness'],
})
calculate_drums_volumes_and_masses(params)
calculate_reflector_mass_LTMR(params)

# **************************************************************************************
#                                   Sec. 4: Overall System
# **************************************************************************************
update_params({
    'Power MWt': 20,
    'Thermal Efficiency': 0.31,
    'Heat Flux Criteria': 0.9,
    'Burnup Steps': [0.1, 1.0, 140],   # MWd/kg (trim/extend as needed)
})
params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
params['Heat Flux'] = calculate_heat_flux(params)

# **************************************************************************************
#             Sec. 4b: Operating conditions used by the A/B/C safety screen
# **************************************************************************************
update_params({
    'Primary Loop Inlet Temperature': 430 + 273.15,   # K
    'Primary Loop Outlet Temperature': 520 + 273.15,  # K
    'Coolant Boiling Temperature': 1058.15,           # K (NaK ~785 C)
    'Particles' : 2000000
})

# **************************************************************************************
#                           Sec. 5: Running OpenMC (ABC Analysis)
# **************************************************************************************
params['Shutdown Margin Calc'] = False
params['Isothermal Temperature Coefficients'] = False

params['ABC Analysis'] = True
params['Temperature Perturbation'] = 100        # K
# Optional per-region base temperatures (default to Common Temperature):
# params['Fuel Temperature'] = 900
# params['Reflector Temperature'] = 750
# params['Coolant Temperature'] = 800

heat_flux_monitor = monitor_heat_flux(params)
run_openmc(build_openmc_model_LTMR_3D, heat_flux_monitor, params)   # 3-arg signature

print("\n================ ABC reactivity coefficients (pcm/K) ================")
print(f"  Temperature (fuel)  2D: {params.get('Temp Coeff 2D'):>10.3f} ± {params.get('Temp Coeff std'):.3f}   "
      f"3D(2D-corr): {params.get('Temp Coeff 3D (2D corrected)'):>10.3f}")
print(f"  Reflector           2D: {params.get('Reflector Coeff 2D'):>10.3f} ± {params.get('Reflector Coeff std'):.3f}   "
      f"3D(2D-corr): {params.get('Reflector Coeff 3D (2D corrected)'):>10.3f}")
print(f"  Coolant             2D: {params.get('Coolant Coeff 2D'):>10.3f} ± {params.get('Coolant Coeff std'):.3f}   "
      f"3D(2D-corr): {params.get('Coolant Coeff 3D (2D corrected)'):>10.3f}")
print("---------------- A/B/C quasi-static safety screen -------------------")
print(f"  A = {params.get('ABC A (pcm)'):>10.3f} pcm")
print(f"  B = {params.get('ABC B (pcm)'):>10.3f} pcm")
print(f"  C = {params.get('ABC C (pcm/K)'):>10.3f} pcm/K")
print(f"  Criteria (1,2,3): {params.get('ABC Criterion 1')}, "
      f"{params.get('ABC Criterion 2')}, {params.get('ABC Criterion 3')}   "
      f"-> Safe: {params.get('ABC Safe')}")
print("=====================================================================\n")

elapsed_time = (time.time() - time_start) / 60
print('Execution time:', np.round(elapsed_time, 2), 'minutes')