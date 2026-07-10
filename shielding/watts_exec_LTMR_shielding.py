# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
This script performs a shielding parametric study for the MOUSE (Mobile Operated
Utility System for Energy) Liquid Metal Thermal Microreactor (LTMR).

Two deployment scenarios are evaluated:
  - Mobile:     Core → In-vessel shield → Vessels → Out-of-vessel shield → ISO container steel
  - Non-Mobile: Core → In-vessel shield → Vessels → Out-of-vessel shield

The workflow is two-step:
  Step 1 (Criticality): Run k-eigenvalue calculation using the standard LTMR model
                         to generate a converged fission source file.
  Step 2 (Shielding):   Run fixed-source transport (neutron + photon) using the
                         shielding geometry and the fission source from Step 1.
                         Cylindrical mesh tallies with ICRP-116 H*(10) dose response
                         functions are used to extract dose rates at key locations.

Users can modify parameters in the "params" dictionary below. The parametric sweep
loops over:
  - Mobile flag (True / False)
  - Out-of-vessel shield materials
  - Out-of-vessel shield thicknesses
"""

import os
import sys
import numpy as np
import watts
from core_design.openmc_template_LTMR import *
from core_design.pins_arrangement import LTMR_pins_arrangement
from core_design.utils import *
from core_design.drums import *
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
from reactor_engineering_evaluation.BOP import *
from reactor_engineering_evaluation.vessels_calcs import *
from reactor_engineering_evaluation.tools import *
from cost.cost_estimation import parametric_studies

# New shielding-specific imports
from shielding.openmc_shielding_template_LTMR import build_openmc_shielding_model_LTMR
from core_design.utils import run_openmc_shielding
from shielding.shielding_calcs import summarize_shielding_results

try:
    number_processes = sys.argv[1]
    mpi_args = ['mpirun', '-np', f'{number_processes}']
    print(f"\n\nMPI enabled with {number_processes} processes")
except IndexError:
    mpi_args = None
    print("\n\nMPI not used (no process count provided, running in serial)\n\n")

import warnings
warnings.filterwarnings("ignore")

import time
time_start = time.time()

params = watts.Parameters()

def update_params(updates):
    params.update(updates)

# **************************************************************************************************************************
#                                                Sec. 0: Settings
# **************************************************************************************************************************

update_params({
    'plotting': "N",  # Shielding runs are expensive; disable geometry plots by default
    'cross_sections_xml_location': '/hpc-common/data/openmc/endfb-viii.0-hdf5/cross_sections.xml',
    'simplified_chain_thermal_xml': '/home/garcsamu/OpenMC/data/chain_casl_pwr.xml',
    'shielding_output_dir': '/home/garcsamu/OpenMC/TEMA/results',
})

os.makedirs(params['shielding_output_dir'], exist_ok=True)

# **************************************************************************************************************************
#                                                Sec. 1: Materials (fixed for shielding study)
# **************************************************************************************************************************

update_params({
    'reactor type': "LTMR",
    'Fuel': 'UN',               # Fix fuel type for shielding study; change as needed
    'Enrichment': 0.1975,
    'TRISO Fueled': "No",
    "H_Zr_ratio": 1.6,
    'U_met_wo': 0.3,
    'Coolant': 'NaK',
    'Radial Reflector': 'Graphite',
    'Axial Reflector': 'Graphite',
    'Moderator': 'ZrH',
    'Control Drum Absorber': 'B4C_enriched',
    'Control Drum Reflector': 'Graphite',
    'Common Temperature': 600,  # K
    'UO2 atom fraction': 0.75,
    'HX Material': 'SS316',
})

# **************************************************************************************************************************
#                                                Sec. 2: Core Geometry (fixed)
# **************************************************************************************************************************

update_params({
    'Fuel Pin Materials': ['Zr', None, params['Fuel'], None, 'SS304'],
    'Fuel Pin Radii': [0.28575, 0.3175, 1.5113, 1.5367, 1.5875],  # cm
    'Moderator Pin Materials': ['ZrH', 'SS304'],
    'Moderator Pin Inner Radius': 1.5367,  # cm
    'Moderator Pin Radii': [1.5367, 1.5875],
    "Pin Gap Distance": 0.1,  # cm
    'Pins Arrangement': LTMR_pins_arrangement,
    'Number of Rings per Assembly': 12,
    'Reflector Thickness': 14,  # cm
})

params['Lattice Radius']            = calculate_lattice_radius(params)
params['Active Height']             = 78.4   # cm
params['Axial Reflector Thickness'] = params['Reflector Thickness']
params['Fuel Pin Count']            = calculate_pins_in_assembly(params, "FUEL")
params['Moderator Pin Count']       = calculate_pins_in_assembly(params, "MODERATOR")
params['Moderator Mass']            = calculate_moderator_mass(params)
params['Core Radius']               = params['Lattice Radius'] + params['Reflector Thickness']

# **************************************************************************************************************************
#                                                Sec. 3: Control Drums (fixed)
# **************************************************************************************************************************

update_params({
    'Drum Radius': 9.016,  # cm
    'Drum Absorber Thickness': 1,  # cm
    'Drum Height': params['Active Height'] + 2 * params['Axial Reflector Thickness'],
})

drum_tube_radius = params['Drum Radius'] + params['Drum Radius'] / 90
cd_distance      = 0.86602540378 * params['Lattice Radius'] + drum_tube_radius
core_radius      = params['Lattice Radius'] + params['Reflector Thickness']

if cd_distance + drum_tube_radius >= core_radius:
    max_drum_radius     = (core_radius - 0.86602540378 * params['Lattice Radius']) / (2 * (1 + 1/90))
    adjusted_drum_radius = max_drum_radius * 0.95
    print(f"WARNING: Drum radius auto-adjusted: {params['Drum Radius']:.3f} -> {adjusted_drum_radius:.3f} cm")
    update_params({'Drum Radius': adjusted_drum_radius})

calculate_drums_volumes_and_masses(params)
calculate_reflector_mass_LTMR(params)

# **************************************************************************************************************************
#                                                Sec. 4: Overall System (fixed)
# **************************************************************************************************************************

update_params({
    'Power MWt': 20,
    'Thermal Efficiency': 0.31,
    'Heat Flux Criteria': 0.9,  # MW/m^2
    'Burnup Steps': [0.1, 0.5, 160],  # MWd/kg (trimmed for speed)
})
params['Power MWe']   = params['Power MWt'] * params['Thermal Efficiency']
params['Heat Flux']   = calculate_heat_flux(params)

# **************************************************************************************************************************
#        Sec. 5: STEP 1 — Criticality Run (generate fission source for shielding transport)
# **************************************************************************************************************************

print("\n\n" + "="*70)
print("STEP 1: Criticality run to generate converged fission source")
print("="*70)

params['SD Margin Calc']                  = False
params['Isothermal Temperature Coefficients'] = False
params['Shielding Run']                   = False  # tells template: normal k-eigen mode

heat_flux_monitor = monitor_heat_flux(params)
run_openmc(build_openmc_model_LTMR, heat_flux_monitor, params, mpi_args)
fuel_calculations(params)

# The k-eigenvalue run writes a source file; record its path for Step 2.
# OpenMC writes the last-batch source to 'source.{batch}.h5' in the run dir.
# watts_exec typically lands run artifacts in a dated subfolder; adjust path if needed.
params['Fission Source File'] = os.path.join(
    params.get('openmc_run_dir', '.'),
    'source.h5'
)

# **************************************************************************************************************************
#                                                Sec. 6: Primary Loop + BoP (for mass/cost context)
# **************************************************************************************************************************

update_params({
    'Secondary HX Mass': 0,
    'Primary Pump': 'Yes',
    'Secondary Pump': 'No',
    'Pump Isentropic Efficiency': 0.8,
    'Primary Loop Inlet Temperature':  430 + 273.15,  # K
    'Primary Loop Outlet Temperature': 520 + 273.15,  # K
    'Secondary Loop Inlet Temperature':  395 + 273.15,  # K
    'Secondary Loop Outlet Temperature': 495 + 273.15,  # K
})

params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)
params.update({
    'BoP Count': 2,
    'BoP per loop load fraction': 0.5,
})
params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']
mass_flow_rate(params)
calculate_primary_pump_mechanical_power(params)

# **************************************************************************************************************************
#                                                Sec. 8: Vessels (fixed geometry)
# **************************************************************************************************************************

update_params({
    'In Vessel Shield Thickness': 10.16,          # cm
    'In Vessel Shield Inner Radius': params['Core Radius'],
    'In Vessel Shield Material': 'B4C_natural',
    'Vessel Radius': params['Core Radius'] + 10.16,
    'Vessel Thickness': 1,                         # cm
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

params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

vessels_specs(params)
calculate_shielding_masses(params)

# **************************************************************************************************************************
#   Sec. 9: STEP 2 — Shielding Parametric Sweep (fixed-source transport)
# **************************************************************************************************************************

print("\n\n" + "="*70)
print("STEP 2: Shielding parametric sweep (fixed-source transport)")
print("="*70)

# Dose rate limit (NRC 10 CFR 20 public boundary: 2 mrem/hr = 0.02 mSv/hr)
# Using 2.5 mSv/hr as controlled area limit for workers during operation
update_params({
    'Photon Transport': True,
    'Dose Rate Limit mSv_hr': 0.02,      # mSv/hr — public boundary regulatory limit
    'Dose Rate Limit Workers mSv_hr': 2.5,  # mSv/hr — controlled area limit
    # Radii at which dose rate is evaluated (cm from core axis)
    'Dose Evaluation Radii cm': {
        'iso_surface':    None,   # filled dynamically from ISO outer radius
        '1m_standoff':    None,   # iso_surface + 100 cm
        '30m_exclusion':  3000,   # fixed: 30 m from core axis
    },
    # Fixed-source transport settings
    'Shielding Particles':  2_000_000,  # particles per batch for shielding run
    'Shielding Batches':    50,
    'Shielding Inactive':   0,          # no inactive batches in fixed-source mode
})

# ---- ISO container geometry (mobile case only) ----
# Standard High-Cube 40' ISO container interior: 12192 mm × 2352 mm × 2698 mm
# Corten steel wall thickness: ~1.9 mm outer skin + structural members.
# Conservatively modeled as a uniform cylindrical shell of equivalent steel.
ISO_CONTAINER_STEEL_THICKNESS_CM = 1.5   # cm effective steel (structural average)
ISO_CONTAINER_MATERIAL           = 'carbon_steel'

# ---- Parametric sweep ----
tracked_params_list = [
    'Mobile', 'Out Of Vessel Shield Material', 'Out Of Vessel Shield Thickness',
    'Isocontainer Steel Thickness',
    'Dose Rate iso_surface mSv_hr', 'Dose Rate 1m_standoff mSv_hr',
    'Dose Rate 30m_exclusion mSv_hr',
    'Meets Public Limit', 'Meets Worker Limit',
    'Fuel', 'Enrichment', 'Power MWt',
]

for params['Mobile'] in [False, True]:
    for params['Out Of Vessel Shield Material'] in ['WEP', 'B4C_natural', 'polyethylene']:
        for params['Out Of Vessel Shield Thickness'] in [20.0, 30.0, 39.37, 50.0]:  # cm

            mobile_tag = "MOBILE" if params['Mobile'] else "STATIONARY"
            print(f"\n--- {mobile_tag} | Shield: {params['Out Of Vessel Shield Material']} "
                  f"| Thickness: {params['Out Of Vessel Shield Thickness']} cm ---")

            # ---- Geometry: derive outer radii for this iteration ----
            params['Out Of Vessel Shield Thickness'] = params['Out Of Vessel Shield Thickness']
            params['Out Of Vessel Shield Inner Radius'] = params['In Vessel Shield Outer Radius'] \
                + sum([
                    params['Vessel Thickness'],
                    params['Gap Between Vessel And Guard Vessel'],
                    params['Guard Vessel Thickness'],
                    params['Gap Between Guard Vessel And Cooling Vessel'],
                    params['Cooling Vessel Thickness'],
                    params['Gap Between Cooling Vessel And Intake Vessel'],
                    params['Intake Vessel Thickness'],
                ])
            params['Out Of Vessel Shield Outer Radius'] = (
                params['Out Of Vessel Shield Inner Radius']
                + params['Out Of Vessel Shield Thickness']
            )

            if params['Mobile']:
                params['Isocontainer Steel Thickness'] = ISO_CONTAINER_STEEL_THICKNESS_CM
                params['Isocontainer Steel Material']  = ISO_CONTAINER_MATERIAL
                params['Isocontainer Inner Radius']    = params['Out Of Vessel Shield Outer Radius']
                params['Isocontainer Outer Radius']    = (
                    params['Isocontainer Inner Radius'] + params['Isocontainer Steel Thickness']
                )
                outer_boundary_r = params['Isocontainer Outer Radius']
            else:
                params['Isocontainer Steel Thickness'] = 0.0
                params['Isocontainer Steel Material']  = None
                params['Isocontainer Inner Radius']    = None
                params['Isocontainer Outer Radius']    = None
                outer_boundary_r = params['Out Of Vessel Shield Outer Radius']

            # Fill in the dynamic dose evaluation radii
            params['Dose Evaluation Radii cm']['iso_surface'] = outer_boundary_r
            params['Dose Evaluation Radii cm']['1m_standoff'] = outer_boundary_r + 100.0

            # ---- Flag shielding run for the template builder ----
            params['Shielding Run'] = True

            # ---- Run Step-2 fixed-source shielding transport ----
            run_openmc_shielding(build_openmc_shielding_model_LTMR, params)

            # ---- Report compliance ----
            dose_surface = params.get('Dose Rate iso_surface mSv_hr', float('nan'))
            dose_1m      = params.get('Dose Rate 1m_standoff mSv_hr', float('nan'))
            dose_30m     = params.get('Dose Rate 30m_exclusion mSv_hr', float('nan'))

            params['Meets Public Limit']  = dose_surface <= params['Dose Rate Limit mSv_hr']
            params['Meets Worker Limit']  = dose_surface <= params['Dose Rate Limit Workers mSv_hr']

            print(f"  Dose @ ISO surface : {dose_surface:.4e} mSv/hr  "
                  f"({'PASS' if params['Meets Public Limit'] else 'FAIL'} public limit)")
            print(f"  Dose @ 1m standoff : {dose_1m:.4e} mSv/hr")
            print(f"  Dose @ 30m exclusion: {dose_30m:.4e} mSv/hr")

            # ---- Save results ----
            summarize_shielding_results(
                params,
                tracked_params_list,
                os.path.join(params['shielding_output_dir'], 'output_shielding_study.csv')
            )

elapsed_time = (time.time() - time_start) / 60
print(f'\nTotal execution time: {np.round(elapsed_time, 2)} minutes')
