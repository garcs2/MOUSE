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
import shutil
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
from shielding.shielding_analysis import run_openmc_shielding, run_bol_source_run
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


PERSISTENT_DIR = '/home/garcsamu/OpenMC/MOUSE/shielding/testing_xml_output'
 
 
# def run_func():
#     plot = openmc.Plot()
#     plot.basis = 'xy'
#     plot.origin = (0, 0, 0)
#     plot.width = (400, 400)      # cm — adjust to comfortably frame your geometry
#     plot.pixels = (2000, 2000)
#     plot.color_by = 'material'
#     openmc.Plots([plot]).export_to_xml()
 
#     openmc.plot_geometry()
 
#     os.makedirs(PERSISTENT_DIR, exist_ok=True)
#     for pattern in ('*.xml', '*.ppm', '*.png'):
#         for f in glob.glob(pattern):
#             shutil.copy2(f, os.path.join(PERSISTENT_DIR, f))
 
#     print(f"  [Testing] XML + plot files copied to: {PERSISTENT_DIR}")

def build_layer1_model(params):
    resolve_drum_radius(params)
    materials_database = collect_materials_data(params)
 
    fuel = materials_database[params['Fuel']]
    coolant = materials_database[params['Coolant']]
    reflector = materials_database[params['Radial Reflector']]
    control_drum_absorber = materials_database[params['Control Drum Absorber']]
    control_drum_reflector = materials_database[params['Control Drum Reflector']]
 
    fuel_materials = [None if m is None else materials_database[m] for m in params['Fuel Pin Materials']]
    fuel_materials.append(coolant)
    moderator_materials = [None if m is None else materials_database[m] for m in params['Moderator Pin Materials']]
    moderator_materials.append(coolant)
 
    all_materials = fuel_materials + moderator_materials + [coolant, reflector, control_drum_absorber, control_drum_reflector]
    all_materials_cleaned = list(set(m for m in all_materials if m is not None))
    materials = openmc.Materials(all_materials_cleaned)
    openmc.Materials.cross_sections = params['cross_sections_xml_location']
    materials.export_to_xml()
 
    fuel_pin_regions = create_pin_regions(params, 'fuel')
    fuel_cells = create_cells(fuel_pin_regions, fuel_materials)
    fuel_pin_universe = openmc.Universe(cells=fuel_cells.values())
 
    moderator_pin_regions = create_pin_regions(params, 'moderator')
    moderator_cells = create_cells(moderator_pin_regions, moderator_materials)
    moderator_pin_universe = openmc.Universe(cells=moderator_cells.values())
 
    coolant_cell = openmc.Cell(fill=coolant)
    coolant_universe = openmc.Universe(cells=(coolant_cell,))
 
    control_drum_positions = update_ltmr_reflector_geometry_from_drums(params)
    drums = create_drums_universe(params, control_drum_absorber, control_drum_reflector, control_drum_positions)
 
    pin_pitch = 2 * params['Fuel Pin Radii'][-1] + params['Pin Gap Distance']
    assembly_universe = create_assembly_universe(params, fuel_pin_universe, moderator_pin_universe, pin_pitch, reflector, coolant_universe)
 
    core_geometry, core_universe = create_core_geometry(
        params, drums, drums_positions=control_drum_positions, assembly_universe=assembly_universe
    )
 
    # Same boundary-type patch the shielding template applies
    for surface in core_geometry.get_all_surfaces().values():
        if hasattr(surface, 'boundary_type') and surface.boundary_type == 'vacuum':
            surface.boundary_type = 'transmission'
 
    # Same wrapping the shielding template applies — but with a plain large
    # vacuum outer boundary instead of real shielding annuli
    core_inner_surface = openmc.ZCylinder(r=params['Core Radius'])
    core_fill_cell = openmc.Cell(name='core_fill', fill=core_universe, region=-core_inner_surface)
 
    outer_surface = openmc.ZCylinder(r=params['Core Radius'] * 3.0, boundary_type='vacuum')
    outer_ring_cell = openmc.Cell(name='outer_ring_void', region=+core_inner_surface & -outer_surface)
    outer_void_cell = openmc.Cell(name='outer_void', fill=None, region=+outer_surface)
 
    geometry = openmc.Geometry([core_fill_cell, outer_ring_cell, outer_void_cell])
    geometry.export_to_xml()
 
    settings = openmc.Settings()
    settings.batches = 10
    settings.inactive = 5
    settings.particles = 100
    settings.source = openmc.Source(space=openmc.stats.Point((0, 0, 0)))
    settings.export_to_xml()
 
    openmc.Tallies([]).export_to_xml()
 
    params['_layer1_materials_database'] = materials_database
    params['_layer1_geometry'] = geometry
 
 
def run_func():
    build_layer1_model(params)
    create_universe_plot(
        params['_layer1_materials_database'], params['_layer1_geometry'],
        plot_width=2.01 * params['Core Radius'],
        num_pixels=2000,
        font_size=32,
        title="Layer 1: Core wrapped like shielding template (no shield materials)",
        fig_size=8,
        output_file_name="layer1_wrapped_core.png"
    )
    os.makedirs(PERSISTENT_DIR, exist_ok=True)
    for f in glob.glob('*.png') + glob.glob('*.xml'):
        shutil.copy2(f, os.path.join(PERSISTENT_DIR, f))
    print(f"  [Layer 1] Copied outputs to: {PERSISTENT_DIR}")






# **************************************************************************************************************************
#                                                Sec. 0: Settings
# **************************************************************************************************************************

update_params({
    'plotting': "Y",  # Shielding runs are expensive; disable geometry plots by default
    'cross_sections_xml_location': '/home/garcsamu/OpenMC/cross_sections/endfb-viii.1-hdf5/cross_sections.xml',
    'simplified_chain_thermal_xml': '/home/garcsamu/OpenMC/TEMA/data/chain_casl_pwr.xml',
    'shielding_output_dir': '/home/garcsamu/OpenMC/MOUSE',
    'XS_type': 'endf8.1'
})

os.makedirs(params['shielding_output_dir'], exist_ok=True)

# **************************************************************************************************************************
#                                                Sec. 1: Materials (fixed for shielding study)
# **************************************************************************************************************************

update_params({
    'reactor type': "LTMR",
    'Fuel': 'UZrH_alloy',               # Fix fuel type for shielding study; change as needed
    'Enrichment': 0.1975,
    'TRISO Fueled': "No",
    "H_Zr_ratio": 1.6,
    'U_met_wo': 0.3,
    'Coolant': 'NaK',
    'er_wo': 0,
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
    'Radial Reflector Thickness': 14,  # cm
})

params['Lattice Apothem'] = calculate_hex_apothem(params)
params['Lattice Radius']            = params['Lattice Apothem']
params['Active Height']             = 78.4   # cm1
params['Assembly FTF']              = 2 * params['Lattice Apothem']
params['Axial Reflector Thickness'] = params['Radial Reflector Thickness']
params['Fuel Pin Count']            = calculate_pins_in_assembly(params, "FUEL")
params['Moderator Pin Count']       = calculate_pins_in_assembly(params, "MODERATOR")
params['Moderator Mass']            = calculate_moderator_mass(params)
params['Core Radius']               = calculate_core_radius_from_hex(params)

# **************************************************************************************************************************
#                                                Sec. 3: Control Drums (fixed)
# **************************************************************************************************************************

update_params({
    'Number of Drums': 12,
    # 'Drum Radius': 9.016,  # cm
    'Drum Absorber Thickness': 1,  # cm
    'Drum Absorber Arc Degrees': 120,
    'Drum Height': params['Active Height'] + 2 * params['Axial Reflector Thickness'],
})

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
    'Particles': 1000
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
    'Out Of Vessel Shield Effective Density Factor': 0.5
})

params['In Vessel Shield Outer Radius'] = params['Core Radius'] + params['In Vessel Shield Thickness']

vessels_specs(params)

# **************************************************************************************************************************
#                                                Sec. 8b: Operation (fixed for shielding study)
# **************************************************************************************************************************

update_params({
    'Operation Mode': "Autonomous",
    'Number of Operators': 2,
    'Levelization Period': 60,  # years
    'Refueling Period': 7,
    'Emergency Shutdowns Per Year': 0.2,
    'Startup Duration after Refueling': 2,
    'Startup Duration after Emergency Shutdown': 14,
    'Reactors Monitored Per Operator': 10,
    'Security Staff Per Shift': 2 if params['Enrichment'] > 0.1 else 1,
    'FTEs Per Onsite Operator Per Year': 1,
    })
## Calculated based on 1 tank; Density of NaK = 855 kg/m^3, Volume = 8.2402 m^3 (standard tank size)
params['Onsite Coolant Inventory']      = 1 * 855 * 8.2402  # kg
params['Replacement Coolant Inventory'] = 0  # NaK assumed not to need replacement


# **************************************************************************************************************************
#                                                Sec. 8c: Buildings & Economic Parameters (fixed for shielding study)
# **************************************************************************************************************************

update_params({
    'Land Area': 18,               # acres
    'Escalation Year': 2024,
    'Discount Rate': 0.07,
    'Excavation Volume': 412.605,  # m^3

    'Reactor Building Slab Roof Volume':      (9750 * 6502.4 * 1500) / 1e9,
    'Reactor Building Basement Volume':       (9750 * 6502.4 * 1500) / 1e9,
    'Reactor Building Exterior Walls Volume': ((2 * 9750 * 3500 * 1500) + (3502.4 * 3500 * (1500 + 750))) / 1e9,
    'Reactor Building Superstructure Area':   ((2 * 3500 * 3500) + (2 * 7500 * 3500)) / 1e6,

    'Integrated Heat Exchanger Building Slab Roof Volume':      0,
    'Integrated Heat Exchanger Building Basement Volume':       0,
    'Integrated Heat Exchanger Building Exterior Walls Volume': 0,
    'Integrated Heat Exchanger Building Superstructure Area':   0,

    'Turbine Building Slab Roof Volume':      (12192 * 2438 * 200) / 1e9,
    'Turbine Building Basement Volume':       (12192 * 2438 * 200) / 1e9,
    'Turbine Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,

    'Control Building Slab Roof Volume':      (12192 * 2438 * 200) / 1e9,
    'Control Building Basement Volume':       (12192 * 2438 * 200) / 1e9,
    'Control Building Exterior Walls Volume': ((12192 * 2496 * 200) + (2038 * 2496 * 200)) * 2 / 1e9,

    'Manipulator Building Slab Roof Volume':      (4876.8 * 2438.4 * 400) / 1e9,
    'Manipulator Building Basement Volume':       (4876.8 * 2438.4 * 1500) / 1e9,
    'Manipulator Building Exterior Walls Volume': ((4876.8 * 4445 * 400) + (2038.4 * 4445 * 400 * 2)) / 1e9,

    'Refueling Building Slab Roof Volume':      0,
    'Refueling Building Basement Volume':       0,
    'Refueling Building Exterior Walls Volume': 0,

    'Spent Fuel Building Slab Roof Volume':      0,
    'Spent Fuel Building Basement Volume':       0,
    'Spent Fuel Building Exterior Walls Volume': 0,

    'Emergency Building Slab Roof Volume':      0,
    'Emergency Building Basement Volume':       0,
    'Emergency Building Exterior Walls Volume': 0,

    'Storage Building Slab Roof Volume':      (8400 * 3500 * 400) / 1e9,
    'Storage Building Basement Volume':       (8400 * 3500 * 400) / 1e9,
    'Storage Building Exterior Walls Volume': ((8400 * 2700 * 400) + (3100 * 2700 * 400 * 2)) / 1e9,

    'Radwaste Building Slab Roof Volume':      0,
    'Radwaste Building Basement Volume':       0,
    'Radwaste Building Exterior Walls Volume': 0,

    'Interest Rate': 0.07,
    'Discount Rate': 0.07,
    'Construction Duration': 12,   # months
    'Debt To Equity Ratio': 0.5,
    'Annual Return': 0.0475,       # decommissioning reserve fund return
    'NOAK Unit Number': 100,
})

params['Number of Samples'] = 1000  # Monte Carlo samples for cost uncertainty (raise for tighter CIs, at compute cost)

# cost_database_filename = '/home/garcsamu/OpenMC/MOUSE/cost/Cost_Database.xlsx'
cost_tracked_params_list = [
    'Mobile', 'Out Of Vessel Shield Material', 'Out Of Vessel Shield Thickness',
    'Isocontainer Steel Thickness',
    'In Vessel Shield Mass', 'Out Of Vessel Shield Mass',
    'Dose Rate iso_surface mSv_hr', 'Dose Rate 1m_standoff mSv_hr', 'Dose Rate 30m_exclusion mSv_hr',
    'Meets Public Limit', 'Meets Worker Limit',
    'Fuel', 'Enrichment', 'Power MWt',
    'Cost_Data'
]

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
    'Shielding Particles':  20_000,  # particles per batch for shielding run
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
update_params({
    'Out Of Vessel Shield Material': 'B4C_natural',
    'Mobile': True,
    'Out Of Vessel Shield Thickness': 50,
    'Shutdown Margin Calc': False
    })

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
calculate_shielding_masses(params)
# Fill in the dynamic dose evaluation radii
params['Dose Evaluation Radii cm']['iso_surface'] = outer_boundary_r
params['Dose Evaluation Radii cm']['1m_standoff'] = outer_boundary_r + 100.0
# openmc_plugin = watts.PluginOpenMC(build_openmc_model_LTMR)
openmc_plugin = watts.PluginOpenMC(build_layer1_model, show_stdout=True, show_stderr=True)
openmc_result = openmc_plugin(params, function=run_func)
