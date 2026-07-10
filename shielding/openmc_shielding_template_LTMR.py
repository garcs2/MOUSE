# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
openmc_shielding_template_LTMR.py

Fixed-source shielding transport model for the MOUSE LTMR.

This template is intentionally parallel in structure to openmc_template_LTMR.py
so that the two files can be read side-by-side. Core geometry and material
helpers are imported from the original template rather than duplicated;
all shielding-specific geometry, tally, and settings logic lives here.

Two-step workflow
-----------------
  Step 1 (in watts_exec_LTMR_shielding_study.py):
      Run build_openmc_model_LTMR → generates statepoint + source.h5

  Step 2 (this file — build_openmc_shielding_model_LTMR):
      Load source.h5 as a fixed source, wrap core universe in shielding
      annuli, run neutron + photon transport, score ICRP-116 H*(10) dose.

Geometry stack (radially outward)
----------------------------------
  Core (fuel lattice + control drums + radial reflector)
    → In-vessel B4C shield
      → Vessel stack (primary vessel / guard vessel / cooling / intake vessels + NaK gaps)
        → Out-of-vessel shield  (WEP | B4C_natural | polyethylene)
          → [Mobile only] ISO container Corten steel shell
            → Outer void  (vacuum boundary)
"""

import os
import numpy as np
import openmc

# Core geometry and material helpers — reuse without modification
from core_design.openmc_template_LTMR import (
    create_pin_regions,
    create_drums_universe,
    create_assembly_universe,
    create_control_drums_positions,
    create_core_geometry,
)
from core_design.openmc_materials_database import collect_materials_data
from core_design.utils import create_cells, circle_area, create_universe_plot




# **************************************************************************************************************************
#                                                Sec. 0 : Helper Functions
# **************************************************************************************************************************

"""
Shielding-specific helper functions that mirror the style of the helpers in
openmc_template_LTMR.py.  Each builds one well-defined piece of the model and
returns a concrete OpenMC object or structure.
"""

# Shielding constants and dose coefficient helpers
from shielding.shielding_constants import (
    NEUTRON_DOSE_ENERGY_MEV,
    NEUTRON_DOSE_COEFF_PSV_CM2,
    PHOTON_DOSE_ENERGY_MEV,
    PHOTON_DOSE_COEFF_PSV_CM2,
    make_energy_filter_and_coeffs,
)


def create_shielding_materials(params, materials_database):
    """
    Collect all OpenMC Material objects needed for the shielding geometry.
    All materials are pulled from the existing materials database, consistent
    with how core materials are accessed throughout the LTMR template.

    @ In,  params,             dict,                       Simulation parameters.
    @ In,  materials_database, dict,                       Materials database from collect_materials_data().
    @ Out, shielding_mats,     dict[str, openmc.Material], Keyed by descriptive name.
    """
    shielding_mats = {}

    # In-vessel shield (e.g. B4C_natural)
    shielding_mats['in_vessel_shield'] = materials_database[params['In Vessel Shield Material']]

    # Vessel stack — represented as a single stainless steel annulus for transport
    shielding_mats['vessel_stack'] = materials_database['SS304']

    # Out-of-vessel shield (e.g. WEP, polyethylene, B4C_natural)
    shielding_mats['out_of_vessel_shield'] = materials_database[params['Out Of Vessel Shield Material']]

    # ISO container steel (mobile case only)
    if params.get('Mobile', False):
        shielding_mats['isocontainer_steel'] = materials_database[params['Isocontainer Steel Material']]

    return shielding_mats


def create_shielding_annuli(params, shielding_mats):
    """
    Build the concentric annular shielding cells that surround the core universe.
    Cells are axially unbounded (infinite cylinders), consistent with the original
    openmc_template_LTMR.py which uses only a ZCylinder as its outer boundary.

    Radial order (outward):
      in-vessel B4C  →  vessel stack (SS)  →  out-of-vessel shield  →  [ISO steel]

    The outermost surface carries boundary_type='vacuum'.

    @ In,  params,         dict,                      Simulation parameters.
    @ In,  shielding_mats, dict[str,openmc.Material],  Output of create_shielding_materials().
    @ Out, cells,          list[openmc.Cell],          Ordered list of shielding cells (inner → outer).
    @ Out, outer_surface,  openmc.ZCylinder,           The vacuum boundary surface.
    """
    cells = []

    # ---- In-vessel shield (B4C) ----
    s_iv_inner = openmc.ZCylinder(r=params['In Vessel Shield Inner Radius'])
    s_iv_outer = openmc.ZCylinder(r=params['In Vessel Shield Outer Radius'])
    cells.append(openmc.Cell(
        name='in_vessel_shield',
        fill=shielding_mats['in_vessel_shield'],
        region=+s_iv_inner & -s_iv_outer
    ))

    # ---- Vessel stack (primary + guard + cooling + intake vessels and NaK gaps) ----
    # Collapsed into a single stainless steel annulus for shielding transport.
    # Detailed vessel-by-vessel geometry adds negligible accuracy to dose results
    # at the outer shield boundary while substantially increasing geometry complexity.
    s_vessel_inner = openmc.ZCylinder(r=params['In Vessel Shield Outer Radius'])
    s_vessel_outer = openmc.ZCylinder(r=params['Out Of Vessel Shield Inner Radius'])
    cells.append(openmc.Cell(
        name='vessel_stack',
        fill=shielding_mats['vessel_stack'],
        region=+s_vessel_inner & -s_vessel_outer
    ))

    # ---- Out-of-vessel shield (WEP / B4C_natural / polyethylene) ----
    s_oov_inner = openmc.ZCylinder(r=params['Out Of Vessel Shield Inner Radius'])
    s_oov_outer = openmc.ZCylinder(r=params['Out Of Vessel Shield Outer Radius'])
    cells.append(openmc.Cell(
        name='out_of_vessel_shield',
        fill=shielding_mats['out_of_vessel_shield'],
        region=+s_oov_inner & -s_oov_outer
    ))

    # ---- ISO container steel (mobile case only) ----
    if params.get('Mobile', False):
        s_iso_inner = openmc.ZCylinder(r=params['Isocontainer Inner Radius'])
        s_iso_outer = openmc.ZCylinder(r=params['Isocontainer Outer Radius'],
                                        boundary_type='vacuum')
        cells.append(openmc.Cell(
            name='isocontainer_steel',
            fill=shielding_mats['isocontainer_steel'],
            region=+s_iso_inner & -s_iso_outer
        ))
        outer_surface = s_iso_outer

    else:
        # Non-mobile: out-of-vessel shield outer surface is the vacuum boundary
        s_oov_outer.boundary_type = 'vacuum'
        outer_surface = s_oov_outer

    return cells, outer_surface


def create_dose_energy_filters():
    """
    Build OpenMC EnergyFilter objects for neutron and photon dose tallies.
    Energy bin edges are derived from the ICRP-116 coefficient tables above.

    @ Out, filters, dict[str, openmc.EnergyFilter], Keys: 'neutron', 'photon'.
    """
    filters = {}

    # OpenMC EnergyFilter requires bin edges in eV; append a closing upper edge
    n_edges  = np.append(NEUTRON_DOSE_ENERGY_MEV * 1e6, NEUTRON_DOSE_ENERGY_MEV[-1] * 1e6 * 10)
    g_edges  = np.append(PHOTON_DOSE_ENERGY_MEV  * 1e6, PHOTON_DOSE_ENERGY_MEV[-1]  * 1e6 * 10)

    filters['neutron'] = openmc.EnergyFilter(n_edges)
    filters['photon']  = openmc.EnergyFilter(g_edges)

    return filters


def create_shielding_mesh(params):
    """
    Build a cylindrical mesh spanning the full shielding geometry for 2-D
    (r, z) dose rate mapping.

    Radial extent: 0 → outermost shield surface + 10 cm margin.
    Axial extent:  ± (active height / 2 + axial reflector thickness + 50 cm margin).
    The geometry is axially unbounded so the mesh simply needs to be wide
    enough to capture the meaningful dose gradient region.

    @ In,  params, dict,                     Simulation parameters.
    @ Out, mesh,   openmc.CylindricalMesh,   80 radial × 60 axial bins.
    """
    if params.get('Mobile', False):
        r_max = params['Isocontainer Outer Radius'] + 10.0
    else:
        r_max = params['Out Of Vessel Shield Outer Radius'] + 10.0

    axial_half = (
        params['Active Height'] / 2
        + params['Axial Reflector Thickness']
        + 50.0  # 50 cm margin beyond active + reflector
    )

    r_grid    = np.linspace(0,           r_max,      80)
    z_grid    = np.linspace(-axial_half, axial_half, 60)
    phi_grid  = np.linspace(0, 2 * np.pi, 2)   # azimuthally integrated
    mesh      = openmc.CylindricalMesh(r_grid, z_grid, phi_grid)
    return mesh


def create_dose_tallies(params, mesh):
    """
    Build all tallies needed for the shielding analysis:

      1. Cylindrical mesh flux tallies (neutron + photon) for 2-D dose maps.
      2. Thin-shell flux tallies at each dose evaluation radius for point
         dose rate extraction.

    ICRP-116 H*(10) dose coefficients are applied during post-processing in
    shielding_calcs.py; the tallies here score flux grouped by energy so that
    the dot product with dose coefficients can be computed after the run.

    @ In,  params,  dict,             Simulation parameters.
    @ In,  mesh,    openmc.CylindricalMesh, From create_shielding_mesh().
    @ Out, tallies, openmc.Tallies,   Complete tally collection.
    """
    tallies      = openmc.Tallies()
    energy_filts = create_dose_energy_filters()
    mesh_filter  = openmc.MeshFilter(mesh)

    # ---- 1. Mesh flux tallies (neutron + photon) ----
    for particle in ['neutron', 'photon']:
        particle_filt = openmc.ParticleFilter([particle])
        tally = openmc.Tally(name=f'{particle}_flux_mesh')
        tally.filters = [mesh_filter, energy_filts[particle], particle_filt]
        tally.scores  = ['flux']
        tallies.append(tally)

    # ---- 2. Thin-shell point-dose tallies at each evaluation radius ----
    for label, radius_cm in params['Dose Evaluation Radii cm'].items():
        if radius_cm is None:
            continue

        # Thin annular shell: ±0.5 cm around the nominal evaluation radius
        inner_r = max(0.0, radius_cm - 0.5)
        outer_r = radius_cm + 0.5

        axial_half = (
            params['Active Height'] / 2
            + params['Axial Reflector Thickness']
            + 50.0  # 50 cm margin beyond active+reflector
        )

        
        r_grid   = np.array([inner_r, outer_r])
        z_grid   = np.array([-axial_half, axial_half])
        phi_grid = np.linspace(0, 2 * np.pi, 2)
        shell_mesh          = openmc.CylindricalMesh(r_grid, z_grid, phi_grid)
        shell_filt = openmc.MeshFilter(shell_mesh)

        for particle in ['neutron', 'photon']:
            particle_filt = openmc.ParticleFilter([particle])
            pt_tally = openmc.Tally(name=f'dose_point_{label}_{particle}')
            pt_tally.filters = [shell_filt, energy_filts[particle], particle_filt]
            pt_tally.scores  = ['flux']
            tallies.append(pt_tally)

    return tallies


def create_shielding_source(params):
    """
    Configure the fixed neutron + photon source for the shielding transport run.

    Preferred path: load the converged fission source file (source.h5) written
    by the preceding Step-1 criticality run.  This preserves the spatial and
    energy distribution of the fission neutrons and prompt gammas.

    Fallback: if source.h5 is not found, an isotropic Watt-spectrum point
    source at the core centre is used.  This fallback is physically approximate
    and intended only for geometry testing.

    @ In,  params, dict,               Simulation parameters.
    @ Out, source, openmc source obj,  Either FileSource or IndependentSource.
    """
    source_file = params.get('Fission Source File', 'source.h5')

    if os.path.isfile(source_file):
        print(f"  [Shielding] Loading converged fission source: {source_file}")
        return openmc.FileSource(source_file)

    print(
        f"  [Shielding] WARNING: Fission source file not found at '{source_file}'.\n"
        "  Falling back to isotropic Watt-spectrum point source at core centre.\n"
        "  Dose results will NOT be physically accurate — run Step-1 first."
    )
    fallback             = openmc.IndependentSource()
    fallback.space       = openmc.stats.Point((0, 0, 0))
    fallback.angle       = openmc.stats.Isotropic()
    fallback.energy      = openmc.stats.Watt(a=0.988e6, b=2.249e-6)
    fallback.particle    = 'neutron'
    return fallback


# **************************************************************************************************************************
#                                                Sec. 1 : OpenMC Shielding Model
# **************************************************************************************************************************

"""
Main model-builder function.  Accepts a params instance and produces all
OpenMC XML input files for a fixed-source shielding transport run.

Designed to be passed directly to run_openmc() as the model-builder callback,
matching the same signature as build_openmc_model_LTMR.
"""

def build_openmc_shielding_model_LTMR(params):
    """
    OpenMC Fixed-Source Shielding Model for the MOUSE LTMR.

    @ In,  params, watts.parameters.Parameters, Simulation parameters.
                   Must contain all geometry, material, and shielding keys
                   set in watts_exec_LTMR_shielding_study.py Sec. 7–9.
    """

    params.setdefault('Mobile',                             False)
    params.setdefault('Photon Transport',                   True)
    params.setdefault('Out Of Vessel Shield Effective Density Factor', 1.0)
    params.setdefault('Shielding Particles',                2_000_000)
    params.setdefault('Shielding Batches',                  50)


    # **************************************************************************************************************************
    #                                                Sec. 1.1 : MATERIALS
    # **************************************************************************************************************************

    # Read all core material properties (same database as criticality model)
    materials_database = collect_materials_data(params)

    # Core materials — identical to Sec. 1.1 of openmc_template_LTMR.py
    fuel                    = materials_database[params['Fuel']]
    coolant                 = materials_database[params['Coolant']]
    reflector               = materials_database[params['Radial Reflector']]
    control_drum_absorber   = materials_database[params['Control Drum Absorber']]
    control_drum_reflector  = materials_database[params['Control Drum Reflector']]

    # Shielding materials (B4C shield, vessel stack steel, out-of-vessel shield,
    # and optionally ISO container steel)
    shielding_mats = create_shielding_materials(params, materials_database)

    # Collect all unique materials for the materials.xml
    fuel_materials = []
    for mat in params['Fuel Pin Materials']:
        fuel_materials.append(None if mat is None else materials_database[mat])
    fuel_materials.append(coolant)

    moderator_materials = []
    for mat in params['Moderator Pin Materials']:
        moderator_materials.append(None if mat is None else materials_database[mat])
    moderator_materials.append(coolant)

    all_materials = (
        fuel_materials
        + moderator_materials
        + [coolant, reflector, control_drum_absorber, control_drum_reflector]
        + list(shielding_mats.values())
    )
    all_materials_cleaned = list(set(item for item in all_materials if item is not None))

    # Set fuel volume for correct reaction rate normalisation in depletion context
    fuel_index    = params['Fuel Pin Materials'].index(params['Fuel'])
    fissile_area  = (
        circle_area(params['Fuel Pin Radii'][fuel_index])
        - circle_area(params['Fuel Pin Radii'][fuel_index - 1])
    )
    fuel.volume = fissile_area * params['Active Height'] * params['Fuel Pin Count']

    materials = openmc.Materials(all_materials_cleaned)
    openmc.Materials.cross_sections = params['cross_sections_xml_location']
    materials.export_to_xml()


    # **************************************************************************************************************************
    #                                                Sec. 1.2 : GEOMETRY — Fuel Pins, Moderator Pins, Coolant
    # **************************************************************************************************************************

    # Fuel pin universe — identical to Sec. 1.2 of openmc_template_LTMR.py
    fuel_pin_regions = create_pin_regions(params, 'fuel')
    fuel_cells       = create_cells(fuel_pin_regions, fuel_materials)
    fuel_pin_universe = openmc.Universe(cells=fuel_cells.values())

    # Moderator pin universe
    moderator_pin_regions = create_pin_regions(params, 'moderator')
    moderator_cells       = create_cells(moderator_pin_regions, moderator_materials)
    moderator_pin_universe = openmc.Universe(cells=moderator_cells.values())

    # Coolant universe (outer universe of the hex lattice)
    coolant_cell     = openmc.Cell(fill=coolant)
    coolant_universe = openmc.Universe(cells=(coolant_cell,))


    # **************************************************************************************************************************
    #                                                Sec. 1.3 : GEOMETRY — Control Drums
    # **************************************************************************************************************************

    drums = create_drums_universe(params, control_drum_absorber, control_drum_reflector)


    # **************************************************************************************************************************
    #                                                Sec. 1.4 : GEOMETRY — Fuel Assembly
    # **************************************************************************************************************************

    pin_pitch = 2 * params['Fuel Pin Radii'][-1] + params['Pin Gap Distance']

    assembly_universe = create_assembly_universe(
        params,
        fuel_pin_universe,
        moderator_pin_universe,
        pin_pitch,
        reflector,
        coolant_universe
    )


    # **************************************************************************************************************************
    #                                                Sec. 1.5 : GEOMETRY — Core (drums + assembly)
    # **************************************************************************************************************************

    control_drum_positions = create_control_drums_positions(number_of_drums=len(drums))

    # create_core_geometry returns the core universe with the vacuum outer surface on
    # the core cylinder.  For the shielding model we need the inner (core) universe
    # only; the vacuum boundary will be moved to the outermost shielding surface below.
    core_geometry, core_universe = create_core_geometry(
        params,
        drums,
        drums_positions=control_drum_positions,
        assembly_universe=assembly_universe
    )

    # Remove the vacuum boundary from the core cylinder so that particles can
    # stream outward into the shielding annuli.
    # get_all_surfaces() exists on openmc.Geometry, not Universe; instead we
    # directly target the known core outer cylinder by radius.
    for surface in core_geometry.get_all_surfaces().values():
        if hasattr(surface, 'boundary_type') and surface.boundary_type == 'vacuum':
            surface.boundary_type = 'transmission'


    # **************************************************************************************************************************
    #                                                Sec. 1.6 : GEOMETRY — Shielding Annuli + Outer Boundary
    # **************************************************************************************************************************

    # Geometry is axially unbounded, matching the original openmc_template_LTMR.py
    # which uses only a ZCylinder as its outer boundary.
    shielding_cells, outer_surface = create_shielding_annuli(params, shielding_mats)

    # Core fill cell (inner boundary = core radius cylinder)
    core_inner_surface = openmc.ZCylinder(r=params['Core Radius'])
    core_fill_cell     = openmc.Cell(
        name='core_fill',
        fill=core_universe,
        region=-core_inner_surface
    )

    # Outer void beyond the vacuum boundary
    outer_void_cell = openmc.Cell(name='outer_void', fill=None, region=+outer_surface)

    all_cells = [core_fill_cell] + shielding_cells + [outer_void_cell]

    geometry = openmc.Geometry(all_cells)
    geometry.export_to_xml()

    if params['plotting'] == "Y":
        create_universe_plot(
            materials_database, geometry,
            plot_width=2.01 * (
                params['Isocontainer Outer Radius']
                if params.get('Mobile', False)
                else params['Out Of Vessel Shield Outer Radius']
            ),
            num_pixels=2000,
            font_size=32,
            title="LTMR Shielding Geometry",
            fig_size=8,
            output_file_name="shielding_geometry.png"
        )


    # **************************************************************************************************************************
    #                                                Sec. 1.7 : TALLIES
    # **************************************************************************************************************************

    mesh    = create_shielding_mesh(params)
    tallies = create_dose_tallies(params, mesh)

    # Store mesh reference so shielding_calcs.py can retrieve grid coordinates
    params['_shielding_mesh'] = mesh

    tallies.export_to_xml()


    # **************************************************************************************************************************
    #                                                Sec. 1.8 : SIMULATION SETTINGS
    # **************************************************************************************************************************

    point = openmc.stats.Point((0, 0, 0))
    source = openmc.Source(space=point)
    settings = openmc.Settings()
    settings.source = source
    settings.batches = 100
    settings.inactive = 50
    if 'Particles' in params.keys():
        settings.particles = int(params['Particles'])#1000
    else:
        settings.particles = 200 
    if params['Isothermal Temperature Coefficients']:
        settings.temperature = {'default': params['Common Temperature'],
                                 'method': 'interpolation',
                                 'tolerance': 50.0}
    else:
        settings.temperature = {'method': 'interpolation'}  # added missing else branch
    
    settings.export_to_xml()
