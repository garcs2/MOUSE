# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
core_design_3D/abc_analysis_3D.py

Temperature / reflector / coolant reactivity-coefficient analysis for the 3D
LTMR, with automatic thermal geometric expansion (core_thermal_geometry.py), plus
the quasi-static A/B/C inherent-safety screen.

Each k-eff is now a SINGLE steady-state eigenvalue run (no depletion), so you can
run with high particle counts cheaply. Control statistics through the OpenMC
settings written in build_openmc_model_LTMR_3D (e.g. params['Particles'],
params['Batches'], params['Inactive Batches']).

Wire into utils_3D.run_openmc with:

    params.setdefault('ABC Analysis', False)
    ...
    # inside the try block, before the Shutdown-Margin / ITC branching:
    if params['ABC Analysis']:
        if 'Temperature Perturbation' not in params:
            raise ValueError("ABC Analysis requires 'Temperature Perturbation' (K).")
        from core_design_3D.abc_analysis_3D import run_abc_analysis
        run_abc_analysis(build_openmc_model, params)
        return   # the existing 'finally' restores Common Temperature
"""
import glob

import numpy as np
import openmc

# ----------------------------------------------------------------------------------
#  MPI-safe helpers (degrade to a single rank if mpi4py is unavailable)
# ----------------------------------------------------------------------------------
def _mpi_rank():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def _mpi_barrier():
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass


def _mpi_bcast(obj, root=0):
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.bcast(obj, root=root)
    except ImportError:
        return obj


# ----------------------------------------------------------------------------------
#  Single steady-state eigenvalue run (replaces the depletion path for ABC)
# ----------------------------------------------------------------------------------
def run_steady_state(params):
    """Run ONE fixed-geometry OpenMC eigenvalue case (no depletion) for the
    CURRENT params and store the resulting k-eff.

    The watts plugin prerun has already written geometry/materials/settings XML
    for this state, so this just runs OpenMC once and reads k-eff from the final
    statepoint. The value is stored as single-element lists under 'keff 2D' and
    'keff 3D (2D corrected)' so the coefficient finite-difference logic (which
    zips over what used to be burnup steps) is unchanged. In the genuine-3D model
    the two are the same single real 3D eigenvalue.
    """
    _mpi_barrier()
    if 'cross_sections_xml_location' in params:
        openmc.config['cross_sections'] = params['cross_sections_xml_location']

    # Uses the XML already written in the run directory; returns the last statepoint.
    sp_path = openmc.run()

    keff = np.nan
    if _mpi_rank() == 0:
        if not sp_path:
            candidates = sorted(glob.glob('statepoint.*.h5'))
            sp_path = candidates[-1] if candidates else None
        if sp_path:
            with openmc.StatePoint(sp_path) as sp:
                keff = float(sp.keff.nominal_value)
                params['keff std_dev'] = float(sp.keff.std_dev)
    _mpi_barrier()
    keff = _mpi_bcast(keff, root=0)

    params['keff 2D'] = keff
    params['keff 3D (2D corrected)'] = keff
    return keff


def _run_model(build_openmc_model, params):
    """Run one steady-state eigenvalue case honoring the CURRENT params
    (geometry + temperatures). Returns (keff_2d_list, keff_3d_corrected_list)."""
    import watts
    openmc_plugin = watts.PluginOpenMC(build_openmc_model, show_stderr=True, show_stdout=True)
    openmc_plugin(params, function=lambda: run_steady_state(params))
    return params['keff 2D'], params['keff 3D (2D corrected)'], params['keff std_dev']


def _coefficient(keff_base, keff_pert, sd_base, sd_pert, dT):
    C = 1e5 / dT
    alpha = C * (1.0/keff_base - 1.0/keff_pert)
    sigma = C * np.sqrt((sd_base/keff_base**2)**2 + (sd_pert/keff_pert**2)**2)
    return alpha, sigma


def _set_region_temperatures(params, T_fuel, T_reflector, T_coolant, T_common):
    """Propagate per-region temperatures to params.

    If params['Per-Region Temperatures'] is True (requires the materials-module
    edit so collect_materials_data honors these keys), each region uses its own
    temperature and everything else (structure, moderator, absorbers) uses
    T_common. This isolates the density/Doppler part of each coefficient.

    If False (unpatched materials module), Common Temperature is set to the
    perturbed region's temperature so at least the dominant density/Doppler
    effect is captured (NOT isolated). The geometric part is isolated either way,
    since expand_derived_geometry T_fuel / T_reflector directly.
    """
    params['Fuel Temperature'] = T_fuel
    params['Reflector Temperature'] = T_reflector
    params['Coolant Temperature'] = T_coolant
    if params.get('Per-Region Temperatures', False):
        params['Common Temperature'] = T_common          # structure/moderator at base
    else:
        params['Common Temperature'] = max(T_fuel, T_reflector, T_coolant)


def _evaluate_abc_criteria(params):
    """Quasi-static reactivity-balance (A/B/C integral parameters) inherent-safety
    screen. A and B are in pcm; C is in pcm/K. Uses the 2D coefficients, which
    equal the 3D-corrected values in the genuine-3D scheme.

    This is a nominal screen based on the quasi-static reactivity balance, not a
    substitute for a full transient analysis. The dT used in the temperature-rise
    criteria is the perturbation 'Temperature Perturbation'; if you intend a
    distinct coolant temperature rise, substitute that quantity.
    """
    dT_f = params['Fuel-Coolant dT']
    dT_c = params['Primary Loop Outlet Temperature'] - params['Primary Loop Inlet Temperature']
    a_T = params['Temp Coeff 2D']
    a_C = params['Coolant Coeff 2D']
    a_R = params['Reflector Coeff 2D']

    A = a_T * dT_f                          # pcm
    B = dT_c * (a_T / 2 + a_C / 2 + a_R)    # pcm
    C = a_T + a_C + a_R                    # pcm/K
    params['ABC A (pcm)'] = A
    params['ABC B (pcm)'] = B
    params['ABC C (pcm/K)'] = C

    # --- Criterion 1: integral power/temperature parameters all negative ---
    criterion_1 = (A < 0) and (B < 0) and (C < 0)
    if not criterion_1:
        print("Warning Criterion 1: A, B, C are not all negative:\n"
              f"  A = {A:.3f} pcm\n  B = {B:.3f} pcm\n  C = {C:.3f} pcm/K")

    # --- Criterion 2: loss-of-flow asymptotic temperature below coolant boiling ---
    T_boil = params.get('Coolant Boiling Temperature', 1156.15)  # K (NaK ~785 C)
    if 'Primary Loop Inlet Temperature' in params and B != 0:
        lof_temp = A / B * dT_c + params['Primary Loop Outlet Temperature']
        criterion_2 = lof_temp < T_boil
        if not criterion_2:
            print("Warning Criterion 2: loss-of-flow temperature reaches/exceeds "
                  f"coolant boiling temperature:\n  T_LOF = {lof_temp:.2f} K  >=  "
                  f"T_boil = {T_boil:.2f} K")
    else:
        criterion_2 = False
        print("Warning Criterion 2: cannot evaluate (need 'Primary Loop Inlet "
              "Temperature' and B != 0).")

    # --- Criterion 3: loss-of-heat-sink / inlet-temperature balance ---
    if B != 0:
        ratio = C / B * dT_c
        criterion_3 = 1.0 < ratio < 2.0
        if not criterion_3:
            print("Warning Criterion 3: loss-of-heat-sink balance out of range "
                  f"(want 1 < C/B*dT_c < 2):\n  C/B*dT_c = {ratio:.3f}")
    else:
        criterion_3 = False
        print("Warning Criterion 3: cannot evaluate (B == 0).")

    params['ABC Criterion 1'] = criterion_1
    params['ABC Criterion 2'] = criterion_2
    params['ABC Criterion 3'] = criterion_3
    params['ABC Safe'] = bool(criterion_1 and criterion_2 and criterion_3)

    if params['ABC Safe']:
        print("ABC Analysis shows safe nominal transient characteristics")
    return params['ABC Safe']


def run_abc_analysis(build_openmc_model, params):
    """Evaluate the fuel-temperature, reflector, and coolant reactivity
    coefficients (pcm/K) for the 3D LTMR with automatic thermal expansion, then
    run the A/B/C quasi-static safety screen. Each k-eff is a single steady-state
    eigenvalue run (no depletion).

      Temperature (fuel) : fuel +dT  -> axial expansion of ALL components
                           (rules 1 & 2) + fuel density/Doppler
      Reflector          : refl +dT  -> radial expansion of the reflector
                           (rule 3) + reflector density
      Coolant            : cool +dT  -> coolant EOS density only (no geometry)

    Required : 'Temperature Perturbation' (K), 'Common Temperature' (K)
    Optional : 'Reference Temperature', 'Fuel/Reflector/Coolant Temperature',
               'Per-Region Temperatures', 'Primary Loop Inlet Temperature',
               'Coolant Boiling Temperature'
    """
    if params.get('Geometric Expansion') and params.get('Thermal Expansion', True):
        raise ValueError(
            "ABC/geometry mode requires 'Thermal Expansion': False so density is "
            "handled by mass_conserving_density_scale (not double-counted with the "
            "isotropic solid rescale). Set 'Thermal Expansion': False.")
    dT = params['Temperature Perturbation']
    T0 = params['Common Temperature']

    Tf0 = params.get('Fuel Temperature', T0)
    Tr0 = params.get('Reflector Temperature', T0)
    Tc0 = params.get('Coolant Temperature', T0)

    # ---- BASE STATE (operating temps, geometry expanded vs reference) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0, T0)
    kb2d, kb3d, sd_base = _run_model(build_openmc_model, params)
    params['keff 2D ABC base'] = kb2d
    params['keff ABC base uncertainty'] = sd_base
    params['keff 3D (2D corrected) ABC base'] = kb3d

    # ---- (A) FUEL TEMPERATURE COEFFICIENT: Doppler + axial expansion ----
    _set_region_temperatures(params, Tf0 + dT, Tr0, Tc0, T0)
    kf2d, kf3d, sd_kf= _run_model(build_openmc_model, params)
    ct, st = _coefficient(kb2d, kf2d, sd_base, sd_kf, dT)
    params['Temp Coeff 2D'] = params['Temp Coeff 3D (2D corrected)'] = ct
    params['Temp Coeff std'] = st

    # ---- (B) REFLECTOR COEFFICIENT: reflector density + radial expansion ----
    _set_region_temperatures(params, Tf0, Tr0 + dT, Tc0, T0)
    kr2d, kr3d, sd_kr = _run_model(build_openmc_model, params)
    ct, st = _coefficient(kb2d, kr2d, sd_base, sd_kr, dT)
    params['Reflector Coeff 2D'] = params['Reflector Coeff 3D (2D corrected)'] = ct
    params['Reflector Coeff std'] = st

    # ---- (C) COOLANT COEFFICIENT: coolant EOS density only (no geometry) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0 + dT, T0)
    kc2d, kc3d, sd_kc = _run_model(build_openmc_model, params)
    ct, st = _coefficient(kb2d, kc2d, sd_base, sd_kc, dT)
    params['Coolant Coeff 2D'] = params['Coolant Coeff 3D (2D corrected)'] = ct
    params['Coolant Coeff std'] = st

    # ---- A/B/C quasi-static inherent-safety screen ----
    _evaluate_abc_criteria(params)

    # ---- Restore reference geometry (Common Temperature restored by run_openmc's finally) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0, T0)