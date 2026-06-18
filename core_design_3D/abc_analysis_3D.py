# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
core_design_3D/abc_analysis_3D.py

Temperature / reflector / coolant reactivity-coefficient analysis for the 3D
LTMR, with automatic thermal geometric expansion (core_thermal_geometry.py).

Kept in its own module so it can be wired into the (large) utils_3D.py with a
small edit rather than a hand-merge. In utils_3D.run_openmc, add:

    params.setdefault('ABC Analysis', False)
    ...
    # inside the try block, before the Shutdown-Margin / ITC branching:
    if params['ABC Analysis']:
        if 'Temperature Perturbation' not in params:
            raise ValueError("ABC Analysis requires 'Temperature Perturbation' (K).")
        from core_design_3D.abc_analysis_3D import run_abc_analysis
        run_abc_analysis(build_openmc_model, params)
        return   # the existing 'finally' restores Common Temperature

This module calls run_depletion_analysis (defined in utils_3D.py) via a deferred
import to avoid a circular import.
"""
import copy

import numpy as np

from core_design_3D.core_thermal_geometry import (
    apply_core_expansion,
    reset_core_geometry,
)


def _run_model(build_openmc_model, params):
    """Run one OpenMC depletion case honoring the CURRENT params (geometry+temps).
    Returns (keff_2d_list, keff_3d_corrected_list)."""
    import watts
    from core_design_3D.utils_3D import run_depletion_analysis  # deferred (no cycle)
    openmc_plugin = watts.PluginOpenMC(build_openmc_model, show_stderr=True)
    openmc_plugin(params, function=lambda: run_depletion_analysis(params))
    return params['keff 2D'], params['keff 3D (2D corrected)']


def _coefficient(keff_base, keff_pert, dT):
    """Most-limiting reactivity coefficient (pcm/K) over the burnup steps.
    Mirrors the existing ITC formula: (k_p - k_b)/(k_p*k_b)/dT * 1e5."""
    return np.max([
        (kp - kb) / (kp * kb) / dT * 1e5
        for kb, kp in zip(keff_base, keff_pert)
    ])


def _set_region_temperatures(params, T_fuel, T_reflector, T_coolant):
    """Propagate per-region temperatures to params.

    NOTE: openmc_materials_database_3D currently keys both density and cross-
    section temperature off the single 'Common Temperature'. Until it honors
    these per-region keys, the density/Doppler part of the reflector and coolant
    coefficients is NOT isolated (Common Temperature is set to the perturbed
    region below, heating every material). The GEOMETRIC part of every
    coefficient IS isolated, because apply_core_expansion reads T_fuel/T_reflector
    directly. A small materials-module change (read 'Fuel/Reflector/Coolant
    Temperature', fall back to 'Common Temperature') fully decomposes the
    density/Doppler term.
    """
    params['Fuel Temperature'] = T_fuel
    params['Reflector Temperature'] = T_reflector
    params['Coolant Temperature'] = T_coolant
    params['Common Temperature'] = max(T_fuel, T_reflector, T_coolant)


def run_abc_analysis(build_openmc_model, params):
    """Evaluate the fuel-temperature, reflector, and coolant reactivity
    coefficients (pcm/K) for the 3D LTMR, adjusting core geometry for thermal
    expansion automatically.

    Each coefficient perturbs one region's temperature by +dT and finite-
    differences against a single shared base run:

      Temperature (fuel) : fuel +dT  -> axial expansion of ALL components
                           (rules 1 & 2) + fuel density/Doppler
      Reflector          : refl +dT  -> radial expansion of the reflector
                           (rule 3) + reflector density
      Coolant            : cool +dT  -> coolant EOS density only (no geometry)

    Required params : 'Temperature Perturbation' (K), 'Common Temperature' (K)
    Optional params : 'Reference Temperature', 'Fuel/Reflector/Coolant Temperature'

    Stores params['Temp/Reflector/Coolant Coeff 2D'] and
    params['... 3D (2D corrected)'].
    """
    dT = params['Temperature Perturbation']
    T_ref = params.get('Reference Temperature', None)
    T0 = params['Common Temperature']

    Tf0 = params.get('Fuel Temperature', T0)
    Tr0 = params.get('Reflector Temperature', T0)
    Tc0 = params.get('Coolant Temperature', T0)

    # ---- BASE STATE (operating temps, geometry expanded vs reference) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0)
    apply_core_expansion(params, T_fuel=Tf0, T_reflector=Tr0, T_ref=T_ref)
    kb2d, kb3d = _run_model(build_openmc_model, params)
    params['keff 2D ABC base'] = kb2d
    params['keff 3D (2D corrected) ABC base'] = kb3d

    # ---- (A) FUEL TEMPERATURE COEFFICIENT: Doppler + axial expansion ----
    _set_region_temperatures(params, Tf0 + dT, Tr0, Tc0)
    apply_core_expansion(params, T_fuel=Tf0 + dT, T_reflector=Tr0, T_ref=T_ref)
    kf2d, kf3d = _run_model(build_openmc_model, params)
    params['Temp Coeff 2D'] = _coefficient(kb2d, kf2d, dT)
    params['Temp Coeff 3D (2D corrected)'] = _coefficient(kb3d, kf3d, dT)

    # ---- (B) REFLECTOR COEFFICIENT: reflector density + radial expansion ----
    _set_region_temperatures(params, Tf0, Tr0 + dT, Tc0)
    apply_core_expansion(params, T_fuel=Tf0, T_reflector=Tr0 + dT, T_ref=T_ref)
    kr2d, kr3d = _run_model(build_openmc_model, params)
    params['Reflector Coeff 2D'] = _coefficient(kb2d, kr2d, dT)
    params['Reflector Coeff 3D (2D corrected)'] = _coefficient(kb3d, kr3d, dT)

    # ---- (C) COOLANT COEFFICIENT: coolant EOS density only (no geometry) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0 + dT)
    apply_core_expansion(params, T_fuel=Tf0, T_reflector=Tr0, T_ref=T_ref)  # geom unchanged
    kc2d, kc3d = _run_model(build_openmc_model, params)
    params['Coolant Coeff 2D'] = _coefficient(kb2d, kc2d, dT)
    params['Coolant Coeff 3D (2D corrected)'] = _coefficient(kb3d, kc3d, dT)

    # ---- Restore reference geometry (Common Temperature restored by run_openmc's finally) ----
    _set_region_temperatures(params, Tf0, Tr0, Tc0)
    reset_core_geometry(params)