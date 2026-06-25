# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
core_design_3D/core_thermal_geometry.py

Thermal *geometric* expansion of the LTMR core, kept separate from utils_3D.py so
the reactivity-coefficient orchestration (abc_analysis_3D.py) just calls in here.

Expansion rules (per design intent)
------------------------------------
  1. The fuel expands ONLY axially (its radial dimensions are held fixed).
  2. Every other component expands axially at the SAME rate as the fuel.
  3. Reflecting components are the ONLY components that expand radially.

  => axial growth is governed by the FUEL linear CTE and applied to every axial
     dimension; radial growth is governed by the REFLECTOR linear CTE and applied
     only to the radial reflector.

Repo conventions (core_design_3D, latest)
-----------------------------------------
  * The model is a genuine 3D build (openmc_template_LTMR_3D.build_openmc_model_LTMR_3D):
    axial fuel zones span +/- Active Height/2 and the axial reflector cells extend
    a further 'Axial Reflector Thickness'. So scaling 'Active Height' and
    'Axial Reflector Thickness' moves the REAL axial geometry (no buckling proxy).
  * Radial reflector is 'Radial Reflector Thickness'; the outer boundary is
    'Core Radius' = hex apothem + 'Radial Reflector Thickness'
    (calculate_core_radius_from_hex). The hex apothem (fuel/lattice radial extent)
    does NOT expand, so we hold it fixed and grow only the reflector annulus.

Reference temperature
---------------------
The dimensions present in params when expansion is first applied are treated as
the values at T_ref (default 293.15 K, matching the density model) and stashed
once, so repeated calls scale from the reference and never compound. For
coefficient work you may set params['Reference Temperature'] equal to the
operating temperature so the base state is unperturbed and only the +/- dT
perturbations move the geometry (the derivative is unaffected).
"""
import warnings

# Reuse the SAME linear-CTE table the density model uses: single source of truth.
from core_design_3D.openmc_materials_database_3D import (
    THERMAL_EXPANSION, T_REF_DEFAULT)


# params['Fuel'] / params['Reflector'] values that key differently in the
# materials database than the CTE table (TRIGA fuel is built as 'UZrH_alloy').
MATERIAL_KEY_ALIASES = {
    'TRIGA_fuel': 'UZrH_alloy',
}

# Reserved params keys used to stash the reference (cold) geometry.
_REF_KEYS = {
    'Active Height':             '_ref Active Height',
    'Axial Reflector Thickness': '_ref Axial Reflector Thickness',
    'Radial Reflector Thickness':'_ref Radial Reflector Thickness',
    'Core Radius':               '_ref Core Radius',
}


def _cte(material_key):
    """Mean linear CTE [1/K] for a material role value (params['Fuel'] etc.)."""
    key = MATERIAL_KEY_ALIASES.get(material_key, material_key)
    alpha = THERMAL_EXPANSION.get(key)
    if alpha is None:
        warnings.warn(
            f"[core-geometry] No CTE for '{material_key}' (resolved '{key}'); "
            f"using 0.0 -> no expansion for this driver."
        )
        return 0.0
    return alpha


def _stash_reference_dims(params):
    """Capture the as-built (reference) dimensions exactly once."""
    for dim_key, ref_key in _REF_KEYS.items():
        if ref_key not in params and dim_key in params:
            params[ref_key] = params[dim_key]


def _fixed_inner_radius(params):
    """Hex apothem = Core Radius - Radial Reflector Thickness, at the reference
    state. This is the fuel-lattice radial extent and is held fixed (rule 1)."""
    Rc0 = params.get('_ref Core Radius')
    t0 = params.get('_ref Radial Reflector Thickness')
    if Rc0 is not None and t0 is not None:
        return Rc0 - t0
    # Fallback if Core Radius wasn't provided
    return params.get('Lattice Apothem')


def linear_factor(alpha_L, T, T_ref):
    """Linear expansion factor L(T)/L(T_ref) = 1 + alpha_L * (T - T_ref)."""
    return 1.0 + alpha_L * (T - T_ref)


def expansion_factors(params, T_fuel, T_reflector, T_ref=None):
    """Return (axial_factor, radial_factor).

    axial  <- FUEL CTE      (rules 1 & 2)
    radial <- REFLECTOR CTE (rule 3)
    """
    if T_ref is None:
        T_ref = params.get('Reference Temperature', T_REF_DEFAULT)
    a_fuel = _cte(params['Fuel'])
    a_refl = _cte(params['Radial Reflector'])
    return (linear_factor(a_fuel, T_fuel, T_ref),
            linear_factor(a_refl, T_reflector, T_ref))


def apply_core_expansion(params, T_fuel, T_reflector, T_ref=None):
    """Adjust LTMR-3D geometric parameters in-place for thermal expansion.

    T_fuel       drives AXIAL expansion of every component (rules 1 & 2)
    T_reflector  drives RADIAL expansion of the reflector only (rule 3)
    T_ref        reference temperature of the stored dimensions
                 (default params['Reference Temperature'] or T_REF_DEFAULT)

    Records per-region volume ratios for the optional mass-conserving density
    scaling:
        params['Vol Ratio Axial']      fuel + general structure (axial-only)
        params['Vol Ratio Reflector']  reflector (radial annulus * axial height)
    Returns (axial_factor, radial_factor).
    """
    if T_ref is None:
        T_ref = params.get('Reference Temperature', T_REF_DEFAULT)

    _stash_reference_dims(params)
    axial, radial = expansion_factors(params, T_fuel, T_reflector, T_ref)

    # --- Axial dimensions (rules 1 & 2): scale from stored reference ---
    if '_ref Active Height' in params:
        params['Active Height'] = params['_ref Active Height'] * axial
    if '_ref Axial Reflector Thickness' in params:
        params['Axial Reflector Thickness'] = (
            params['_ref Axial Reflector Thickness'] * axial)

    # --- Radial reflector (rule 3) ---
    R_inner = _fixed_inner_radius(params)   # hex apothem, fixed
    if '_ref Radial Reflector Thickness' in params:
        params['Radial Reflector Thickness'] = (
            params['_ref Radial Reflector Thickness'] * radial)
        if R_inner is not None:
            # Core Radius = apothem + radial reflector thickness (matches
            # calculate_core_radius_from_hex with the apothem held fixed).
            params['Core Radius'] = R_inner + params['Radial Reflector Thickness']

    # --- Derived axial dimension (kept consistent if present) ---
    if 'Active Height' in params and 'Axial Reflector Thickness' in params and \
            'Drum Height' in params:
        params['Drum Height'] = (
            params['Active Height'] + 2.0 * params['Axial Reflector Thickness'])

    # --- Per-region volume ratios for optional mass conservation ---
    params['Vol Ratio Axial'] = axial          # axial-only solids: V/V0 = axial
    if R_inner is not None and '_ref Radial Reflector Thickness' in params:
        t0 = params['_ref Radial Reflector Thickness']
        annulus0 = (R_inner + t0) ** 2 - R_inner ** 2
        annulus = (R_inner + t0 * radial) ** 2 - R_inner ** 2
        area_ratio = (annulus / annulus0) if annulus0 != 0 else 1.0
        params['Vol Ratio Reflector'] = area_ratio * axial
    else:
        params['Vol Ratio Reflector'] = axial * radial ** 2  # crude fallback

    return axial, radial


def reset_core_geometry(params):
    """Restore geometric parameters to the stored reference (cold) values."""
    for dim_key, ref_key in _REF_KEYS.items():
        if ref_key in params:
            params[dim_key] = params[ref_key]
    if 'Active Height' in params and 'Axial Reflector Thickness' in params and \
            'Drum Height' in params:
        params['Drum Height'] = (
            params['Active Height'] + 2.0 * params['Axial Reflector Thickness'])
    params['Vol Ratio Axial'] = 1.0
    params['Vol Ratio Reflector'] = 1.0


def mass_conserving_density_scale(materials_database, params):
    """OPTIONAL (recommended for rigorous coefficients).

    Scale material densities so atom inventory is conserved under the geometric
    expansion applied by apply_core_expansion. Call this INSIDE
    build_openmc_model_LTMR_3D right after collect_materials_data, and build the
    materials at their cold reference density (params['Thermal Expansion']=False)
    so density and geometry are not double counted:

        materials_database = collect_materials_data(params)
        from core_design_3D.core_thermal_geometry import mass_conserving_density_scale
        mass_conserving_density_scale(materials_database, params)

    Scaling:
        fuel + general solids (axial-only) : rho /= params['Vol Ratio Axial']
        reflector material(s)              : rho /= params['Vol Ratio Reflector']
        coolant(s)                         : left to their own EOS density(T)

    CAVEAT: this scales each *material* by one factor. If a single material is
    used in regions with different volume changes (e.g. the same material as both
    radial annulus and axial cap, or graphite as reflector and matrix), the
    scaling is approximate; for exact per-region conservation set the density per
    cell in the geometry builder.
    """
    axial_vr = params.get('Vol Ratio Axial', 1.0)
    refl_vr = params.get('Vol Ratio Reflector', 1.0)

    reflector_roles = {params.get('Reflector'), params.get('Control Drum Reflector')}
    coolant_roles = {params.get('Coolant'), params.get('Secondary Coolant')}
    handled_units = ('g/cm3', 'g/cc', 'atom/b-cm', 'atom/cm3', 'kg/m3')

    for name, mat in materials_database.items():
        if name in coolant_roles:
            continue  # EOS density handled by the materials module
        units = (getattr(mat, 'density_units', '') or '').lower()
        rho = getattr(mat, 'density', None)
        if rho is None or units not in handled_units:
            continue
        vr = refl_vr if name in reflector_roles else axial_vr
        mat.set_density(mat.density_units, rho / vr)
    return materials_database

def expand_derived_geometry(params, T_ref=None):
    """Scale the drum-derived (cold) reflector/axial geometry for thermal
    expansion, IN PLACE. Call inside the build AFTER
    update_ltmr_reflector_geometry_from_drums (which sets the cold, drum-derived
    Core Radius / thicknesses) and BEFORE the core surfaces are created.

    Axial dims (Active Height, Axial Reflector Thickness) scale with the FUEL
    temperature (rules 1 & 2). The radial reflector grows the outer boundary
    (Core Radius) with the REFLECTOR temperature (rule 3), holding the hex apothem
    fixed. No-op unless params['Geometric Expansion'] is True. Also sets
    Vol Ratio Axial / Vol Ratio Reflector for mass_conserving_density_scale.
    """
    if not params.get('Geometric Expansion', False):
        params.setdefault('Vol Ratio Axial', 1.0)
        params.setdefault('Vol Ratio Reflector', 1.0)
        return 1.0, 1.0
    if T_ref is None:
        T_ref = params.get('Reference Temperature', T_REF_DEFAULT)
    T_common = params['Common Temperature']
    T_fuel = params.get('Fuel Temperature', T_common)
    T_refl = params.get('Reflector Temperature', T_common)

    a_fuel = _cte(params.get('Fuel'))
    a_refl = _cte(params.get('Radial Reflector', params.get('Reflector')))
    axial  = 1.0 + a_fuel * (T_fuel - T_ref)
    radial = 1.0 + a_refl * (T_refl - T_ref)

    H0, ax0 = params['Active Height'], params['Axial Reflector Thickness']
    rad0, Rc0 = params['Radial Reflector Thickness'], params['Core Radius']
    apothem = Rc0 - rad0                       # fixed hex apothem (no fuel radial expansion)

    params['Active Height']              = H0 * axial
    params['Axial Reflector Thickness']  = ax0 * axial
    params['Radial Reflector Thickness'] = rad0 * radial
    params['Core Radius']                = apothem + rad0 * radial
    if 'Drum Height' in params:
        params['Drum Height'] = params['Active Height'] + 2.0 * params['Axial Reflector Thickness']

    params['Vol Ratio Axial'] = axial
    a0 = (apothem + rad0) ** 2 - apothem ** 2
    a1 = (apothem + rad0 * radial) ** 2 - apothem ** 2
    params['Vol Ratio Reflector'] = (a1 / a0 if a0 else 1.0) * axial
    return axial, radial