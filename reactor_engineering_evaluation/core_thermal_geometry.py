# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""
core_thermal_geometry.py

Thermal *geometric* expansion of the LTMR core, kept deliberately separate from
utils_3D.py so the reactivity-coefficient orchestration just calls into here.

Expansion rules (per design intent)
------------------------------------
  1. The fuel expands ONLY axially (its radial dimensions are held fixed).
  2. Every other component expands axially at the SAME rate as the fuel.
  3. Reflecting components are the ONLY components that expand radially.

Consequences:
  * Axial expansion is governed by the FUEL linear CTE and applied to every
    axial dimension (Active Height, Axial Reflector Thickness, Drum Height).
  * Radial expansion is governed by the REFLECTOR linear CTE and applied only to
    the radial reflector (Reflector Thickness -> Core Radius). The lattice radius
    (fuel/structure radial extent) does not change.

How the dimensions reach the physics (LTMR 2D->3D corrected scheme)
-------------------------------------------------------------------
  * Active Height + 2*Axial Reflector Thickness feeds the axial-leakage
    correction in corrected_keff_2d -> axial expansion shows up in
    keff 3D (2D corrected).
  * Reflector Thickness / Core Radius change the actual 2D OpenMC geometry ->
    radial expansion shows up in keff 2D (and the corrected value).

Reference temperature
---------------------
The dimensions present in `params` when expansion is first applied are treated
as the values at T_ref (default 293.15 K, matching the density model). They are
stashed once so repeated calls scale from the reference and never compound.
For coefficient work you may instead set params['Reference Temperature'] equal
to the operating temperature, so the base state is unperturbed and only the
+/- dT perturbations move the geometry (the derivative is unaffected).
"""
import warnings

# Reuse the SAME linear-CTE table the density model uses: single source of truth.
# Adjust the import path to wherever your materials module lives.
from core_design_3D.openmc_materials_database_3D import THERMAL_EXPANSION, T_REF_DEFAULT


# params['Fuel'] / params['Reflector'] values that key differently in the
# materials database than the CTE table (e.g. TRIGA fuel is built as 'UZrH_alloy').
MATERIAL_KEY_ALIASES = {
    'TRIGA_fuel': 'UZrH_alloy',
}

# Reserved params keys used to stash the reference (cold) geometry.
_REF_KEYS = {
    'Active Height':             '_ref Active Height',
    'Axial Reflector Thickness': '_ref Axial Reflector Thickness',
    'Reflector Thickness':       '_ref Reflector Thickness',
    'Lattice Radius':            '_ref Lattice Radius',
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


def linear_factor(alpha_L, T, T_ref):
    """Linear expansion factor L(T)/L(T_ref) = 1 + alpha_L * (T - T_ref)."""
    return 1.0 + alpha_L * (T - T_ref)


def expansion_factors(params, T_fuel, T_reflector, T_ref=None):
    """Return (axial_factor, radial_factor).

    axial  <- FUEL CTE      (rules 1 & 2: fuel drives the axial growth of all)
    radial <- REFLECTOR CTE (rule 3: only the reflector grows radially)
    """
    if T_ref is None:
        T_ref = params.get('Reference Temperature', T_REF_DEFAULT)
    a_fuel = _cte(params['Fuel'])
    a_refl = _cte(params['Reflector'])
    return (linear_factor(a_fuel, T_fuel, T_ref),
            linear_factor(a_refl, T_reflector, T_ref))


def apply_core_expansion(params, T_fuel, T_reflector, T_ref=None):
    """Adjust LTMR geometric parameters in-place for thermal expansion.

    Parameters
    ----------
    params : watts.Parameters / dict
    T_fuel : float
        Temperature (K) driving AXIAL expansion of every component (rules 1 & 2).
    T_reflector : float
        Temperature (K) driving RADIAL expansion of the reflector only (rule 3).
    T_ref : float, optional
        Reference temperature of the stored dimensions. Defaults to
        params['Reference Temperature'] or the density model's T_REF_DEFAULT.

    Side effects
    ------------
    Sets the expanded dimensions and records per-region volume ratios used by the
    optional mass-conserving density scaling:
        params['Vol Ratio Axial']      fuel + general structure (axial-only)
        params['Vol Ratio Reflector']  reflector (radial annulus * axial height)
    Returns (axial_factor, radial_factor).
    """
    if T_ref is None:
        T_ref = params.get('Reference Temperature', T_REF_DEFAULT)

    _stash_reference_dims(params)
    axial, radial = expansion_factors(params, T_fuel, T_reflector, T_ref)

    # --- Axial dimensions (rules 1 & 2): scale from stored reference ---
    if 'Active Height' in params:
        params['Active Height'] = params['_ref Active Height'] * axial
    if 'Axial Reflector Thickness' in params:
        params['Axial Reflector Thickness'] = (
            params['_ref Axial Reflector Thickness'] * axial)

    # --- Radial reflector (rule 3) ---
    if 'Reflector Thickness' in params:
        params['Reflector Thickness'] = params['_ref Reflector Thickness'] * radial

    # --- Derived dimensions ---
    R_lat = params.get('_ref Lattice Radius', params.get('Lattice Radius'))
    if R_lat is not None:
        params['Lattice Radius'] = R_lat  # fuel/structure radial extent: unchanged
        if 'Reflector Thickness' in params:
            params['Core Radius'] = R_lat + params['Reflector Thickness']
    if 'Active Height' in params and 'Axial Reflector Thickness' in params:
        params['Drum Height'] = (
            params['Active Height'] + 2.0 * params['Axial Reflector Thickness'])

    # --- Per-region volume ratios for optional mass conservation ---
    # Axial-only solids (fuel + general structure): V/V0 = axial
    params['Vol Ratio Axial'] = axial
    # Reflector: exact radial annulus-area ratio (inner = fixed lattice radius)
    # times the axial height ratio.
    if R_lat is not None and '_ref Reflector Thickness' in params:
        t0 = params['_ref Reflector Thickness']
        annulus0 = (R_lat + t0) ** 2 - R_lat ** 2
        annulus = (R_lat + t0 * radial) ** 2 - R_lat ** 2
        area_ratio = (annulus / annulus0) if annulus0 != 0 else 1.0
        params['Vol Ratio Reflector'] = area_ratio * axial
    else:
        params['Vol Ratio Reflector'] = axial * radial ** 2  # crude fallback

    return axial, radial


def reset_core_geometry(params):
    """Restore geometric parameters to the stored reference values."""
    for dim_key, ref_key in _REF_KEYS.items():
        if ref_key in params:
            params[dim_key] = params[ref_key]
    if 'Lattice Radius' in params and 'Reflector Thickness' in params:
        params['Core Radius'] = params['Lattice Radius'] + params['Reflector Thickness']
    if 'Active Height' in params and 'Axial Reflector Thickness' in params:
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
    so density and geometry are not double counted.

        fuel + general solids (axial-only) : rho /= params['Vol Ratio Axial']
        reflector material(s)              : rho /= params['Vol Ratio Reflector']
        coolant(s)                         : left to their own EOS density(T)

    CAVEAT: this scales each *material* by one factor. If a single material is
    used in regions with different volume changes (e.g. the same material as both
    radial and axial reflector, or graphite used as both reflector and matrix),
    the scaling is approximate. For exact per-region conservation, set the
    density per cell where the material is placed in the geometry builder.
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