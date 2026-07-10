# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
shielding_calcs.py

Post-processing utilities for the MOUSE LTMR shielding study.

Functions
---------
extract_dose_results(params)
    Reads the OpenMC statepoint file from the most recent shielding run,
    applies ICRP-116 H*(10) dose response coefficients to neutron and photon
    flux tallies, and writes dose rate values (mSv/hr) back into params.

summarize_shielding_results(params, tracked_params, output_csv)
    Appends the current iteration's tracked parameters to a running CSV file,
    creating it with a header row if it does not yet exist.

Dose rate conversion chain
--------------------------
    flux [particles/cm²/s] × dose_coeff [pSv·cm²] → dose_rate [pSv/s]
    × 3600 [s/hr] × 1e-9 [pSv→mSv] → dose_rate [mSv/hr]

    Flux from OpenMC tallies is per source particle; multiply by the
    fission source rate [neutrons/s] derived from reactor power:

        source_rate [n/s] = Power [W] / (E_fission [J] × k_eff × nu)

    where E_fission ≈ 200 MeV = 3.204e-11 J, nu ≈ 2.43 n/fission.
"""

import os
import glob
import csv
import numpy as np
import openmc
from shielding.shielding_constants import (
    NEUTRON_DOSE_COEFF_PSV_CM2,
    PHOTON_DOSE_COEFF_PSV_CM2,
    make_energy_filter_and_coeffs,
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
E_FISSION_J  = 200e6 * 1.60218e-19   # 200 MeV in Joules
NU_BAR       = 2.43                   # average neutrons per fission
PSV_S_TO_MSV_HR = 3600e-9             # pSv/s → mSv/hr conversion


def _find_statepoint(run_dir: str = '.') -> str:
    """Return the path to the most recent statepoint file in run_dir."""
    pattern   = os.path.join(run_dir, 'statepoint.*.h5')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No statepoint file found matching '{pattern}'. "
            "Ensure the OpenMC shielding run completed successfully."
        )
    return candidates[-1]  # highest batch number


def _source_rate_n_per_s(params) -> float:
    """
    Compute the fission neutron source rate [n/s] from reactor thermal power.

        S = P_thermal / (E_fission * k_eff * nu^-1)
          = P_thermal * nu / (E_fission * k_eff)

    k_eff is read from params if available (written by Step-1 criticality run);
    defaults to 1.0 (at-power critical condition).
    """
    power_w  = params['Power MWt'] * 1e6   # W
    k_eff    = params.get('k_eff', 1.0)
    source_rate = (power_w * NU_BAR) / (E_FISSION_J * k_eff)
    return source_rate


def _apply_dose_coefficients(flux_per_source: np.ndarray,
                              flux_std_dev: np.ndarray,
                              dose_coeffs: np.ndarray,
                              source_rate: float):
    """
    Convert a multi-group flux array [cm⁻² per source particle] to a
    total dose rate [mSv/hr] with Gaussian-propagated 1-σ uncertainty.

    Gaussian error propagation through the dot product (energy bins assumed
    independent):

        dose   = Σ_g (flux_g × coeff_g) × source_rate × conversion
        σ_dose = √(Σ_g (coeff_g × σ_flux_g)²) × source_rate × conversion

    Parameters
    ----------
    flux_per_source : ndarray, shape (n_groups,)
        Mean flux per source particle from OpenMC tally.
    flux_std_dev : ndarray, shape (n_groups,)
        1-σ standard deviation of flux per source particle from OpenMC tally.
    dose_coeffs : ndarray, shape (n_groups,)
        ICRP-116 H*(10) coefficients [pSv·cm²].
    source_rate : float
        Fission source rate [neutrons/s].

    Returns
    -------
    dose_rate_mSv_hr : float
    dose_unc_mSv_hr  : float   1-σ uncertainty via Gaussian error propagation.
    """
    scale      = source_rate * PSV_S_TO_MSV_HR
    dose_rate  = np.dot(flux_per_source, dose_coeffs) * scale
    dose_unc   = np.sqrt(np.sum((dose_coeffs * flux_std_dev) ** 2)) * scale
    return dose_rate, dose_unc


def _extrapolate_isq(dose_iso: float, unc_iso: float,
                     r_iso: float, r_target: float):
    """
    Extrapolate dose rate and its uncertainty from r_iso to r_target using
    the inverse-square law.  Since r_iso and r_target are exact geometry
    values, the relative uncertainty is preserved identically:

        D(r)   = D_iso × (r_iso / r)²
        σ_D(r) = σ_D_iso × (r_iso / r)²

    Parameters
    ----------
    dose_iso : float   Dose rate at iso_surface [mSv/hr].
    unc_iso  : float   1-σ uncertainty at iso_surface [mSv/hr].
    r_iso    : float   Radial distance of iso_surface tally shell [cm].
    r_target : float   Radial distance of the target evaluation point [cm].

    Returns
    -------
    dose_target : float
    unc_target  : float
    """
    factor      = (r_iso / r_target) ** 2
    return dose_iso * factor, unc_iso * factor


def _get_tally_by_name(sp: openmc.StatePoint, name: str) -> openmc.Tally:
    """Retrieve a tally from a statepoint by its name."""
    for tally in sp.tallies.values():
        if tally.name == name:
            return tally
    raise KeyError(f"Tally '{name}' not found in statepoint. "
                   "Check that the shielding model built tallies with this name.")


def extract_dose_results(params: dict) -> None:
    """
    Read the shielding statepoint and populate dose rate entries in params.

    Direct tally results (iso_surface) include Gaussian-propagated 1-σ
    uncertainties from OpenMC flux tallies.  Dose rates at 1m_standoff and
    30m_exclusion are derived analytically from the iso_surface result using
    the inverse-square law; their uncertainties are propagated accordingly.

    Keys written to params for each evaluation point:
      'Dose Rate {label} mSv_hr'         — combined neutron + photon dose rate
      'Dose Rate {label} unc mSv_hr'     — 1-σ absolute uncertainty
      'Dose Rate {label} neutron mSv_hr' — neutron-only component
      'Dose Rate {label} photon mSv_hr'  — photon-only component

    Parameters
    ----------
    params : watts.Parameters (or dict)
        Must contain 'Dose Evaluation Radii cm', 'Power MWt', optionally 'k_eff'.
    """
    run_dir     = params.get('openmc_run_dir', '.')
    sp_path     = _find_statepoint(run_dir)
    source_rate = _source_rate_n_per_s(params)

    print(f"  Reading statepoint: {sp_path}")
    print(f"  Source rate: {source_rate:.4e} n/s")

    # ---- Step A: Extract direct tally results at iso_surface ----
    # Only the iso_surface tally is inside the transport geometry; 1m and 30m
    # are derived via inverse-square law after the statepoint read.
    iso_label  = 'iso_surface'
    iso_radius = params['Dose Evaluation Radii cm'].get(iso_label)

    iso_total_dose = 0.0
    iso_total_unc  = 0.0   # combined in quadrature over neutron + photon

    with openmc.StatePoint(sp_path) as sp:

        for label, radius_cm in params['Dose Evaluation Radii cm'].items():
            if radius_cm is None:
                continue

            # 1m_standoff and 30m_exclusion are handled analytically below
            if label in ('1m_standoff', '30m_exclusion'):
                continue

            dose_components = {}
            unc_components  = {}

            for particle in ['neutron', 'photon']:
                tally_name = f'dose_point_{label}_{particle}'

                try:
                    tally = _get_tally_by_name(sp, tally_name)
                except KeyError as e:
                    print(f"  WARNING: {e} — skipping {label}/{particle}")
                    dose_components[particle] = float('nan')
                    unc_components[particle]  = float('nan')
                    continue

                flux_mean   = tally.get_values(scores=['flux']).flatten()
                flux_stddev = tally.get_values(
                    scores=['flux'], value='std_dev').flatten()

                # ---- Normalise by shell volume to get flux density ----
                # OpenMC track-length tallies score flux integrated over the
                # mesh volume [cm per source particle].  Dividing by the shell
                # volume [cm³] converts to average flux density [cm⁻² per
                # source particle], which is what the dose coefficients expect.
                axial_half = (
                    params['Active Height'] / 2.0
                    + params['Axial Reflector Thickness']
                    + 50.0
                )
                r_inner    = max(0.0, radius_cm - 0.5)
                r_outer    = radius_cm + 0.5
                shell_vol  = np.pi * (r_outer**2 - r_inner**2) * (2.0 * axial_half)
                flux_mean   = flux_mean   / shell_vol
                flux_stddev = flux_stddev / shell_vol

                _, coeffs = make_energy_filter_and_coeffs(particle)

                # Guard against energy group mismatch
                if len(flux_mean) != len(coeffs):
                    print(f"  WARNING: Energy group mismatch for {tally_name}: "
                          f"flux has {len(flux_mean)} groups, coeffs have {len(coeffs)}. "
                          "Truncating to minimum length.")
                    n           = min(len(flux_mean), len(coeffs))
                    flux_mean   = flux_mean[:n]
                    flux_stddev = flux_stddev[:n]
                    coeffs      = coeffs[:n]

                dose_rate, dose_unc = _apply_dose_coefficients(
                    flux_mean, flux_stddev, coeffs, source_rate)
                dose_components[particle] = dose_rate
                unc_components[particle]  = dose_unc

            # ---- Combine neutron + photon (sum doses, quadrature for unc) ----
            neutron_dose = dose_components.get('neutron', 0.0)
            photon_dose  = dose_components.get('photon',  0.0)
            neutron_unc  = unc_components.get('neutron',  0.0)
            photon_unc   = unc_components.get('photon',   0.0)

            total_dose = (
                (neutron_dose if not np.isnan(neutron_dose) else 0.0)
                + (photon_dose  if not np.isnan(photon_dose)  else 0.0)
            )
            total_unc = np.sqrt(
                (neutron_unc if not np.isnan(neutron_unc) else 0.0) ** 2
                + (photon_unc  if not np.isnan(photon_unc)  else 0.0) ** 2
            )

            params[f'Dose Rate {label} mSv_hr']         = total_dose
            params[f'Dose Rate {label} unc mSv_hr']     = total_unc
            params[f'Dose Rate {label} neutron mSv_hr'] = neutron_dose
            params[f'Dose Rate {label} photon mSv_hr']  = photon_dose

            print(f"  [{label:20s}] r = {radius_cm:7.1f} cm | "
                  f"Total: {total_dose:.3e} ± {total_unc:.3e} mSv/hr  "
                  f"(n: {neutron_dose:.3e}, γ: {photon_dose:.3e})")

            # Cache iso_surface values for ISL extrapolation
            if label == iso_label:
                iso_total_dose = total_dose
                iso_total_unc  = total_unc

    # ---- Step B: Inverse-square law extrapolation to 1m and 30m ----
    if iso_radius is not None and iso_total_dose > 0.0:
        for label in ('1m_standoff', '30m_exclusion'):
            r_target = params['Dose Evaluation Radii cm'].get(label)
            if r_target is None:
                continue

            dose_r, unc_r = _extrapolate_isq(
                iso_total_dose, iso_total_unc, iso_radius, r_target)

            params[f'Dose Rate {label} mSv_hr']     = dose_r
            params[f'Dose Rate {label} unc mSv_hr'] = unc_r

            print(f"  [{label:20s}] r = {r_target:7.1f} cm | "
                  f"Total: {dose_r:.3e} ± {unc_r:.3e} mSv/hr  "
                  f"(ISL extrapolated from iso_surface)")

    # ---- Also extract full mesh dose map for plotting ----
    _extract_mesh_dose_map(sp_path, params, source_rate)


def _extract_mesh_dose_map(sp_path: str, params: dict, source_rate: float) -> None:
    """
    Extract the 2D (r, z) dose rate map from the cylindrical mesh tally and
    save it as a NumPy .npz file for downstream plotting.
    """
    output_dir  = params.get('shielding_output_dir', '.')
    mobile_tag  = "mobile" if params['Mobile'] else "stationary"
    mat_tag     = params['Out Of Vessel Shield Material'].replace(' ', '_')
    thick_tag   = f"{params['Out Of Vessel Shield Thickness']:.0f}cm"
    npz_name    = f"dose_map_{mobile_tag}_{mat_tag}_{thick_tag}.npz"
    npz_path    = os.path.join(output_dir, npz_name)

    with openmc.StatePoint(sp_path) as sp:
        combined_dose = None

        for particle in ['neutron', 'photon']:
            tally_name = f'{particle}_flux_mesh'
            try:
                tally     = _get_tally_by_name(sp, tally_name)
            except KeyError:
                continue

            # Shape: (r_bins, z_bins, energy_groups)  after reshaping
            flux_data = tally.get_values(scores=['flux']).squeeze()
            _, coeffs = make_energy_filter_and_coeffs(particle)

            # Align energy axis
            n_groups = min(flux_data.shape[-1], len(coeffs))
            flux_data = flux_data[..., :n_groups]
            coeffs    = coeffs[:n_groups]

            # Dose rate map [mSv/hr], shape: (r_bins, z_bins)
            dose_map = np.einsum('...g,g->...', flux_data, coeffs) * source_rate * PSV_S_TO_MSV_HR

            if combined_dose is None:
                combined_dose = dose_map
            else:
                combined_dose += dose_map

    if combined_dose is not None:
        mesh = params.get('_shielding_mesh')
        np.savez(
            npz_path,
            dose_map_mSv_hr=combined_dose,
            r_grid=mesh.r_grid if mesh else None,
            z_grid=mesh.z_grid if mesh else None,
            mobile=params['Mobile'],
            shield_material=params['Out Of Vessel Shield Material'],
            shield_thickness_cm=params['Out Of Vessel Shield Thickness'],
        )
        print(f"  Dose map saved: {npz_path}")


def summarize_shielding_results(params: dict,
                                tracked_params: list,
                                output_csv: str) -> None:
    """
    Append the current shielding iteration's results to a CSV file.

    Parameters
    ----------
    params : dict
        Contains all tracked parameter values.
    tracked_params : list of str
        Parameter keys to write as columns.
    output_csv : str
        Path to the output CSV file.
    """
    write_header = not os.path.isfile(output_csv)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=tracked_params, extrasaction='ignore')
        if write_header:
            writer.writeheader()

        row = {}
        for key in tracked_params:
            val = params.get(key, 'N/A')
            # Format floats for readability
            if isinstance(val, float):
                row[key] = f"{val:.6e}" if abs(val) < 1e-3 or abs(val) > 1e4 else f"{val:.4f}"
            else:
                row[key] = val

        writer.writerow(row)

    print(f"  Results appended to: {output_csv}")


def print_shielding_summary_table(output_csv: str) -> None:
    """
    Print a formatted summary table of all shielding study results from the CSV.
    Useful at the end of a parametric sweep run.
    """
    if not os.path.isfile(output_csv):
        print(f"No results file found at {output_csv}")
        return

    with open(output_csv, 'r') as f:
        reader  = csv.DictReader(f)
        rows    = list(reader)
        headers = reader.fieldnames

    if not rows:
        print("No results recorded.")
        return

    col_widths = {h: max(len(h), max(len(str(r.get(h, ''))) for r in rows)) for h in headers}
    sep        = ' | '
    header_line = sep.join(h.ljust(col_widths[h]) for h in headers)
    divider     = '-' * len(header_line)

    print("\n" + "="*len(header_line))
    print("SHIELDING STUDY SUMMARY")
    print("="*len(header_line))
    print(header_line)
    print(divider)
    for row in rows:
        print(sep.join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers))
    print("="*len(header_line) + "\n")
