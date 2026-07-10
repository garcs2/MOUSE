# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
shielding_constants.py

Shared constants and helpers for the MOUSE LTMR shielding analysis package.

Centralising the ICRP-116 H*(10) dose coefficients and the energy filter
builder here avoids both duplication and circular imports between
openmc_shielding_template_LTMR.py and shielding_calcs.py.

References
----------
ICRP Publication 116, Annex A (neutron) and Annex B (photon).
"""

import numpy as np
import openmc

# **************************************************************************************************************************
#                              ICRP-116 H*(10) Ambient Dose Equivalent Coefficients
# **************************************************************************************************************************

# Neutron H*(10) — ICRP-116 Table A.1
# Energies in MeV, coefficients in pSv·cm²
NEUTRON_DOSE_ENERGY_MEV = np.array([
    1.00e-9, 1.00e-8, 2.53e-8, 1.00e-7, 2.00e-7, 5.00e-7,
    1.00e-6, 2.00e-6, 5.00e-6, 1.00e-5, 2.00e-5, 5.00e-5,
    1.00e-4, 2.00e-4, 5.00e-4, 1.00e-3, 2.00e-3, 5.00e-3,
    1.00e-2, 2.00e-2, 5.00e-2, 1.00e-1, 2.00e-1, 5.00e-1,
    1.00,    2.00,    5.00,    1.00e1,  2.00e1,
])
NEUTRON_DOSE_COEFF_PSV_CM2 = np.array([
    3.09e-4, 3.10e-4, 3.26e-4, 3.64e-4, 4.07e-4, 4.97e-4,
    5.98e-4, 6.88e-4, 7.63e-4, 7.79e-4, 7.54e-4, 6.54e-4,
    5.61e-4, 4.93e-4, 4.51e-4, 4.51e-4, 4.74e-4, 5.66e-4,
    7.13e-4, 9.56e-4, 1.48e-3, 2.74e-3, 4.24e-3, 6.39e-3,
    7.27e-3, 8.22e-3, 1.06e-2, 1.48e-2, 2.12e-2,
])

# Photon H*(10) — ICRP-116 Table B.1
# Energies in MeV, coefficients in pSv·cm²
PHOTON_DOSE_ENERGY_MEV = np.array([
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10,
    0.15, 0.20,  0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.50, 2.00, 3.00,
])
PHOTON_DOSE_COEFF_PSV_CM2 = np.array([
    7.43e-4, 3.12e-4, 1.68e-4, 7.89e-5, 5.09e-5, 4.13e-5, 3.93e-5, 3.96e-5,
    4.16e-5, 4.83e-5, 7.13e-5, 9.67e-5, 1.46e-4, 1.93e-4, 2.39e-4, 2.84e-4,
    3.73e-4, 4.61e-4, 6.72e-4, 8.85e-4, 1.31e-3,
])


def make_energy_filter_and_coeffs(particle: str):
    """
    Build an OpenMC EnergyFilter and return the matching ICRP-116 H*(10)
    dose coefficient array for the given particle type.

    Energy bin edges are derived from the ICRP-116 coefficient tables above
    and converted to eV (OpenMC's internal unit).  A closing upper edge is
    appended at 10× the highest tabulated energy.

    @ In,  particle,    str,                  'neutron' or 'photon'.
    @ Out, energy_filt, openmc.EnergyFilter,  Filter aligned with dose coefficient bins.
    @ Out, coeffs,      np.ndarray,           H*(10) coefficients in pSv·cm².
    """
    if particle == 'neutron':
        energies_ev = NEUTRON_DOSE_ENERGY_MEV * 1e6
        coeffs      = NEUTRON_DOSE_COEFF_PSV_CM2
    elif particle == 'photon':
        energies_ev = PHOTON_DOSE_ENERGY_MEV * 1e6
        coeffs      = PHOTON_DOSE_COEFF_PSV_CM2
    else:
        raise ValueError(f"Unsupported particle type '{particle}'. Must be 'neutron' or 'photon'.")

    edges       = np.append(energies_ev, energies_ev[-1] * 10)
    energy_filt = openmc.EnergyFilter(edges)
    return energy_filt, coeffs
