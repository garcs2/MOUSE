# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
shielding/
----------
MOUSE LTMR shielding analysis package.

Modules
-------
shielding_constants.py
    Shared ICRP-116 H*(10) dose coefficients (neutron + photon) and the
    make_energy_filter_and_coeffs() helper. Imported by both the template
    and shielding_calcs to avoid duplication and circular imports.

openmc_shielding_template_LTMR.py
    OpenMC model builder for the LTMR fixed-source shielding transport run.
    Mirrors the section structure of openmc_template_LTMR.py.
    Wraps the LTMR core universe in concentric shielding annuli and sets up
    ICRP-116 H*(10) dose tallies. Settings are fully isolated from the
    criticality template.

shielding_calcs.py
    Post-processing utilities: reads OpenMC statepoints, applies dose
    response functions, writes dose rates back to params, and serialises
    results to CSV and NumPy .npz files.
"""

from shielding.openmc_shielding_template_LTMR import build_openmc_shielding_model_LTMR
from shielding.shielding_calcs import (
    extract_dose_results,
    summarize_shielding_results,
    print_shielding_summary_table,
)

__all__ = [
    'build_openmc_shielding_model_LTMR',
    'extract_dose_results',
    'summarize_shielding_results',
    'print_shielding_summary_table',
]