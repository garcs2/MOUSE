# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np
import openmc
import openmc.deplete
import watts
import traceback # tracing errors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from core_design.correction_factor import corrected_keff_2d
from core_design.peaking_factor import compute_pin_peaking_factors
import glob
import os
import pandas,copy

import shutil  # add to the top of shielding_analysis.py if not already imported

def run_bol_source_run(params):
    """
    One-time, standalone BOL (fresh-fuel) k-eigenvalue solve used only to
    generate a fission source for reuse across the shielding parametric sweep.

    IMPORTANT: WATTS deletes its tmp run directory as soon as this callback
    returns, so the persistent copy of the source file MUST happen here,
    while cwd is still the WATTS-managed run directory — copying it back
    in the exec script after the plugin call is too late.
    """
    settings = openmc.Settings.from_xml()
    settings.sourcepoint = {'batches': [settings.batches], 'separate': True, 'write': True}
    settings.export_to_xml()

    openmc.run()

    source_candidates = sorted(glob.glob('source.*.h5'))
    if not source_candidates:
        raise FileNotFoundError(
            "BOL source run completed but no source.*.h5 was written — "
            "check that settings.sourcepoint write=True took effect."
        )
    source_file = source_candidates[-1]

    persistent_dir = params['shielding_output_dir']
    os.makedirs(persistent_dir, exist_ok=True)
    persistent_source_path = os.path.join(persistent_dir, 'bol_fission_source.h5')
    shutil.copy2(source_file, persistent_source_path)

    params['Fission Source File'] = persistent_source_path
    print(f"  [Shielding] BOL fission source saved for reuse at: {persistent_source_path}")

def run_shielding_analysis(params):
    """
    Callback passed to the WATTS plugin for the shielding transport run.

    Step 1 — Fission source acquisition:
        If params['Fission Source File'] points to an existing persistent
        source file (written once, up front, by run_bol_source_run via the
        exec script), that file is reused directly and no eigenvalue solve
        happens here at all. Otherwise, falls back to a local eigenvalue
        solve via openmc.run() on the geometry built by
        build_openmc_shielding_model_LTMR, writing statepoint.*.h5.

    Step 2 — Fixed-source shielding run:
        settings.xml is modified in-place to fixed-source mode using the
        source file from Step 1, and openmc.run() executes the neutron +
        photon transport. extract_dose_results() then reads the resulting
        statepoint.

    @ In, params, watts.Parameters, Simulation parameters.
    """
    from shielding.shielding_calcs import extract_dose_results

    # ---- Step 1: Reuse persistent BOL source if available, else run locally ----
    params.setdefault('Save Dose Map', True)
    source_file = params.get('Fission Source File')
    if source_file and os.path.isfile(source_file):
        print(f"\n  [Shielding] Reusing persistent fission source: {source_file}")
        local_source_file = source_file
    else:
        print("\n  [Shielding] No persistent source found — running local eigenvalue solve...")
        openmc.run()
        source_files = sorted(glob.glob('statepoint.*.h5'))
        if not source_files:
            raise FileNotFoundError(
                "No statepoint.*.h5 file found after local eigenvalue run. "
                "Ensure OpenMC is writing statepoints (check settings.batches)."
            )
        local_source_file = source_files[-1]
    print(f"  [Shielding] Converged source: {local_source_file}")

    # ---- Step 2: Switch to fixed-source mode and rerun ----
    print("\n  [Shielding] Step 2: Fixed-source shielding transport run...")
    settings                  = openmc.Settings.from_xml()
    settings.run_mode         = 'fixed source'
    settings.particles        = params.get('Shielding Particles', 10_000)
    settings.batches          = params.get('Shielding Batches', 50)
    settings.inactive         = 0
    settings.photon_transport = params.get('Photon Transport', True)
    settings.source           = openmc.FileSource(local_source_file)
    settings.create_fission_neutrons = False
    settings.export_to_xml()

    openmc.run()

    # ---- Post-process: extract dose rates from statepoint ----
    extract_dose_results(params)


def run_openmc_shielding(build_openmc_shielding_model, params):
    """
    Execute a fixed-source OpenMC shielding transport run via the WATTS plugin.

    This function mirrors the structure of run_openmc() but is specifically
    designed for fixed-source (non-eigenvalue) shielding calculations.  The
    SD margin / isothermal temperature coefficient branching logic in run_openmc()
    is not applicable here and is intentionally omitted.

    Workflow
    --------
    1. Build the shielding model XML files via build_openmc_shielding_model(params).
    2. Execute OpenMC in fixed-source mode through the WATTS plugin.
    3. Call extract_dose_results(params) to read the statepoint and write dose
       rate values back into params.

    The WATTS plugin handles run directory management identically to the
    criticality workflow so that results are stored in the same database.

    @ In,  build_openmc_shielding_model, callable,         Model-builder callback
                                                             (e.g. build_openmc_shielding_model_LTMR).
                                                             Accepts params and produces OpenMC XML files.
    @ In,  params,                       watts.Parameters,  Simulation parameters.  Must contain all
                                                             geometry, material, shielding, and tally
                                                             keys required by the model builder.
    """
    # These flags are not used in fixed-source mode but are set as defaults so
    # that any shared helper functions that read them do not raise KeyErrors.
    params.setdefault('SD Margin Calc', False)
    params.setdefault('Isothermal Temperature Coefficients', False)
    # watts.Database().clear()  # force fresh run, bypass caching
    try:
        print(f"\n\nThe shielding results are saved at: {watts.Database().path}\n\n")

        openmc_plugin = watts.PluginOpenMC(build_openmc_shielding_model, show_stdout = True, show_stderr=True)
        openmc_plugin(params, function=lambda: run_shielding_analysis(params))

    except Exception as e:
        print("\n\n\033[91mAn error occurred while running the OpenMC shielding simulation:\033[0m\n\n")
        traceback.print_exc()