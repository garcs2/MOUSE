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
import pandas,copy


def run_shielding_analysis(params):
    """
    Callback passed to the WATTS plugin for the shielding transport run.

    Executes a two-step sequence entirely within the single WATTS working
    directory, avoiding any cross-directory file access issues:

    Step 1 — Eigenvalue run:
        openmc.run() executes the k-eigenvalue calculation built by
        build_openmc_shielding_model_LTMR, converging the fission source
        and writing it to source.{batch}.h5 in the current directory.

    Step 2 — Fixed-source shielding run:
        The converged source file is found via glob, the settings are
        modified in-place to fixed-source mode with photon transport,
        and openmc.run() is called again in the same directory.
        extract_dose_results() then reads the resulting statepoint.

    @ In, params, watts.Parameters, Simulation parameters.
    """
    from shielding.shielding_calcs import extract_dose_results

    # ---- Step 1: Eigenvalue run to converge fission source ----
    print("\n  [Shielding] Step 1: Eigenvalue run to converge fission source...")
    openmc.run()

    # Find the source file written at the final batch
    source_files = sorted(glob.glob('statepoint.*.h5'))
    if not source_files:
        raise FileNotFoundError(
            "No source.*.h5 file found after eigenvalue run. "
            "Ensure OpenMC is writing source files (check settings.batches)."
        )
    source_file = source_files[-1]
    print(f"  [Shielding] Converged source: {source_file}")

    # ---- Step 2: Switch to fixed-source mode and rerun ----
    print("\n  [Shielding] Step 2: Fixed-source shielding transport run...")
    settings                  = openmc.Settings.from_xml()
    settings.run_mode         = 'fixed source'
    settings.particles        = params.get('Shielding Particles', 10_000)
    settings.batches          = params.get('Shielding Batches', 50)
    settings.inactive         = 0
    settings.photon_transport = params.get('Photon Transport', True)
    settings.source           = openmc.FileSource(source_file)
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
    watts.Database().clear()  # force fresh run, bypass caching
    try:
        print(f"\n\nThe shielding results are saved at: {watts.Database().path}\n\n")

        openmc_plugin = watts.PluginOpenMC(build_openmc_shielding_model, show_stdout = True, show_stderr=True)
        openmc_plugin(params, function=lambda: run_shielding_analysis(params))

    except Exception as e:
        print("\n\n\033[91mAn error occurred while running the OpenMC shielding simulation:\033[0m\n\n")
        traceback.print_exc()
