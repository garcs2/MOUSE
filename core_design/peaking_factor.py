import re
import glob
from pathlib import Path

import openmc
import pandas as pd


def natural_sort_key(s: str):
    """Natural sort key: n0, n1, ..., n10 instead of n0, n1, n10, n2..."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


def compute_pin_peaking_factors(current_dir="."):
    """
    Compute pin peaking factors for all OpenMC depletion statepoints 

    For each file:
        Computes per-pin kappa-fission power
        Computes peaking factor PF = P_i / P_mean
        Prints table: Rod_ID, Peaking_Factor for that depletion step

    Prints:
      Per-step PF tables
      Final summary: [Step, Max_PF, Rod_ID_Max]

    Returns:
      summary       : DataFrame with columns [Step, Max_PF, Rod_ID_Max]
      per_step_data : dict[step] -> DataFrame [Rod_ID, Peaking_Factor, Step]
    """

    base = Path(current_dir)

    # Find depletion statepoint files: openmc_simulation_n*.h5
    sp_files = glob.glob(str(base / "openmc_simulation_n*.h5"))
    sp_files = sorted(sp_files, key=natural_sort_key)

    if not sp_files:
        print("\n[PF] No depletion statepoint files found in:", base)
        print("[PF] Expected files like 'openmc_simulation_n0.h5', 'openmc_simulation_n1.h5', ...\n")
        return pd.DataFrame(), {}

    tally_name = "pin_power_kappa"
    results = []
    per_step_data = {}

    print("\n================ PEAKING FACTOR RESULTS ================\n")

    for sp_file in sp_files:
        sp_path = Path(sp_file)
        basename = sp_path.name

        # Extract raw numeric index from "openmc_simulation_nX.h5"
        m = re.search(r"n(\d+)\.h5", basename)
        if m:
            step_raw = int(m.group(1))       # 0, 1, 2, ..., 15
            step = step_raw + 1              # 1, 2, 3, ..., 16   (shifted numbering)
        else:
            # Fallback: if pattern doesn't match, just keep the basename
            step = basename

        # Load statepoint and tally
        sp = openmc.StatePoint(str(sp_path))
        t = sp.get_tally(name=tally_name)

        # Avoid needing summary.h5 for distribcell paths
        df = t.get_pandas_dataframe(paths=False)

        # Pick index column: distribcell for pin-based, or mesh index for mesh-based tallies
        if "distribcell" in df.columns:
            id_col = "distribcell"
        elif "mesh 1" in df.columns:
            id_col = "mesh 1"
        else:
            id_col = df.columns[0]

        # Per-pin/cell power and PF
        per_pin = df.groupby(id_col)["mean"].sum()

        # Filter out zero-power cells (e.g., mesh cells with no fuel)
        per_pin = per_pin[per_pin > 0]

        pf = per_pin / per_pin.mean()

        # Per-step PF
        out = pd.DataFrame({
            "Rod_ID": per_pin.index,
            "Peaking_Factor": pf.values,
            "Step": step
        })
        per_step_data[step] = out

        # Print per-step PF table
        print(f"--- Peaking factors for depletion step {step} ---")
        print(out[["Rod_ID", "Peaking_Factor"]].to_string(index=False))
        print()

        # Collect for summary
        results.append({
            "Step": step,
            "Max_PF": float(pf.max()),
            "Rod_ID_Max": pf.idxmax()
        })

    # Build and print summary
    summary = pd.DataFrame(results).sort_values("Step")

    print("========== Peaking Factor Summary ==========")
    print(summary.to_string(index=False))
    print("============================================\n")

    return summary, per_step_data
