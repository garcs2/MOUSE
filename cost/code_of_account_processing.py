# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import pandas as pd

def remove_irrelevant_account(df, params):
    indices_to_drop = []
    
    has_sec_optional = 'Sec Optional Variable' in df.columns  # ← add this check

    for index, row in df.iterrows():
        def _optional_matches(param_val, expected_val):
            """Return True if param_val equals expected_val, or if param_val is a list that contains expected_val."""
            if isinstance(param_val, list):
                return expected_val in param_val
            return param_val == expected_val

        # Check for 'Optional Variable'
        if not pd.isna(row['Optional Variable']):
            if row['Optional Variable'] in params and _optional_matches(params[row['Optional Variable']], row['Optional Value']):
                print("\n")
                print(f"For the cost of the Account {row['Account']}: {row['Account Name']}, the {row['Optional Variable']} is selected to be {row['Optional Value']}")
                # Append the selected optional value to Account Title for clarity in the output
                df.at[index, 'Account Title'] = str(row['Account Title']) + ' - ' + str(row['Optional Value'])
            else:
                indices_to_drop.append(index)
                continue

        # Check for 'Sec Optional Variable' only if column exists
        if has_sec_optional and not pd.isna(row['Sec Optional Variable']):
            if row['Sec Optional Variable'] in params and _optional_matches(params[row['Sec Optional Variable']], row['Sec Optional Value']):
                print("\n")
                print(f"For the cost of the Account {row['Account']}: {row['Account Name']}, the {row['Sec Optional Variable']} is selected to be {row['Sec Optional Value']}")
                # Also append the sec optional value
                df.at[index, 'Account Title'] = str(df.at[index, 'Account Title']) + ' - ' + str(row['Sec Optional Value'])
            else:
                indices_to_drop.append(index)
                continue

    df.drop(indices_to_drop, inplace=True)
    return df
    return df



# --- PERFORMANCE PATCH for cost/code_of_account_processing.py ---
#
# WHAT CHANGED AND WHY IT'S SAFE:
#
# find_children_accounts() computes, for every row, the list of "child" rows
# that need to be summed into it (based on Account/Level structure) — but it
# only assigns that list to rows whose own cost is still NaN (i.e. rows that
# are aggregation targets rather than leaves with a direct cost).
#
# The key fact that makes caching safe: whether a given row is a "leaf" (gets
# a real, non-NaN direct cost from scale_cost / scale_central_facility_cost)
# or an "aggregation target" (starts NaN, to be filled in by
# calculate_high_level_accounts_cost) is a STRUCTURAL property of the chart of
# accounts. It does not depend on the randomly sampled cost values — a leaf
# account always gets a real number (or an explicit 0) every sample, and a
# parent/aggregation account always starts NaN at the point find_children_accounts
# is called for a given stage (base/other/finance/annual). So the NaN pattern,
# and therefore the entire children-accounts mapping, is IDENTICAL across all
# 1000 Monte Carlo samples for a given call site (e.g. every 'base'-stage call
# produces the same mapping, every 'other'-stage call produces the same
# mapping, etc. — only the four stages differ from each other, since costs
# get progressively filled in between stages within one sample).
#
# This patch memoizes on a fingerprint of (Account tuple, Level tuple, NaN
# pattern of the estimated-cost column). If that fingerprint has been seen
# before, the previously computed 'Children Accounts' column is reused instead
# of re-running the O(n^2) nested loop. If anything about the structure ever
# does change (e.g. a different reactor config drops/adds accounts), the
# fingerprint changes too and it recomputes automatically — so this is a
# strict speed optimization with no behavior change, not a shortcut that
# could silently go stale.
#
# HOW TO APPLY:
# Replace the existing find_children_accounts() function in
# cost/code_of_account_processing.py with the version below (it keeps the
# exact same computation logic — the numpy-array inner loop — just gates it
# behind the cache).

import pandas as pd

_children_accounts_cache = {}


def find_children_accounts(df):
    # Find the column name that starts with "Estimated Cost"
    estimated_cost_column = [col for col in df.columns if col.startswith("FOAK Estimated Cost")][0]

    levels = df['Level'].to_numpy()
    is_nan_cost = pd.isna(df[estimated_cost_column].to_numpy())

    # Fingerprint: same structure + same NaN pattern => same children mapping.
    # tuple() of small numpy arrays is fast and hashable.
    fingerprint = (
        tuple(df['Account'].to_numpy()),
        tuple(levels),
        tuple(is_nan_cost),
    )

    cached = _children_accounts_cache.get(fingerprint)
    if cached is not None:
        df['Children Accounts'] = cached
        return df

    index_strs = [str(idx) for idx in df.index]
    n = len(df)

    children_accounts = [None] * n

    for target_level in range(4, -1, -1):
        source_level = target_level + 1
        for i in range(n):
            if levels[i] == target_level and is_nan_cost[i]:
                children = []
                for j in range(i + 1, n):
                    lvl_j = levels[j]
                    if lvl_j == source_level:
                        children.append(index_strs[j])
                    elif lvl_j < source_level:
                        break
                children_accounts[i] = ','.join(children) if children else None

    _children_accounts_cache[fingerprint] = children_accounts
    df['Children Accounts'] = children_accounts
    return df


def clear_children_accounts_cache():
    """
    Call this if you run multiple *different* reactor configurations in the
    same Python process/session (e.g. a parametric sweep that changes which
    accounts are active between runs) and want to guard against unbounded
    cache growth. Not required for correctness — the fingerprint already
    prevents stale hits — this is purely to free memory if you run many
    distinct configurations back to back.
    """
    _children_accounts_cache.clear()


def get_estimated_cost_column(df, option):
    if option == 'F':
        for col in df.columns:
            if col.startswith("FOAK Estimated Cost ("):
                return col
    elif option == 'N'   :
        for col in df.columns:
            if col.startswith("NOAK Estimated Cost ("):
                return col       
    elif option == 'F std'   :
        for col in df.columns:
            if col.startswith("FOAK Estimated Cost std ("):
                return col  
    elif option == 'N std'   :
        for col in df.columns:
            if col.startswith("NOAK Estimated Cost std ("):
                return col                              
    return None



def create_cost_dictionary(df, params, tracked_params_list):
    # create a dictionary of costs we are interested in tracking
    
    # start with params we are tracking
    filtered_params = {key: params[key] for key in tracked_params_list if key in params}

    # Base accounts that are always tracked regardless of tax credit selection
    base_accounts = [
        'OCC', 'OCC per kW',
        'OCC excl. fuel', 'OCC excl. fuel per kW',
        'TCI', 'TCI per kW',
        'AC', 'AC per MWh',
        'LCOE'
    ]

    # Physics safety metrics — tracked from params directly (not from the cost dataframe)
    # These are always included if present in params; set to nan if not calculated
    # (e.g. when Shutdown Margin Calc or Isothermal Temperature Coefficients are False)
    physics_metrics = ['Temp Coeff 3D (2D corrected)', 'SDM 3D (2D corrected)']
    for metric in physics_metrics:
        if metric in params.keys():
            filtered_params[metric] = params[metric]

    # ITC-related accounts — only present if user provided 'ITC credit level' in params
    itc_accounts = [
        'OCC (ITC-adjusted)', 'OCC (ITC-adjusted) per kW',
        'TCI (ITC-adjusted)', 'TCI (ITC-adjusted) per kW',
        'LCOE (ITC-adjusted)'
    ] if 'ITC credit level' in params.keys() else []

    # PTC-related accounts — only present if user provided 'PTC credit value' in params
    ptc_accounts = [
        'LCOE with PTC'
    ] if 'PTC credit value' in params.keys() else []

    # Combine all accounts to track
    accounts = base_accounts + itc_accounts + ptc_accounts

    cost_dict = {}
    
    for account in accounts:
        cost_dict[f"{account}_FOAK Estimated Cost"] = None
        cost_dict[f"{account}_NOAK Estimated Cost"] = None
        cost_dict[f"{account}_FOAK Estimated Cost std"] = None
        cost_dict[f"{account}_NOAK Estimated Cost std"] = None
    
    # Populate the dictionary with values from the dataframe
    # If an account doesn't exist in the dataframe (e.g. ITC/PTC not used), it stays None
    for _, row in df.iterrows():
        account = row['Account']
        if account in accounts:
            cost_dict[f"{account}_FOAK Estimated Cost"] =     row[get_estimated_cost_column(df, 'F')]
            cost_dict[f"{account}_NOAK Estimated Cost"] =     row[get_estimated_cost_column(df, 'N')]
            cost_dict[f"{account}_FOAK Estimated Cost std"] = row[get_estimated_cost_column(df, 'F std')]
            cost_dict[f"{account}_NOAK Estimated Cost std"] = row[get_estimated_cost_column(df, 'N std')]  
    
    filtered_params.update(cost_dict)

    return filtered_params