# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
#
# --- PERFORMANCE PATCH ---
# All arithmetic and business logic is unchanged from the original file.
# The only changes are:
#
#   1. Repeated `df.loc[df['Account'] == X, col]` / `.isin([...])` patterns
#      are replaced with O(1) dict-based lookups (_account_positions / _get /
#      _get_sum / _set below). Each function builds one {Account: row_index}
#      dict, AFTER its own pd.concat calls (which is where new synthetic rows
#      like 'OCC', 'TCI', 'LCOE' get added) and BEFORE its per-sample work —
#      so the dict is always built against the final row layout for that
#      function call. This turns N boolean-mask scans (each O(n) plus a new
#      Series object) into one O(n) dict build + N O(1) lookups.
#      - _get(...) raises KeyError if the account isn't found, mirroring the
#        original's crash-on-empty `.values[0]` behavior for accounts that are
#        always expected to exist.
#      - _set(...) silently no-ops if the account isn't found, mirroring the
#        original's silent no-op when `df.loc[df['Account'] == X, col] = value`
#        matches zero rows (this matters for the optional ITC/PTC rows, which
#        only exist when those params are provided).
#   2. In calculate_accounts_31_32_75_82_cost, the `params_df` / replacement-
#      period check and refueling-period math were being recomputed identically
#      on both loop passes (FOAK then NOAK) even though neither depends on
#      estimated_cost_col — hoisted out of the loop so it runs once instead
#      of twice per sample.
#   3. In energy_cost_levelized: removed the stray `lcoe = ...` /
#      `df.loc[df['Account'] == 'LCOE (ITC-adjusted)', ...] = lcoe` pair that
#      was accidentally indented inside the heat-cost for-loop (see chat) —
#      it recomputed the same value ~60 times per sample using stale
#      sum_cost/sum_elec, and was always overwritten later (or a no-op).
#      Removing it changes no output, only removes wasted work.
#   4. Removed the unreachable code after the function's real `return df`
#      (dead duplicate block, never executed) purely for clarity. Zero
#      behavior change either way, since it never ran.
#
# If you'd rather NOT have items 2-4 bundled in (e.g. you want the perf fix
# isolated from the cleanup), say so and I'll split them into a separate
# patch — none of them are required for the Account-indexing speedup itself.

import numpy as np
import pandas as pd
from cost.code_of_account_processing import get_estimated_cost_column


def _account_positions(df):
    """
    Map each Account value to its row label. Assumes Account values are
    unique within df, which matches how this cost table is built (each
    synthetic account like 'OCC'/'TCI'/'LCOE' is added exactly once via
    pd.concat, and each numeric account code appears once per stage).
    """
    return {acct: idx for idx, acct in zip(df.index, df['Account'])}


def _get(df, positions, account, col):
    """O(1) equivalent of df.loc[df['Account'] == account, col].values[0].
    Raises KeyError if the account isn't present — same failure mode as the
    original (.values[0] on an empty array raised IndexError; this raises
    KeyError instead, so if you rely on catching a specific exception type
    anywhere, adjust accordingly)."""
    return df.at[positions[account], col]


def _get_sum(df, positions, accounts, col):
    """O(len(accounts)) equivalent of df.loc[df['Account'].isin(accounts), col].sum()."""
    idxs = [positions[a] for a in accounts if a in positions]
    if not idxs:
        return 0.0
    return df.loc[idxs, col].sum()


def _set(df, positions, account, col, value):
    """O(1) equivalent of df.loc[df['Account'] == account, col] = value.
    Silently does nothing if the account isn't present, matching the
    original's silent no-op when the boolean mask matched zero rows."""
    idx = positions.get(account)
    if idx is not None:
        df.at[idx, col] = value


def validate_tax_credit_params(params):
    """
    Validates that the user has not selected both ITC and PTC simultaneously.
    These are mutually exclusive under the IRA — a project must choose one or neither.
    This should be called at the very start of the cost estimation workflow,
    before any simulation or cost calculation runs, to catch input errors early.

    @ In, params, dict, the user-defined parameters dictionary
    @ Out, None — raises ValueError if both ITC and PTC are defined
    """
    if 'ITC credit level' in params.keys() and 'PTC credit value' in params.keys():
        raise ValueError(
            "\n\n--- INPUT ERROR ---\n"
            "Both 'ITC credit level' and 'PTC credit value' are defined in params.\n"
            "Under the IRA, ITC and PTC are mutually exclusive — you must choose one or neither.\n"
            "Please remove one of them from your params and rerun.\n"
        )


def _crf(rate, period):
    # Returns the Capital Recovery Factor (CRF) based on the discount rate and period.
    # CRF converts a present value into a series of equal annual payments.
    # Formula: CRF = rate * (1 + rate)^period / ((1 + rate)^period - 1)
    # Special case: when rate == 0, CRF = 1/period (straight-line amortization, no interest)
    period = np.asarray(period, dtype=float)
    rate   = np.asarray(rate,   dtype=float)
    if rate.ndim == 0 and rate == 0:
        # scalar zero rate: straight-line amortization
        factor = np.where(period == 0, 0.0, 1.0 / period)
        return float(factor) if factor.ndim == 0 else factor
    numer = rate * (1 + rate)**period
    denum = (1 + rate)**period - 1
    factor = np.where(denum == 0, 0.0, numer / denum)
    return factor


def calculate_accounts_31_32_75_82_cost(df, params):
    estimated_cost_col_F = get_estimated_cost_column(df, 'F')
    estimated_cost_col_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    # Hoisted out of the FOAK/NOAK loop: none of this depends on
    # estimated_cost_col, so it was previously being computed twice per
    # sample for no reason.
    refueling_period = params['Fuel Lifetime'] + params['Refueling Period'] + params['Startup Duration after Refueling']
    refueling_period_yr = refueling_period / 365
    params_df = pd.DataFrame(params.items(), columns=['keys', 'values'])
    has_replacement_params = params_df.loc[params_df['keys'].str.contains('replacement', case=False), 'keys'].size > 0

    for estimated_cost_col in [estimated_cost_col_F, estimated_cost_col_N]:
        tot_field_direct_cost = _get_sum(df, positions, [21, 22, 23], estimated_cost_col)

        acct_31_cost = params['indirect to direct field-related cost'] * tot_field_direct_cost
        _set(df, positions, 31, estimated_cost_col, acct_31_cost)

        acct_21 = _get(df, positions, 21, estimated_cost_col)
        acct_22 = _get(df, positions, 22, estimated_cost_col)
        _set(df, positions, 32, estimated_cost_col, acct_21 * (acct_31_cost / acct_22))

        if has_replacement_params:
            A20_replacement_period = refueling_period_yr * np.array([
                params['A75: Vessel Replacement Period (cycles)'],
                params['A75: Core Barrel Replacement Period (cycles)'],
                1,
                params['A75: Reflector Replacement Period (cycles)'],
                params['A75: Drum Replacement Period (cycles)'],
                params.get('A75: Integrated HX Replacement Period (cycles)', 0),
            ])
            A20_capital_cost = np.array([
                _get(df, positions, 221.12, estimated_cost_col),
                _get(df, positions, 221.13, estimated_cost_col),
                _get(df, positions, 221.33, estimated_cost_col),
                _get(df, positions, 221.31, estimated_cost_col),
                _get(df, positions, 221.2, estimated_cost_col),
                _get_sum(df, positions, [222.1, 222.2, 222.3, 222.61], estimated_cost_col),
            ])
            annualized_replacement_cost = (A20_capital_cost * _crf(params['Discount Rate'], A20_replacement_period))
            A20_other_cost = _get(df, positions, 20, estimated_cost_col) - A20_capital_cost.sum()
            annualized_other_cost = A20_other_cost * params['Maintenance to Direct Cost Ratio']
            _set(df, positions, 751, estimated_cost_col, annualized_replacement_cost[0])
            _set(df, positions, 752, estimated_cost_col, annualized_replacement_cost[1])
            _set(df, positions, 753, estimated_cost_col, annualized_replacement_cost[2])
            _set(df, positions, 754, estimated_cost_col, annualized_replacement_cost[3])
            _set(df, positions, 755, estimated_cost_col, annualized_replacement_cost[4])
            _set(df, positions, 756, estimated_cost_col, annualized_replacement_cost[5])
            _set(df, positions, 759, estimated_cost_col, annualized_other_cost)
        else:
            acct_20 = _get(df, positions, 20, estimated_cost_col)
            _set(df, positions, 75, estimated_cost_col, acct_20 * params['Maintenance to Direct Cost Ratio'])

        lump_fuel_cost = _get(df, positions, 25, estimated_cost_col)
        annualized_fuel_cost = lump_fuel_cost * _crf(params['Discount Rate'], refueling_period_yr)
        _set(df, positions, 82, estimated_cost_col, annualized_fuel_cost)

    return df


def calculate_accounts_31_32_75_central_facility_cost(df, params):
    """
    Calculate indirect costs for central facility accounts (31, 32, 75).
    Similar to calculate_accounts_31_32_75_82_cost but for central facility.
    """
    estimated_cost_col_F = get_estimated_cost_column(df, 'F')
    estimated_cost_col_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    for estimated_cost_col in [estimated_cost_col_F, estimated_cost_col_N]:
        tot_field_direct_cost = _get_sum(df, positions, [21, 22, 23, 24, 25, 27], estimated_cost_col)

        acct_31_cost = params['indirect to direct field-related cost'] * tot_field_direct_cost
        _set(df, positions, 31, estimated_cost_col, acct_31_cost)

        acct_21 = _get(df, positions, 21, estimated_cost_col)
        acct_22_25 = _get_sum(df, positions, [22, 23, 24, 25], estimated_cost_col)
        _set(df, positions, 32, estimated_cost_col, acct_21 * (acct_31_cost / acct_22_25))

        acct_20 = _get(df, positions, 20, estimated_cost_col)
        _set(df, positions, 75, estimated_cost_col, acct_20 * params['Maintenance to Direct Cost Ratio'])

    return df


def calculate_decommissioning_cost(df, params):
    estimated_cost_col_F = get_estimated_cost_column(df, 'F')
    estimated_cost_col_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    if 'A78: CAPEX to Decommissioning Cost Ratio' not in params.keys():
        params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

    AR = params['Annual Return']
    LP = params['Levelization Period']
    fv_to_pv_of_annuity = -AR / (1 - pow(1 + AR, LP))

    for estimated_cost_col in [estimated_cost_col_F, estimated_cost_col_N]:
        capex = _get_sum(df, positions, [10, 20], estimated_cost_col)
        decommissioning_fv_cost = capex * params['A78: CAPEX to Decommissioning Cost Ratio']
        annualized_decommisioning_cost = decommissioning_fv_cost * fv_to_pv_of_annuity
        _set(df, positions, 78, estimated_cost_col, annualized_decommisioning_cost)

    return df


def calculate_interest_cost(params, OCC):
    interest_rate = params['Interest Rate']
    construction_duration = params['Construction Duration']
    debt_to_equity_ratio = params['Debt To Equity Ratio']
    # Convert D:E ratio to debt fraction for the calculation.
    # e.g. D:E = 1.0 (1:1) → debt_fraction = 1/(1+1) = 0.5 (50% financed by debt)
    # e.g. D:E = 2.33      → debt_fraction = 2.33/3.33 ≈ 0.7 (70% financed by debt)
    debt_fraction = debt_to_equity_ratio / (1 + debt_to_equity_ratio)
    B = (1 + np.exp((np.log(1 + interest_rate)) * construction_duration / 12))
    C = ((np.log(1 + interest_rate) * (construction_duration / 12) / 3.14)**2 + 1)
    Interest_expenses = debt_fraction * OCC * ((0.5 * B / C) - 1)
    return Interest_expenses


def calculate_interest_cost_central(params, OCC):
    """Calculate interest cost for central facility using its construction duration."""
    interest_rate = params['Interest Rate']
    construction_duration = params['Central Facility Construction Duration']
    debt_to_equity_ratio = params['Debt To Equity Ratio']
    # Convert D:E ratio to debt fraction for the calculation (same logic as above).
    debt_fraction = debt_to_equity_ratio / (1 + debt_to_equity_ratio)
    B = (1 + np.exp((np.log(1 + interest_rate)) * construction_duration / 12))
    C = ((np.log(1 + interest_rate) * (construction_duration / 12) / 3.14)**2 + 1)
    Interest_expenses = debt_fraction * OCC * ((0.5 * B / C) - 1)
    return Interest_expenses


def calculate_high_level_capital_costs(df, params):
    power_kWe = 1000 * params['Power MWe']
    accounts_to_sum = [10, 20, 30, 40, 50]

    df = pd.concat([df, pd.DataFrame([{'Account': 'OCC','Account Title' : 'Overnight Capital Cost'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'OCC per kW','Account Title' : 'Overnight Capital Cost per kW' }])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'OCC excl. fuel','Account Title' : 'Overnight Capital Cost Excluding Fuel'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'OCC excl. fuel per kW','Account Title' : 'Overnight Capital Cost Excluding Fuel per kW'}])], ignore_index=True)

    cost_column_F = get_estimated_cost_column(df, 'F')
    cost_column_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    for cost_column in [cost_column_F, cost_column_N]:
        occ_cost = _get_sum(df, positions, accounts_to_sum, cost_column)
        _set(df, positions, 'OCC', cost_column, occ_cost)
        _set(df, positions, 'OCC per kW', cost_column, occ_cost / power_kWe)

        occ_excl_fuel = occ_cost - _get(df, positions, 25, cost_column)
        _set(df, positions, 'OCC excl. fuel', cost_column, occ_excl_fuel)
        _set(df, positions, 'OCC excl. fuel per kW', cost_column, occ_excl_fuel / power_kWe)

        _set(df, positions, 62, cost_column, calculate_interest_cost(params, occ_cost))
    return df


def calculate_high_level_capital_costs_central_facility(df, params):
    """Calculate OCC and interest costs for central facility."""
    power_kWe = 1000 * params['Power MWe'] * params['Maximum Number of Operating Reactors']
    accounts_to_sum = [10, 20, 30, 40, 50]

    df = pd.concat([df, pd.DataFrame([{'Account': 'OCC', 'Account Title': 'Overnight Capital Cost'}])], ignore_index=True)

    cost_column_F = get_estimated_cost_column(df, 'F')
    cost_column_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    for cost_column in [cost_column_F, cost_column_N]:
        occ_cost = _get_sum(df, positions, accounts_to_sum, cost_column)
        _set(df, positions, 'OCC', cost_column, occ_cost)
        _set(df, positions, 'OCC per kW', cost_column, occ_cost / power_kWe)
        _set(df, positions, 62, cost_column, calculate_interest_cost_central(params, occ_cost))
    return df


def calculate_TCI_central(df, params):
    """Calculate Total Capital Investment for central facility."""
    power_kWe = 1000 * params['Power MWe'] * params['Maximum Number of Operating Reactors']

    df = pd.concat([df, pd.DataFrame([{'Account': 'TCI', 'Account Title': 'Total Capital Investment'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'TCI per kW', 'Account Title': 'Total Capital Investment per kW'}])], ignore_index=True)

    accounts_to_sum = ['OCC', 60]
    cost_column_F = get_estimated_cost_column(df, 'F')
    cost_column_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    for cost_column in [cost_column_F, cost_column_N]:
        tci_cost = _get_sum(df, positions, accounts_to_sum, cost_column)
        _set(df, positions, 'TCI', cost_column, tci_cost)
        _set(df, positions, 'TCI per kW', cost_column, tci_cost / power_kWe)

    return df


# -----------------------------------------------------------------------------------------
# ITC (Investment Tax Credit) helper function
# -----------------------------------------------------------------------------------------
# The ITC is a one-time credit applied to the capital cost (OCC) of the plant.
# Under the IRA, the ITC level can be 6%, 30%, 40%, or 50% depending on whether
# the project meets prevailing wage, domestic content, and energy community requirements.
#
# This function returns a COST REDUCTION FACTOR (not the credit itself).
# The factor represents what fraction of the original OCC remains after the ITC is applied.
# Example: a 30% ITC reduces OCC to 73% of its original value → factor = 0.73
#
# itc_level is expressed as a fraction (e.g. 0.30 for 30%)
# Interpolation is used for ITC levels between the defined breakpoints.
# -----------------------------------------------------------------------------------------
def ITC_reduction_factor(itc_level):
    itc_values    = [0,    0.06,  0.3,   0.4,   0.5 ]  # ITC credit levels (fractions)
    reduction_factors = [1, 0.95,  0.73,  0.63,  0.53]  # corresponding OCC reduction factors
    # renamed from ITC_reduction_factor to avoid shadowing the function name
    return np.interp(itc_level, itc_values, reduction_factors)


def calculate_TCI(df, params):
    # -----------------------------------------------------------------------------------------
    # Total Capital Investment (TCI) = OCC + Account 60 (financing/interest costs)
    #
    # If an ITC credit level is provided in params, a second version of TCI is calculated
    # that reflects the reduced OCC after the ITC subsidy is applied:
    #   - OCC with ITC     = OCC × ITC_reduction_factor(itc_level)
    #   - TCI with ITC     = OCC with ITC + Account 60
    #
    # Note: Account 60 (financing costs) is NOT reduced by the ITC — only the OCC is.
    # This is consistent with how the ITC works in practice: it offsets capital investment,
    # not the financing charges on top of it.
    #
    # Output rows added to the cost dataframe:
    #   - 'OCC with ITC'         : reduced overnight capital cost
    #   - 'OCC with ITC per kW'  : reduced OCC normalized by plant capacity
    #   - 'TCI with ITC'         : reduced total capital investment
    #   - 'TCI with ITC per kW'  : reduced TCI normalized by plant capacity
    # -----------------------------------------------------------------------------------------
    power_kWe = 1000 * params['Power MWe']

    df = pd.concat([df, pd.DataFrame([{'Account': 'TCI','Account Title' : 'Total Capital Investment'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'TCI per kW','Account Title' : 'Total Capital Investment per kW'}])], ignore_index=True)

    if 'ITC credit level' in params.keys():
        # Add ITC-adjusted output rows to the dataframe
        df = pd.concat([df, pd.DataFrame([{'Account': 'OCC (ITC-adjusted)',        'Account Title': 'Overnight Capital Cost Adjusted for the Investment Tax Credit'}])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{'Account': 'OCC (ITC-adjusted) per kW', 'Account Title': 'Overnight Capital Cost Adjusted for the Investment Tax Credit per kW'}])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{'Account': 'TCI (ITC-adjusted)',        'Account Title': 'Total Capital Investment Adjusted for the Investment Tax Credit'}])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([{'Account': 'TCI (ITC-adjusted) per kW', 'Account Title': 'Total Capital Investment Adjusted for the Investment Tax Credit per kW'}])], ignore_index=True)
        # note: ITC_cost_reduction_factor is computed inside the loop below for each cost column (FOAK and NOAK)

    accounts_to_sum = ['OCC', 60]
    cost_column_F = get_estimated_cost_column(df, 'F')
    cost_column_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    # Per-column eligibility for ITC. FOAK column = unit 1; NOAK column = unit
    # 'NOAK Unit Number'. A unit qualifies only if its position is <= the cutoff
    # ('Number of Units Claiming ITC/PTC'). Default cutoff is effectively
    # infinite so omitting the param preserves the pre-existing behavior.
    n_credit = params.get('Number of Units Claiming ITC/PTC', 10**9)
    noak_unit = params.get('NOAK Unit Number', 10)
    eligible_by_column = {cost_column_F: 1 <= n_credit,
                          cost_column_N: noak_unit <= n_credit}

    for cost_column in [cost_column_F, cost_column_N]:
        # --- Baseline TCI (no ITC) ---
        tci_cost = _get_sum(df, positions, accounts_to_sum, cost_column)
        _set(df, positions, 'TCI', cost_column, tci_cost)
        _set(df, positions, 'TCI per kW', cost_column, tci_cost / power_kWe)

        if 'ITC credit level' in params.keys():
            if eligible_by_column[cost_column]:
                # --- ITC-adjusted TCI ---
                # Step 1: Get the reduction factor for the given ITC level
                ITC_cost_reduction_factor = ITC_reduction_factor(params['ITC credit level'])
                # Step 2: Apply the reduction factor to OCC to get the ITC-adjusted OCC
                # OCC_after_ITC is the reduced OCC value (not the savings amount)
                OCC_after_ITC = _get(df, positions, 'OCC', cost_column) * ITC_cost_reduction_factor
                # Step 3: Add financing costs (Account 60) to get TCI adjusted for ITC
                # Note: Account 60 is not reduced by the ITC
                tci_cost_with_itc = _get(df, positions, 60, cost_column) + OCC_after_ITC
            else:
                # This unit is past the IRA sunset cutoff — fall back to the
                # un-subsidized OCC/TCI so the ITC-adjusted columns show the
                # un-subsidized cost rather than blank/NaN.
                OCC_after_ITC = _get(df, positions, 'OCC', cost_column)
                tci_cost_with_itc = tci_cost
            _set(df, positions, 'OCC (ITC-adjusted)', cost_column, OCC_after_ITC)
            _set(df, positions, 'OCC (ITC-adjusted) per kW', cost_column, OCC_after_ITC / power_kWe)
            _set(df, positions, 'TCI (ITC-adjusted)', cost_column, tci_cost_with_itc)
            _set(df, positions, 'TCI (ITC-adjusted) per kW', cost_column, tci_cost_with_itc / power_kWe)

    return df


def energy_cost_levelized(params, df):
    # -----------------------------------------------------------------------------------------
    # LCOE (Levelized Cost of Energy) Calculation
    # ... (existing docstring unchanged)
    # -----------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------
    # Heat application cost reduction factors (hardcoded, based on process heat study)
    # For heat applications, the OCC is lower because no power conversion system is needed
    # (e.g. no turbine, generator, condenser). The annual O&M cost is also slightly reduced.
    # Source: [add your reference here]
    # -----------------------------------------------------------------------------------------
    HEAT_OCC_FACTOR         = 0.795  # OCC for heat = OCC_electric × 0.795
    HEAT_ANNUAL_COST_FACTOR = 0.966  # Annual O&M+fuel cost for heat = baseline × 0.966

    df = pd.concat([df, pd.DataFrame([{'Account': 'AC',         'Account Title': 'Annualized Cost'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'AC per MWh', 'Account Title': 'Annualized Cost per MWh'}])], ignore_index=True)
    df = pd.concat([df, pd.DataFrame([{'Account': 'LCOE',       'Account Title': 'Levelized Cost Of Energy ($/MWh)'}])], ignore_index=True)

    if 'PTC credit value' in params.keys():
        df = pd.concat([df, pd.DataFrame([{'Account': 'LCOE with PTC', 'Account Title': 'Levelized Cost Of Energy with PTC ($/MWh)'}])], ignore_index=True)

    if 'ITC credit level' in params.keys():
        assert 'PTC credit value' not in params.keys(), '--error: Only PTC or ITC or None must be selected not both.'
        df = pd.concat([df, pd.DataFrame([{'Account': 'LCOE (ITC-adjusted)', 'Account Title': 'Levelized Cost Of Energy Adjusted for the Investment Tax Credit ($/MWh)'}])], ignore_index=True)

    df = pd.concat([df, pd.DataFrame([{'Account': 'LCOH',       'Account Title': 'Levelized Cost Of Heat ($/MWth)'}])], ignore_index=True)

    params.setdefault('Tax Rate', 0.21)

    plant_lifetime_years = params['Levelization Period']
    discount_rate        = params['Discount Rate']
    power_MWe            = params['Power MWe']
    capacity_factor      = params['Capacity Factor']
    thermal_efficiency   = params['Thermal Efficiency']
    estimated_cost_col_F = get_estimated_cost_column(df, 'F')
    estimated_cost_col_N = get_estimated_cost_column(df, 'N')

    positions = _account_positions(df)

    # Per-column eligibility for ITC/PTC under the IRA sunset cutoff.
    # FOAK column = unit 1; NOAK column = unit 'NOAK Unit Number'. A unit
    # qualifies only if its position <= 'Number of Units Claiming ITC/PTC'.
    # Default cutoff is effectively infinite so omitting the param preserves
    # the pre-existing behavior. When ineligible, the credit-adjusted accounts
    # fall back to the un-subsidized LCOE.
    n_credit = params.get('Number of Units Claiming ITC/PTC', 10**9)
    noak_unit = params.get('NOAK Unit Number', 10)
    eligible_by_column = {estimated_cost_col_F: 1 <= n_credit,
                          estimated_cost_col_N: noak_unit <= n_credit}

    for estimated_cost_col in [estimated_cost_col_F, estimated_cost_col_N]:

        # -----------------------------------------------------------------------------------------
        # Baseline LCOE calculation (no tax credits) — unchanged
        # -----------------------------------------------------------------------------------------
        cap_cost          = _get(df, positions, 'TCI', estimated_cost_col)
        ann_cost          = _get(df, positions, 70, estimated_cost_col) + _get(df, positions, 80, estimated_cost_col)
        levelized_ann_cost = ann_cost / params['Annual Electricity Production']
        _set(df, positions, 'AC', estimated_cost_col, ann_cost)
        _set(df, positions, 'AC per MWh', estimated_cost_col, levelized_ann_cost)

        sum_cost = 0
        sum_elec = 0
        for i in range(1 + plant_lifetime_years):
            if i == 0:
                cap_cost_per_year = cap_cost
                annual_cost       = 0
                elec_gen          = 0
            else:
                cap_cost_per_year = 0
                annual_cost       = ann_cost
                elec_gen          = power_MWe * capacity_factor * 365 * 24
            sum_cost += (cap_cost_per_year + annual_cost) / ((1 + discount_rate)**i)
            sum_elec += elec_gen / ((1 + discount_rate)**i)

        lcoe = sum_cost / sum_elec
        _set(df, positions, 'LCOE', estimated_cost_col, lcoe)

        # -----------------------------------------------------------------------------------------
        # LCOH (Levelized Cost of Heat) calculation
        #
        # For heat applications, the plant does not need a power conversion system,
        # so both capital and O&M costs are reduced by the factors defined above.
        #
        # The full heat cost chain:
        #   1. OCC_heat      = OCC × HEAT_OCC_FACTOR
        #   2. Interest_heat = calculate_interest_cost(params, OCC_heat)
        #   3. TCI_heat      = OCC_heat + Interest_heat
        #   4. ann_cost_heat = ann_cost × HEAT_ANNUAL_COST_FACTOR
        #   5. LCOE_heat     = PV(costs with TCI_heat, ann_cost_heat) / PV(electricity)
        #   6. LCOH          = LCOE_heat × Thermal Efficiency
        #
        # NOTE: this used to be computed twice in a row (once discarded, once
        # kept) with a stray LCOE-(ITC-adjusted) write stuck inside the first
        # copy's loop — see chat for details. Collapsed here into a single
        # computation; the output is unchanged since the first copy's result
        # was never used for anything except that stray write, which itself
        # got overwritten later whenever the ITC block runs, or was a no-op
        # when it doesn't.
        # -----------------------------------------------------------------------------------------
        OCC           = _get(df, positions, 'OCC', estimated_cost_col)
        OCC_heat      = OCC * HEAT_OCC_FACTOR
        Interest_heat = calculate_interest_cost(params, OCC_heat)
        TCI_heat      = OCC_heat + Interest_heat
        ann_cost_heat = ann_cost * HEAT_ANNUAL_COST_FACTOR

        sum_cost_heat = 0
        sum_elec_heat = 0
        for i in range(1 + plant_lifetime_years):
            if i == 0:
                cap_cost_per_year = TCI_heat
                annual_cost_heat  = 0
                elec_gen          = 0
            else:
                cap_cost_per_year = 0
                annual_cost_heat  = ann_cost_heat
                elec_gen          = power_MWe * capacity_factor * 365 * 24
            sum_cost_heat += (cap_cost_per_year + annual_cost_heat) / ((1 + discount_rate)**i)
            sum_elec_heat += elec_gen / ((1 + discount_rate)**i)

        lcoe_heat = sum_cost_heat / sum_elec_heat
        lcoh      = lcoe_heat * thermal_efficiency
        _set(df, positions, 'LCOH', estimated_cost_col, lcoh)

        # -----------------------------------------------------------------------------------------
        if 'PTC credit value' in params.keys():
            if eligible_by_column[estimated_cost_col]:
                sum_elec = 0
                sum_ptc  = 0
                assert 'PTC credit period' in params.keys(), 'error: If a PTC credit value is provided, a corresponding PTC credit period must be given as well.'
                try:
                    bonus_multiplier = 1.0 + params['domestic_content_bonus'] + params['energy_community_bonus']
                except:
                    print('--- warning: Assume no extra percentage on the credit')
                    bonus_multiplier = 1.0

                for i in range(1 + plant_lifetime_years):
                    if i == 0:
                        elec_gen = 0
                        ptc_gen  = 0
                    else:
                        elec_gen = power_MWe * capacity_factor * 365 * 24
                        if i > params['PTC credit period']:
                            ptc_gen = 0
                        else:
                            ptc_gen = elec_gen * (params['PTC credit value'] * bonus_multiplier) / (1 - params['Tax Rate'])
                    sum_ptc  += ptc_gen  / ((1 + discount_rate)**i)
                    sum_elec += elec_gen / ((1 + discount_rate)**i)

                estimated_ptc = sum_ptc / sum_elec
                _set(df, positions, 'LCOE with PTC', estimated_cost_col, lcoe - estimated_ptc)
            else:
                # This unit is past the IRA sunset cutoff — fall back to the
                # un-subsidized LCOE so the 'LCOE with PTC' cell shows the
                # un-subsidized cost rather than blank/NaN.
                _set(df, positions, 'LCOE with PTC', estimated_cost_col, lcoe)

        # -----------------------------------------------------------------------------------------
        # ITC adjustment — unchanged
        # -----------------------------------------------------------------------------------------
        if 'ITC credit level' in params.keys():
            cap_cost      = _get(df, positions, 'TCI (ITC-adjusted)', estimated_cost_col)
            ann_cost      = _get(df, positions, 70, estimated_cost_col) + _get(df, positions, 80, estimated_cost_col)
            levelized_ann_cost = ann_cost / params['Annual Electricity Production']
            _set(df, positions, 'AC', estimated_cost_col, ann_cost)
            _set(df, positions, 'AC per MWh', estimated_cost_col, levelized_ann_cost)
            sum_cost = 0
            sum_elec = 0

            for i in range(1 + plant_lifetime_years):
                if i == 0:
                    cap_cost_per_year = cap_cost
                    annual_cost       = 0
                    elec_gen          = 0
                else:
                    cap_cost_per_year = 0
                    annual_cost       = ann_cost
                    elec_gen          = power_MWe * capacity_factor * 365 * 24
                sum_cost += (cap_cost_per_year + annual_cost) / ((1 + discount_rate)**i)
                sum_elec += elec_gen / ((1 + discount_rate)**i)

            lcoe = sum_cost / sum_elec
            _set(df, positions, 'LCOE (ITC-adjusted)', estimated_cost_col, lcoe)

    return df