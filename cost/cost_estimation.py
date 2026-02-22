# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import os
import pandas as pd
import numpy as np
import csv
from cost.cost_escalation import escalate_cost_database
from cost.code_of_account_processing import remove_irrelevant_account, get_estimated_cost_column, find_children_accounts, create_cost_dictionary
from cost.cost_scaling import scale_cost, scale_redundant_BOP_and_primary_loop
from cost.non_direct_cost import validate_tax_credit_params, calculate_accounts_31_32_75_82_cost, calculate_decommissioning_cost, calculate_high_level_capital_costs, calculate_TCI, energy_cost_levelized
from cost.params_registry import PARAMS_REGISTRY, GROUP_ORDER
from reactor_engineering_evaluation.operation import reactor_operation


def calculate_high_level_accounts_cost(df, target_level, option, FOAK_or_NOAK):
    cost_column = get_estimated_cost_column(df, FOAK_or_NOAK)
    # print(f"Updating costs of the level {target_level} accounts for the {cost_column}")

    if option == "base":
        valid_prefixes = ('1', '2')
    elif option == "other":
        valid_prefixes = ('3', '4', '5')
    elif option == "finance": 
        valid_prefixes = ('6')  
    elif option == "annual": 
        valid_prefixes = ('7', '8')      
    else:
        raise ValueError("Invalid option. Choose 'base' or 'other' or 'finance' or 'annual'.")

    for index, row in df.iterrows():
        if str(row["Account"]).startswith(valid_prefixes):
            if row["Level"] == target_level and pd.isna(row[cost_column]):
                children_accounts = row["Children Accounts"]
                if not pd.isna(children_accounts):
                    children_accounts_list = children_accounts.split(",")
                    total_sum = 0
                    for account in children_accounts_list:
                        account_value = float(account)
                        total_sum += df[df["Account"] == account_value][cost_column].values[0]
                    df.at[index, cost_column] = total_sum

    return df


def update_high_level_costs(scaled_cost, option, sample):
    df_with_children_accounts = find_children_accounts(scaled_cost)
    no_subaccounts_list = []

    if option == "base":
        valid_prefixes = ('1', '2')
    elif option == "other":
        valid_prefixes = ('3', '4', '5')
    elif option == "finance": 
        valid_prefixes = ('6')  
    elif option == "annual": 
        valid_prefixes = ('7', '8')      
    else:
        raise ValueError("Invalid option. Choose 'base' or 'other' or 'finance' or 'annual'.")

    for level in range(4, -1, -1):
        df_updated = calculate_high_level_accounts_cost(df_with_children_accounts, level, option, 'F')
        df_updated_2 = calculate_high_level_accounts_cost(df_updated, level, option, 'N')
        
        for index, row in df_updated_2.iterrows():
            if str(row["Account"]).startswith(valid_prefixes):
                if row['Level'] == level and pd.isna(row[get_estimated_cost_column(df_updated_2, 'F')]) and pd.isna(row['Children Accounts']):
                    df_updated_2.at[index, get_estimated_cost_column(df_updated_2, 'F')] = 0
                    no_subaccounts_list.append(row['Account'])
                if row['Level'] == level and pd.isna(row[get_estimated_cost_column(df_updated_2, 'N')]) and pd.isna(row['Children Accounts']):
                    df_updated_2.at[index, get_estimated_cost_column(df_updated_2, 'N')] = 0
                    no_subaccounts_list.append(row['Account'])
    
    if sample == 0:
        if no_subaccounts_list:
            print(f"Warning: The following accounts do not have any subaccounts: {', '.join(map(str, set(no_subaccounts_list))) }")
    return df_updated_2


def save_params_to_excel_file(excel_file, params):
    """
    Saves the params dictionary to the 'Parameters' sheet of the output Excel file.
    Parameters are organized into labeled groups, sorted alphabetically within each group,
    with units, descriptions, and source (User Input vs Calculated) for each parameter.
    Array parameters are summarized (BOL, EOL, min, max) rather than shown as raw lists.
    Parameters not found in the registry are placed in an 'Uncategorized' group with a warning.
    """

    def format_value(val):
        """Format a single scalar value for display."""
        if isinstance(val, float) and np.isnan(val):
            return 'N/A'
        if isinstance(val, bool):
            return str(val).upper()
        return val

    def handle_array(name, val, mode, units, description, source):
        """
        Expand an array parameter into multiple display rows based on mode:
          'summary' → BOL, EOL, min, max
          'steps'   → first step, last step, number of steps
          'as_is'   → single row with the list as a string
        Returns a list of (display_name, value, units, description, source) tuples.
        """
        rows = []
        if not isinstance(val, (list, tuple)) or len(val) == 0:
            rows.append((name, format_value(val), units, description, source))
            return rows

        if mode == 'summary':
            rows.append((f'{name} (BOL)',   round(val[0], 6),   units, f'{description} — beginning of life', source))
            rows.append((f'{name} (EOL)',   round(val[-1], 6),  units, f'{description} — end of life',       source))
            rows.append((f'{name} (min)',   round(min(val), 6), units, f'{description} — minimum value',     source))
            rows.append((f'{name} (max)',   round(max(val), 6), units, f'{description} — maximum value',     source))
        elif mode == 'steps':
            rows.append((f'{name} (first)', format_value(val[0]),  units, f'{description} — first step',     source))
            rows.append((f'{name} (last)',  format_value(val[-1]), units, f'{description} — last step',      source))
            rows.append((f'{name} (count)', len(val),              '',    f'{description} — number of steps', source))
        elif mode == 'as_is':
            rows.append((name, str(val), units, description, source))
        else:
            rows.append((name, str(val), units, description, source))
        return rows

    # ---------------------------------------------------------------
    # Build grouped rows from params using the registry
    # ---------------------------------------------------------------
    groups = {g: [] for g in GROUP_ORDER}
    params_dict = dict(params)

    for param_name, value in sorted(params_dict.items()):  # alphabetical within each group
        entry = PARAMS_REGISTRY.get(param_name)

        if entry is None:
            # Not in registry — place in Uncategorized with a warning marker
            if isinstance(value, (list, tuple)) and len(value) > 10:
                display_value = f'[list of {len(value)} items — see input file]'
            else:
                display_value = format_value(value)
            groups['Uncategorized'].append((
                param_name,
                display_value,
                '',
                '--- Not in params registry. Please add to cost/params_registry.py ---',
                'Unknown'
            ))
            continue

        # Skip hidden parameters
        if entry.get('hidden', False):
            continue

        units       = entry.get('units', '')
        description = entry.get('description', '')
        source      = entry.get('source', '')
        array_mode  = entry.get('array_mode', None)
        group       = entry.get('group', 'Uncategorized')

        if group not in groups:
            group = 'Uncategorized'

        if array_mode is not None and isinstance(value, (list, tuple)):
            rows = handle_array(param_name, value, array_mode, units, description, source)
            groups[group].extend(rows)
        else:
            groups[group].append((param_name, format_value(value), units, description, source))

    # ---------------------------------------------------------------
    # Build the final list of rows with group headers and separators
    # ---------------------------------------------------------------
    all_rows = []
    columns = ['Group', 'Parameter', 'Value', 'Units', 'Description', 'Source']

    for group_name in GROUP_ORDER:
        rows = groups.get(group_name, [])
        if not rows:
            continue  # skip empty groups

        # Group header row
        all_rows.append([f'=== {group_name.upper()} ===', '', '', '', '', ''])

        for (pname, pval, punits, pdesc, psource) in rows:
            all_rows.append([group_name, pname, pval, punits, pdesc, psource])

        # Blank separator row between groups
        all_rows.append(['', '', '', '', '', ''])

    # ---------------------------------------------------------------
    # Write to the Parameters sheet using the existing ExcelWriter
    # ---------------------------------------------------------------
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_excel(excel_file, sheet_name='Parameters', index=False)

    total_params = sum(len(rows) for rows in groups.values())
    active_groups = sum(1 for g in GROUP_ORDER if groups.get(g))
    print(f"\n\nParameters saved — {total_params} entries across {active_groups} groups.\n\n")


def transform_dataframe(df):
    """
    Divides all values in the specified column by one million, except the last two rows,
    rounds to one non-zero digit after the decimal point, and appends 'M'. Converts the last two rows to integers.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.

    Returns:
    pd.DataFrame: The modified dataframe.
    """
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df = df.loc[~(df[numerical_columns] == 0).all(axis=1)]
    df[numerical_columns] = df[numerical_columns].astype(int)
    return df


def learning_rate_multiplier(learning_rate, number_of_units):
    return pow(1-learning_rate, np.log2(min(100, number_of_units)))


def FOAK_to_NOAK(df, params):
    # Additional Cost Scaling Based on Assumed Learning Rate
    # Learning Rate and Cost multiplier is based on 
    # DOI: 10.1080/00295450.2023.2206779
    # Cost Multiplier is capped after the 100th Unit for any component
    if 'NOAK Unit Number' not in params.keys():
        # Assume the default value if no `NOAK Unit Number` is specified.
        params['NOAK Unit Number'] = 10
        # Custom Check to see if input specifies which Nth-of-a-Kind
        # Default is ~10th with 20(2*10)-units assumed for Onsite Learning
    params['Assumed Number Of Units For Onsite Learning'] = params['NOAK Unit Number'] * 2
    
    for multiplier_type in ['No Learning', 
                            'Licensing Learning', 
                            'Factory Primary Structure', 
                            'Factory Drums',
                            'Factory Other', 
                            'Factory Be',
                            'Factory BeO',
                            'Non-nuclear off-the-shelf']:
        params[f"{multiplier_type} Cost Multiplier"] = learning_rate_multiplier(params[f'{multiplier_type}'], 
                                                                                params['NOAK Unit Number'])
    params['Onsite Learning Cost Multiplier'] = learning_rate_multiplier(params['Onsite Learning'], 
                                                                         params['Assumed Number Of Units For Onsite Learning'])

    def get_multiplier(multiplier_type):
        if multiplier_type in ['No Learning', 
                               'Licensing Learning', 
                               'Factory Primary Structure', 
                               'Factory Drums',
                               'Factory Other', 
                               'Factory Be',
                               'Factory BeO',
                               'Onsite Learning',
                               'Non-nuclear off-the-shelf']:
            return params[f"{multiplier_type} Cost Multiplier"]
        else:
            return np.nan
    
    df['Multiplier'] = df['FOAK to NOAK Multiplier Type'].apply(get_multiplier)
    foak_col = get_estimated_cost_column(df, 'F')
    noak_column = foak_col.replace("FOAK", "NOAK")
    df[noak_column] = df['Multiplier'] * df[foak_col]
    return df


def reorder_dataframe(df):
    first_columns = ['Account', 'Account Title']
    other_columns = [col for col in df.columns if col not in first_columns]
    new_column_order = first_columns + other_columns
    df = df[new_column_order]
    return df


def bottom_up_cost_estimate(cost_database_filename, params):
    # Validate tax credit params early — before any simulation or cost calculation runs.
    # This catches the case where a user accidentally defines both ITC and PTC,
    # which are mutually exclusive under the IRA.
    validate_tax_credit_params(params)

    escalated_cost = escalate_cost_database(cost_database_filename, params['Escalation Year'], params)
    escalated_cost_cleaned = remove_irrelevant_account(escalated_cost, params)
    reactor_operation(params)

    COA_list = []
    for i in range(params['Number of Samples']):
        if (i + 1) % 100 == 0:
            print(f"\n\nSample # {i+1}")

        scaled_cost = scale_cost(escalated_cost_cleaned, params)
        scaled_cost = scale_redundant_BOP_and_primary_loop(scaled_cost, params)
        NOAK_COA = FOAK_to_NOAK(scaled_cost, params)

        updated_cost = update_high_level_costs(scaled_cost, 'base', i)
        updated_cost_with_indirect_cost = calculate_accounts_31_32_75_82_cost(updated_cost, params)
        cost_with_decommissioning = calculate_decommissioning_cost(updated_cost_with_indirect_cost, params)
        updated_accounts_10_40 = update_high_level_costs(cost_with_decommissioning, 'other', i)
        high_Level_capital_cost = calculate_high_level_capital_costs(updated_accounts_10_40, params)
        
        updated_accounts_10_60 = update_high_level_costs(high_Level_capital_cost, 'finance', i)
        TCI = calculate_TCI(updated_accounts_10_60, params)
        updated_accounts_70_80 = update_high_level_costs(TCI, 'annual', i)
        Final_COA = energy_cost_levelized(params, updated_accounts_70_80)
        FOAK_column = get_estimated_cost_column(Final_COA, 'F')
        NOAK_column = get_estimated_cost_column(Final_COA, 'N')
        Final_COA = Final_COA[['Account', 'Account Title', FOAK_column, NOAK_column]]
        
        COA_list.append(Final_COA)
    
    concatenated_df = pd.concat(COA_list)
    numeric_columns = concatenated_df.select_dtypes(include='number').columns
    mean_df = concatenated_df[numeric_columns].groupby(concatenated_df.index).mean()

    if params["Number of Samples"] > 1:
        std_df = concatenated_df[numeric_columns].groupby(concatenated_df.index).std()
    else:
        std_df = concatenated_df[numeric_columns].groupby(concatenated_df.index).std(ddof=0)

    mean_df[FOAK_column.replace('Cost', 'Cost std')] = std_df[FOAK_column]
    mean_df[NOAK_column.replace('Cost', 'Cost std')] = std_df[NOAK_column]

    non_numeric_columns = concatenated_df.select_dtypes(exclude='number').groupby(concatenated_df.index).first()
    result_df = mean_df.join(non_numeric_columns)
    reordered_df = reorder_dataframe(result_df)
    return reordered_df


def parametric_studies(cost_database_filename, params, tracked_params_list, output_csv_filename):
    detatiled_cost_table = bottom_up_cost_estimate(cost_database_filename, params)
    tracked_costs = create_cost_dictionary(detatiled_cost_table, params, tracked_params_list)
    
    file_exists = os.path.isfile(output_csv_filename)
    
    with open(output_csv_filename, 'a', newline='') as csvfile:
        fieldnames = tracked_costs.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists or os.stat(output_csv_filename).st_size == 0:
            writer.writeheader()
        
        writer.writerow(tracked_costs)
        print(f"Results are being saved on {output_csv_filename}")


def detailed_bottom_up_cost_estimate(cost_database_filename, params, output_filename):
    detatiled_cost_table = bottom_up_cost_estimate(cost_database_filename, params)
    pretty_df = transform_dataframe(detatiled_cost_table)
    with pd.ExcelWriter(output_filename) as writer:
        pretty_df.to_excel(writer, sheet_name="cost estimate", index=False)
        save_params_to_excel_file(writer, params)
        
    print(f"\n\nThe cost estimate and all the paramters are saved at {output_filename}\n\n")
    return detatiled_cost_table