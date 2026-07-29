# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pandas as pd
from cost.sampling import sampler

def non_standard_cost_scale(account, unit_cost, scaling_variable_value, exponent, params):
    # pumps
    if account == 222.11 or account == 222.12:
        cost_multiplier = (0.2 / (1 - params['Pump Isentropic Efficiency'])) + 1
        cost = cost_multiplier * unit_cost * pow(scaling_variable_value,exponent)
    
    # compressors
    elif account == 222.13:
        if 'Primary Loop Count' in params.keys():
            # Account for multiple primary loops and their individual rated load
            ## PR1: Updated cost correlation based on ANL/NSE-20/28 in place of the default,
            ## due to inherent uncertainty in the compressor pressure ratio across GCMR designs
            cost_multiplier = (((params['Primary Loop Outlet Temperature'] - 273.15)/650)**1.29 *
                                (params['Primary Loop Compressor Power']/1e6/2.6)**0.74)
            cost = cost_multiplier * unit_cost
        else:
            # Old Correlation kept as backup
            cost_multiplier = (1 / (0.95 - params['Compressor Isentropic Efficiency'])) * params['Compressor Pressure Ratio'] * np.log(params['Compressor Pressure Ratio'])
            cost = cost_multiplier * unit_cost * pow(scaling_variable_value,exponent)
    
    elif account == 253:
        # Fuel enrichment / SWU pricing
        params['Enrichment Category'] = 'HALEU' if params['Enrichment'] >= 0.1 else 'LEU'
        if params['Enrichment'] >= 0.2:
            print("\033[91m ERROR: Enrichment is too high \033[0m")
            raise ValueError("Enrichment is too high")
        leu_swu_price   = unit_cost                               # DB 253 unit cost = $184.2/SWU (LEU base)
        haleu_swu_price = params.get('HALEU SWU Price', 1000.0)   # premium until market matures
        if params['Enrichment Category'] == 'LEU':
            effective_unit_cost = leu_swu_price
        else:
            matured = params.get('NOAK Unit Number', 0) >= params.get('HALEU NOAK Threshold', 100)
            effective_unit_cost = leu_swu_price if matured else haleu_swu_price
        cost = effective_unit_cost * pow(scaling_variable_value, exponent)
    elif account == 27:   # Reactor transport to site (fresh unit) — scenario- & category-aware
        D = scaling_variable_value
        mode = params.get('Deployment Mode', 'Mobile')   # Mobile | Semi-Mobile | Stationary

        # Fresh-fuel security category from enrichment AND U-235 mass (10 CFR 73.2)
        u235_kg = params.get('Mass U235', 0) / 1000.0
        if params['Enrichment'] >= 0.10:
            category = 'II' if u235_kg >= 10 else 'III'
        else:
            category = 'III' if u235_kg >= 10 else 'Exempt'
        params['Transport Security Category'] = category
        sec_cost = {'II':  params.get('Cat II Security Cost', 40000),
                    'III': params.get('Cat III Security Cost', 10000),
                    'Exempt': 0}[category]

        # Mode-dependent fuel packaging: Mobile ships the whole fueled+shielded unit
        # (heavier packaging); Semi-Mobile / Stationary ship bare fuel.
        fuel_pkg = (params.get('Mobile Fuel Packaging Cost', 25000) if mode == 'Mobile'
                    else params.get('Bare Fuel Packaging Cost', 10000))

        def ship(radioactive, n_esc):
            r_nuc = params.get('Transport Nuclear Premium', 1.5) if radioactive else 0.0
            return (unit_cost + r_nuc
                    + params.get('Transport Team Driver Premium', 0.5)
                    + n_esc * params.get('Transport Escort Rate', 2.5)) * pow(D, exponent)

        common_fixed = (params.get('Transport Permit Cost', 5000)
                        + params.get('Transport Mobilization Cost', 20000)
                        + params.get('Transport State Fees', 2000))

        fuel_ship = (ship(True, params.get('Outbound Escort Count', 1))
                     + fuel_pkg + sec_cost + common_fixed)
        nonfuel_ship = (ship(False, 0)
                        + params.get('Reactor Packaging Cost', 5000) + common_fixed)

        n_nonfuel = {'Mobile': 0, 'Semi-Mobile': 1, 'Stationary': 2}[mode]
        cost = fuel_ship + n_nonfuel * nonfuel_ship

    elif account == 711:
        cost_multiplier = params['FTEs Per Onsite Operator Per Year'] 
        cost = cost_multiplier * unit_cost * pow(scaling_variable_value,exponent)
    elif account == 712:
        cost_multiplier = params['FTEs Per Offsite Operator (24/7)']
        cost = cost_multiplier * unit_cost * pow(1 / scaling_variable_value, exponent) 
    elif account == 713:
        cost_multiplier = params['FTEs Per Security Staff (24/7)']
        cost = cost_multiplier * unit_cost * pow(scaling_variable_value,exponent)       
    elif account == 721:
        cost_multiplier = params['Annual Coolant Supply Frequency']
        cost = cost_multiplier * unit_cost * scaling_variable_value
    elif account == 722:  # Activated-unit return transport — scenario-aware, recurring
            D = scaling_variable_value
            mode = params.get('Deployment Mode', 'Mobile')

            # Every return load is activated (radioactive) -> Class 7 applies to all legs.
            def ship(n_esc):
                return (unit_cost + params.get('Transport Nuclear Premium', 2.0)
                        + params.get('Transport Team Driver Premium', 0.5)
                        + n_esc * params.get('Transport Escort Rate', 2.5)) * pow(D, exponent)

            common_fixed = (params.get('Return Permit Routing Cost', 35000)
                            + params.get('Return Mobilization Cost', 50000)
                            + params.get('Return State Fees', 5000))

            # Irradiated-fuel-bearing shipment: Type B cask + 10 CFR 73.37 armed escorts.
            fuel_ship = (ship(params.get('Return Escort Count', 2))
                        + params.get('Return Cask Amortized Cost', 75000)
                        + params.get('Return Security Fixed Cost', 60000) + common_fixed)
            nonfuel_ship = (ship(1)
                            + params.get('Return Reactor Packaging Cost', 15000) + common_fixed)

            n_nonfuel = {'Mobile': 0, 'Semi-Mobile': 1, 'Stationary': 2}[mode]
            per_trip = fuel_ship + n_nonfuel * nonfuel_ship
            cost = params.get('Annual Reactor Return Frequency', 0) * per_trip
    elif account == 81:
        cost_multiplier =  params['FTEs Per Operator Per Year Per Refueling'] 
        cost = cost_multiplier * unit_cost * pow(scaling_variable_value, exponent)
    return cost



def scale_redundant_BOP_and_primary_loop(df, params):
    # Scales special cases to handle redundant or multiple coolant/BoP loops
    escalation_year = params['Escalation Year']
    cost_col = f'FOAK Estimated Cost (${escalation_year })'

    if 'Primary Loop Count' in params.keys():
        df.loc[df['Account'].astype(str).str.startswith('222'), cost_col] *= params['Primary Loop Count']
    if 'BoP Count' in params.keys():
        # Balance of plant
        df.loc[df['Account'].astype(str).str.startswith('232'), cost_col] *= params['BoP Count']
        # Balance-of-plant building — assumed to be a high 40-ft CONEX container with 20 cm wall thickness (including the CONEX wall)
        df.loc[df['Account'].astype(str).str.startswith('213.1'), cost_col] *= params['BoP Count']
    if 'Primary Loop Purification' in params.keys():
        df.loc[df['Account'] == 226, cost_col] *= int(params['Primary Loop Purification'])

    return df



def scale_cost(initial_database, params):
    scaled_cost = initial_database[['Account', 'Level', 'Account Title', 'FOAK to NOAK Multiplier Type',\
                                    "Fixed Cost Low End", "Fixed Cost High End", "Fixed Cost Distribution",\
                                    "Unit Cost Low End", "Unit Cost High End", "Unit Cost Distribution",\
                                    "Exponent std",  "Exponent Max", "Exponent Min", "Exponent Distribution"]]
    
    escalation_year = params['Escalation Year']
    

    # Iterate through each row in the DataFrame
    for index, row in initial_database.iterrows():
        
        # Check if cost data are available (fixed or unit cost)
        if row['Fixed Cost ($)'] > 0 or	row['Unit Cost'] > 0:
            
            scaling_variable_value = params[row['Scaling Variable']] if pd.notna(row['Scaling Variable']) else 0
            
            # Calculate the 'Estimated Cost
            fixed_cost_0 = row['Adjusted Fixed Cost ($)'] 
            fixed_cost_lo = row['Adjusted Fixed Cost Low End ($)'] 
            fixed_cost_hi = row['Adjusted Fixed Cost High End ($)'] 
            fixed_cost_dist = row['Fixed Cost Distribution']

            if pd.notna(row['Fixed Cost ($)']):
                if params['Number of Samples'] > 1:
                    if fixed_cost_dist == 'Lognormal':
                        fixed_cost = sampler("Lognormal", low_cost=fixed_cost_lo, high_cost=fixed_cost_hi, class3_cost=fixed_cost_0)
                    elif fixed_cost_dist == 'Uniform': 
                        fixed_cost = sampler('Uniform', low=fixed_cost_lo, high=fixed_cost_hi)
                    else:
                        fixed_cost = fixed_cost_0
                else:
                    fixed_cost = fixed_cost_0
            else:
                fixed_cost = 0    
            
            unit_cost_0 = row['Adjusted Unit Cost ($)'] 
            unit_cost_lo = row['Adjusted Unit Cost Low End ($)'] 
            unit_cost_hi = row['Adjusted Unit Cost High End ($)'] 
            unit_cost_dist = row['Unit Cost Distribution']

            if pd.notna(row['Unit Cost']):
                if params['Number of Samples'] > 1:
                    if unit_cost_dist == 'Lognormal':
                        unit_cost = sampler("Lognormal", low_cost=unit_cost_lo, high_cost=unit_cost_hi, class3_cost=unit_cost_0)
                    elif unit_cost_dist == 'Uniform': 
                        unit_cost = sampler('Uniform', low=unit_cost_lo, high=unit_cost_hi)
                    else:
                        unit_cost = unit_cost_0
                else:
                    unit_cost =unit_cost_0

            else:
                unit_cost = 0  

            scaling_variable_ref_value  = row['Scaling Variable Ref Value']
            exponent_0 = row['Exponent']
            exponent_min = row['Exponent Min']
            exponent_max = row['Exponent Max']
            exponent_std = row['Exponent std']
            exponent_dist = row['Exponent Distribution']

            if pd.notna(row['Exponent']):
                if params['Number of Samples'] > 1:
                    if exponent_dist == 'Truncated Normal':
                        exponent = sampler("Truncated Normal", mean=exponent_0, std=exponent_std, lower_bound=exponent_min, upper_bound=exponent_max)
                    else:
                        exponent = exponent_0
                else:
                    exponent = exponent_0
            
            if row['Standard Cost Equation?'] == 'standard' :
                
                if pd.notna(row['Scaling Variable']) and scaling_variable_value == 0:
                    estimated_cost = 0
                
                else:     
                    # Check whether a reference value exists for the scaling variable; otherwise use the unit cost directly
                    if row['Scaling Variable Ref Value'] > 0:
                        estimated_cost = fixed_cost +\
                        unit_cost * pow(scaling_variable_value,exponent) /(pow(scaling_variable_ref_value,exponent-1))

                    else:
                        # Calculate the estimated cost
                        estimated_cost = fixed_cost + unit_cost * scaling_variable_value
            
            elif row['Standard Cost Equation?'] == 'nonstandard':
                if pd.notna(row['Scaling Variable']) and scaling_variable_value == 0:
                    estimated_cost = 0
                else:    
                    estimated_cost = non_standard_cost_scale(row['Account'],\
                    unit_cost, scaling_variable_value, exponent, params)


            # Assign the calculated value to the corresponding row in the DataFrame
            scaled_cost.at[index, f'FOAK Estimated Cost (${escalation_year })'] = estimated_cost
    return scaled_cost


def scale_central_facility_cost(initial_database, params):
    """
    Scale costs for central facility accounts.
    Similar to scale_cost() but includes Count Scaling Variable support.
    """
    scaled_cost = initial_database[['Account', 'Level', 'Account Title', 'FOAK to NOAK Multiplier Type',
                                    "Fixed Cost Low End", "Fixed Cost High End", "Fixed Cost Distribution",
                                    "Unit Cost Low End", "Unit Cost High End", "Unit Cost Distribution",
                                    "Exponent std", "Exponent Max", "Exponent Min", "Exponent Distribution"]]

    escalation_year = params['Escalation Year']
    params['Constant'] = 1

    for index, row in initial_database.iterrows():

        if row['Fixed Cost ($)'] > 0 or row['Unit Cost'] > 0:

            scaling_variable_value = params[row['Scaling Variable']] if pd.notna(row['Scaling Variable']) else 0
            count_variable_value = (params[row['Count Scaling Variable']] * row['Count per Variable']
                                    if pd.notna(row['Count Scaling Variable']) else 0)

            fixed_cost_0 = row['Adjusted Fixed Cost ($)']
            fixed_cost_lo = row['Adjusted Fixed Cost Low End ($)']
            fixed_cost_hi = row['Adjusted Fixed Cost High End ($)']
            fixed_cost_dist = row['Fixed Cost Distribution']

            if pd.notna(row['Fixed Cost ($)']):
                if params['Number of Samples'] > 1:
                    if fixed_cost_dist == 'Lognormal':
                        fixed_cost = sampler("Lognormal", low_cost=fixed_cost_lo, high_cost=fixed_cost_hi, class3_cost=fixed_cost_0)
                    elif fixed_cost_dist == 'Uniform':
                        fixed_cost = sampler('Uniform', low=fixed_cost_lo, high=fixed_cost_hi)
                    else:
                        fixed_cost = fixed_cost_0
                else:
                    fixed_cost = fixed_cost_0
            else:
                fixed_cost = 0

            unit_cost_0 = row['Adjusted Unit Cost ($)']
            unit_cost_lo = row['Adjusted Unit Cost Low End ($)']
            unit_cost_hi = row['Adjusted Unit Cost High End ($)']
            unit_cost_dist = row['Unit Cost Distribution']

            if pd.notna(row['Unit Cost']):
                if params['Number of Samples'] > 1:
                    if unit_cost_dist == 'Lognormal':
                        unit_cost = sampler("Lognormal", low_cost=unit_cost_lo, high_cost=unit_cost_hi, class3_cost=unit_cost_0)
                    elif unit_cost_dist == 'Uniform':
                        unit_cost = sampler('Uniform', low=unit_cost_lo, high=unit_cost_hi)
                    else:
                        unit_cost = unit_cost_0
                else:
                    unit_cost = unit_cost_0
            else:
                unit_cost = 0

            scaling_variable_ref_value = row['Scaling Variable Ref Value']
            exponent_0 = row['Exponent']
            exponent_min = row['Exponent Min']
            exponent_max = row['Exponent Max']
            exponent_std = row['Exponent std']
            exponent_dist = row['Exponent Distribution']

            if pd.notna(row['Exponent']):
                if params['Number of Samples'] > 1:
                    if exponent_dist == 'Truncated Normal':
                        exponent = sampler("Truncated Normal", mean=exponent_0, std=exponent_std, lower_bound=exponent_min, upper_bound=exponent_max)
                    else:
                        exponent = exponent_0
                else:
                    exponent = exponent_0

            if row['Standard Cost Equation?'] == 'standard':

                if pd.notna(row['Scaling Variable']) and scaling_variable_value == 0:
                    estimated_cost = 0
                else:
                    if row['Scaling Variable Ref Value'] > 0:
                        estimated_cost = (fixed_cost
                                          + unit_cost * pow(scaling_variable_value, exponent)
                                          / (pow(scaling_variable_ref_value, exponent - 1)))
                    else:
                        estimated_cost = fixed_cost + unit_cost * scaling_variable_value

                # Apply count scaling if specified
                if pd.notna(row['Count Scaling Variable']):
                    if count_variable_value == 0:
                        estimated_cost = 0
                    else:
                        estimated_cost = estimated_cost * count_variable_value

            elif row['Standard Cost Equation?'] == 'nonstandard':
                if pd.notna(row['Scaling Variable']) and scaling_variable_value == 0:
                    estimated_cost = 0
                else:
                    estimated_cost = non_standard_cost_scale(row['Account'],
                                                             unit_cost, scaling_variable_value, exponent, params)

            scaled_cost.at[index, f'FOAK Estimated Cost (${escalation_year })'] = estimated_cost
        else:
                # Explicitly assign 0 so these rows do not remain NaN
                # and cause parent account aggregation to fail
                scaled_cost.at[index, f'FOAK Estimated Cost (${escalation_year })'] = 0
    return scaled_cost
