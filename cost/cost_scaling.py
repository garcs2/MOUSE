# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pandas as pd
from cost.sampling import sampler

def _transport_escalation(params):
    # Fixed transport inputs share the Dollar Year of their database row; scale_cost stores
    # that row's escalation multiplier here so they escalate exactly like the unit cost.
    return params.get('_Transport Escalation Multiplier', 1.0)


def _transport_nuclear_fraction(params, unit_cost):
    # Class 7 carrier premium as a fraction of the line-haul rate. The older $/mile input
    # ('Transport Nuclear Premium') is still honoured if it is the only one supplied.
    if 'Transport Nuclear Premium Fraction' in params:
        return params['Transport Nuclear Premium Fraction']
    if 'Transport Nuclear Premium' in params and unit_cost > 0:
        print("\033[93m'Transport Nuclear Premium' ($/mile) is deprecated; "
              "use 'Transport Nuclear Premium Fraction' instead.\033[0m")
        return params['Transport Nuclear Premium'] / unit_cost
    return 0.30


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
        # Per-shipment model (all fixed inputs in the account's Dollar Year, 2024 $):
        #   C = [r_base(1 + f_nuc) + r_team + N_esc*r_esc]*D + k_FOAK*(C_pkg + C_permit + C_mob + C_sec) + C_state
        # r_base is the database unit cost. k_FOAK calibrates the bottom-up fixed costs of the
        # fuel-bearing truck to the MARVEL HALEU shipment estimate; set it to 1 for the pure
        # bottom-up (NOAK-like) value. Non-fuel trucks are heavy specialized freight: r_base*D + C_mob.
        D = scaling_variable_value
        mode = params.get('Deployment Mode', 'Mobile')   # Mobile | Semi-Mobile | Stationary
        esc = _transport_escalation(params)
        k_foak = params.get('Transport FOAK Calibration Factor', 2.92)
        f_nuc = _transport_nuclear_fraction(params, unit_cost)

        # Fresh-fuel security category from enrichment AND U-235 mass (10 CFR 73.2)
        u235_kg = params.get('Mass U235', 0) / 1000.0
        if params['Enrichment'] >= 0.10:
            category = 'II' if u235_kg >= 10 else 'III'
        else:
            category = 'III' if u235_kg >= 10 else 'Exempt'
        params['Transport Security Category'] = category
        sec_cost = {'II':  params.get('Cat II Security Cost', 2500),
                    'III': params.get('Cat III Security Cost', 2500),
                    'Exempt': 0}[category]

        # Fresh-fuel package amortization (MARVEL container recertification / shipments).
        fuel_pkg = (params.get('Mobile Fuel Packaging Cost', 10900) if mode == 'Mobile'
                    else params.get('Bare Fuel Packaging Cost', 10900))

        line_rate = (unit_cost * (1 + f_nuc)
                     + params.get('Outbound Team Driver Premium', 0.0)
                     + params.get('Outbound Escort Count', 0) * params.get('Transport Escort Rate', 1.84))
        fuel_ship = (line_rate * pow(D, exponent)
                     + esc * (k_foak * (fuel_pkg + sec_cost
                                        + params.get('Transport Permit Cost', 3700)
                                        + params.get('Transport Mobilization Cost', 14000))
                              + params.get('Transport State Fees', 0)))
        nonfuel_ship = (unit_cost * pow(D, exponent)
                        + esc * (params.get('Non-Fuel Mobilization Cost', 14000)
                                 + params.get('Reactor Packaging Cost', 0)))

        n_nonfuel = {'Mobile': 0, 'Semi-Mobile': 1, 'Stationary': 2}[mode]
        params['Transport Outbound Fuel Truck Cost'] = fuel_ship
        params['Transport Outbound Non-Fuel Truck Cost'] = nonfuel_ship
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
            # Same model as account 27 for the irradiated-fuel-bearing truck. k_FOAK applies to
            # permits, crane and security, not to the cask (own source) or state fees (statutory).
            # Activated non-fuel loads pay the Class 7 premium and crane cost but no Type B cask
            # or 10 CFR 73.37 security.
            D = scaling_variable_value
            mode = params.get('Deployment Mode', 'Mobile')
            esc = _transport_escalation(params)
            k_foak = params.get('Transport FOAK Calibration Factor', 2.92)
            f_nuc = _transport_nuclear_fraction(params, unit_cost)

            line_rate = (unit_cost * (1 + f_nuc)
                         + params.get('Return Team Driver Premium', 1.01)
                         + params.get('Return Escort Count', 2) * params.get('Transport Escort Rate', 1.84))
            cask_uses = max(params.get('Return Cask Uses', 50), 1)
            fuel_ship = (line_rate * pow(D, exponent)
                         + esc * (k_foak * (params.get('Return Permit Routing Cost', 6400)
                                            + params.get('Return Mobilization Cost', 22900)
                                            + params.get('Return Security Fixed Cost', 37800))
                                  + params.get('Return Cask Capital Cost', 6_400_000) / cask_uses
                                  + params.get('Return State Fees', 3500)))
            nonfuel_ship = (unit_cost * (1 + f_nuc) * pow(D, exponent)
                            + esc * (params.get('Non-Fuel Mobilization Cost', 14000)
                                     + params.get('Return Reactor Packaging Cost', 0)))

            n_nonfuel = {'Mobile': 0, 'Semi-Mobile': 1, 'Stationary': 2}[mode]
            per_trip = fuel_ship + n_nonfuel * nonfuel_ship
            params['Transport Return Fuel Truck Cost'] = fuel_ship
            params['Transport Return Non-Fuel Truck Cost'] = nonfuel_ship
            params['Transport Return Cost Per Trip'] = per_trip
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
            
            if (row['Account'] == 34 and params.get('Transport Distance', 0) > 0
                    and params.get('Exclude Account 34 With Transport', True)):
                # Account 34 is MARVEL's lump-sum shipping (mostly fuel supply-chain shipments).
                # When the per-shipment transport model (accounts 27/722) is on, drop it to
                # avoid double counting.
                estimated_cost = 0

            elif row['Standard Cost Equation?'] == 'standard' :

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
                    params['_Transport Escalation Multiplier'] = (
                        row['Adjusted Unit Cost ($)'] / row['Unit Cost']
                        if pd.notna(row['Unit Cost']) and row['Unit Cost'] > 0 else 1.0)
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
