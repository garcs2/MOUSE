# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""
This script performs a bottom-up cost estimate for a heat pipe Microreactor.
OpenMC is used for core design calculations, and other Balance of Plant components are estimated.
Users can modify parameters in the "params" dictionary below.
"""
import numpy as np
import watts  # Simulation workflows for one or multiple codes
from core_design.openmc_template_HPMR import *
from core_design.utils import *
from core_design.drums import *
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
from reactor_engineering_evaluation.BOP import *
from reactor_engineering_evaluation.vessels_calcs import *
from reactor_engineering_evaluation.tools import *
from cost.cost_estimation import detailed_bottom_up_cost_estimate

import warnings
warnings.filterwarnings("ignore")

import time
time_start = time.time()

params = watts.Parameters()

def update_params(updates):
    params.update(updates)

# **************************************************************************************************************************
#                                                Sec. 0: Settings
# **************************************************************************************************************************

update_params({
    'plotting': "Y",  # "Y" or "N": Yes or No
    'cross_sections_xml_location': '/projects/MRP_MOUSE/openmc_data/endfb-viii.0-hdf5/cross_sections.xml', # on INL HPC
    'simplified_chain_thermal_xml': '/projects/MRP_MOUSE/openmc_data/simplified_thermal_chain11.xml'       # on INL HPC
})

# **************************************************************************************************************************
#                                                Sec. 1: Materials
# **************************************************************************************************************************
# These params are based on this report: https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_99962.pdf
update_params({
    'reactor type': "HPMR",
    'TRISO Fueled': "Yes",
    
    # The fuel TRISO particles dispersed in a graphite matrix with a packing fraction of 36%. 
    # TRISO particles have a UO2 kernels and dimensions typical for fuel used in the AGR-2 campaign (https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_50872.pdf)
    # TRISO particles are homogenization with the surrounding graphite matrix. 
    'Fuel': 'homog_TRISO',     
    'Enrichment': 0.19985,
    'Reflector' : 'Graphite',
    'Moderator': 'monolith_graphite',
    'Secondary Coolant': 'Helium', # gap between the fuel and the moderator OR between heatpipe and moderator is filled with the secondary coolant (e.g. Helium)
    'Control Drum Absorber': 'B4C_natural',  # The absorber material in the control drums
    'Control Drum Reflector': 'Graphite',
    'Cooling Device': 'heatpipe', # The reactor is cooled by heatpipes which are modeled as a mixture of SS-316 and sodium
    'Common Temperature': 1000,  #K
    'HX Material': 'SS316'
})

# **************************************************************************************************************************
#                                           Sec. 2: Geometry: Fuel Pins, Moderator Pins, Coolant, Hexagonal Lattice
# **************************************************************************************************************************

update_params({
    'Fuel Pin Materials': ['homog_TRISO', 'Helium'],
    'Fuel Pin Radii': [1.00, 1.05], #cm
    'Heat Pipe Materials': ['heatpipe', 'Helium'],
    'Heat Pipe Radii': [1.10, 1.15],
    'Number of Rings per Assembly': 6, # number of pins (fuel or heatpipe) along the side of the hex assembly
    'Number of Rings per Core': 3, # number of assemblies alog the side of the core
    'Lattice Pitch': 3.4, # center-to-center distance between adjacent fuel/heatpipe pins
    
})
params['Assembly FTF'] = (params['Lattice Pitch'] * (params['Number of Rings per Assembly'] - 1) +  1.4 * params['Fuel Pin Radii'][-1]) * np.sqrt(3)
params['hexagonal Core Edge Length'] = (params['Assembly FTF'] * (params['Number of Rings per Core']-1)) + (params['Assembly FTF']/2) + 6.6  # The edge lenght is 86.6 as in the originial input so 6.6 is added based on this value
params['Reflector Thickness'] = 50 # it was 37 #cm
params['Core Radius'] = 0.5*np.sqrt(3)*params['hexagonal Core Edge Length'] +  params['Reflector Thickness']
params['Active Height'] =  2 * params['Core Radius'] # it was 160 #cm  
params['Axial Reflector Thickness'] = params['Reflector Thickness']
params['Fuel Pin Count per Assembly'] = calculate_number_fuel_elements_hpmr(params['Number of Rings per Assembly'])
params['Fuel Assemblies Count'] =  (3 * params['Number of Rings per Core']**2) - (3 * params['Number of Rings per Core'])
params['Fuel Pin Count'] = params['Fuel Assemblies Count'] * params['Fuel Pin Count per Assembly']
# calculate the number of heatpipes
number_of_heatpipes_hmpr(params)
# **************************************************************************************************************************
#                                           Sec. 3: Control Drums
# ************************************************************************************************************************** 

update_params({
    'Drum Radius': 0.4 * params['Reflector Thickness'], 
    'Drum Absorber Thickness': 1,  # cm
    'Drum Height': params['Active Height']
})

calculate_drums_volumes_and_masses(params)
calculate_reflector_and_moderator_mass_HPMR(params)
# **************************************************************************************************************************
#                                           Sec. 4: Overall System
# ************************************************************************************************************************** 
update_params({
    'Power MWt': 7, 
    'Thermal Efficiency': 0.36,
    'Heat Flux Criteria': 0.9,  # MW/m^2 
    'Time Steps': [t * 86400 for t in [0.01,   0.99,   3,   6,  20,  70, 100, 165, 365, 365, 365, 365, 365, 365, 365.00] ] # seconds
})
params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
params['Heat Flux'] =  calculate_heat_flux(params)

# **************************************************************************************************************************
#                                           Sec. 5: Running OpenMC
# ************************************************************************************************************************** 
heat_flux_monitor = monitor_heat_flux(params)
run_openmc(build_openmc_model_HPMR, heat_flux_monitor, params)
fuel_calculations(params)  # calculate the fuel mass and SWU

# **************************************************************************************************************************
#                                        Sec. 6: Primary Loop + Balance of Plant
# ************************************************************************************************************************** 
params.update({
    'Primary Loop Purification': True,
    'Secondary HX Mass': 0, # assume only one HX
    'Primary Loop Count': 2, # Number of Primary Coolant Loops present in plant

    # temperatures are based on https://www.sciencedirect.com/science/article/pii/S1359431125014711
    'Primary Loop Inlet Temperature': 900 + 273.15, # K
    'Primary Loop Outlet Temperature': 650 + 273.15, # K
    'Secondary Loop Inlet Temperature': 300 + 273.15, # K
    'Secondary Loop Outlet Temperature': 630 + 273.15, # K,
   })
params['Primary HX Mass'] = calculate_heat_exchanger_mass(params)  # Kg

  
# Update BoP Parameters
params.update({
    'BoP Count': 2, # Number of BoP present in plant
    'BoP per loop load fraction': 0.5, # based on assuming that each BoP Handles the total load evenly (1/2)
    })
params['BoP Power kWe'] = 1000 * params['Power MWe'] * params['BoP per loop load fraction']

# # **************************************************************************************************************************
# #                                           Sec. 7 : Shielding
# # ************************************************************************************************************************** 
update_params({
    'In Vessel Shield Thickness': 0,  # cm (no shield in vessel for HPMR)
    'In Vessel Shield Inner Radius': params['Core Radius'],
    'In Vessel Shield Material': 'B4C_natural',
    'Out Of Vessel Shield Thickness': 39.37,  # cm
    'Out Of Vessel Shield Material': 'WEP',
    'Out Of Vessel Shield Effective Density Factor': 0.5 # The out of vessel shield is not fully made of the out of vessel material (e.g. WEP) so we use an effective density factor
})
params['In Vessel Shield Outer Radius'] =  params['Core Radius'] + params['In Vessel Shield Thickness']

# **************************************************************************************************************************
#                                           Sec. 8 : Vessels Calculations
# ************************************************************************************************************************** 
update_params({
    # Assume to be the Core Barrel
    'Vessel Radius': params['Core Radius'] +  params['In Vessel Shield Thickness'],
    'Vessel Thickness': 1,  # cm
    'Vessel Lower Plenum Height': 42.848 - 40,  # cm, based on Reflecting Barrel~RPV Liner (-Reflector Thickness, which is currently missing in CAD)
    'Vessel Upper Plenum Height': 47.152,       # cm, based on Reflector Ends~RPV Liner distance
    'Vessel Upper Gas Gap': 0,                  # cm, assumed non-existed for GCMRv1
    'Vessel Bottom Depth': 32.129,              # cm, bot/top head (ellipsoid): 32.129 cm (not exact match with CAD, estimated to match RPV Height)
    'Vessel Material': 'stainless_steel',
    # Assumed no guard vessel
    'Gap Between Vessel And Guard Vessel': 0,  
    'Guard Vessel Thickness': 0,  # cm
    'Guard Vessel Material': 'low_alloy_steel', 
    'Gap Between Guard Vessel And Cooling Vessel': 5,  # cm
    'Cooling Vessel Thickness': 0.5,  # cm
    'Cooling Vessel Material': 'stainless_steel',
    'Gap Between Cooling Vessel And Intake Vessel': 4,  # cm
    'Intake Vessel Thickness': 0.5,  # cm
    'Intake Vessel Material': 'stainless_steel'
})

vessels_specs(params)  # calculate the volumes and masses of the vessels
calculate_shielding_masses(params)  # calculate the masses of the shieldings

# # **************************************************************************************************************************
# #                                           Sec. 9 : Operation
# # **************************************************************************************************************************
update_params({
    'Operation Mode': "Autonomous", # "Non-Autonomous" or "Autonomous"
    'Number of Operators': 2,
    'Levelization Period': 60,  # years
    'Refueling Period': 7,
    'Emergency Shutdowns Per Year': 0.2,
    'Startup Duration after Refueling': 2,
    'Startup Duration after Emergency Shutdown': 14,
    'Reactors Monitored Per Operator': 10,
    'Security Staff Per Shift': 1
})

# A721: Coolant Refill
## 20 Tanks total are on-site. 
## Assuming ~50% are used for fresh coolant, 50% are used for dirty
## Calculated based on 1 tanks w/ 291 cuft ea @ 2400psi, 30°C
## Density=24.417 kg/m3, Volume=8.2402 m3 (standard tank size?)
## Refill Frequency: 1 /yr if purified, 6 /yr if not purified
params['Onsite Coolant Inventory'] = 1 * 24.417 * 8.2402 # kg
params['Replacement Coolant Inventory'] = params['Onsite Coolant Inventory'] / 4
params['Annual Coolant Supply Frequency'] = 1 if params['Primary Loop Purification'] else 6


# A75: Annualized Capital Expenditures
## Input for replacement of large capital equipments. Replacements are made during refueling cycles
## Components to be replaced:
## If the period is 0, it is assumed to never be replaced throughout Levelization period
total_refueling_period = params['Fuel Lifetime'] + params['Refueling Period'] + params['Startup Duration after Refueling'] # days
total_refueling_period_yr = total_refueling_period/365
params['A75: Vessel Replacement Period (cycles)']        = np.floor(10/total_refueling_period_yr) # change each 10 years similar to the ATR
params['A75: Core Barrel Replacement Period (cycles)']   = np.floor(10/total_refueling_period_yr)
params['A75: Reflector Replacement Period (cycles)']     = np.floor(10/total_refueling_period_yr)
params['A75: Drum Replacement Period (cycles)']          = np.floor(10/total_refueling_period_yr)
params['Mainenance to Direct Cost Ratio']                = 0.015

# A78: Annualized Decommisioning Cost
params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15

# **************************************************************************************************************************
#                                           Sec. 10 : Economic Parameters
# **************************************************************************************************************************
update_params({
    # A conservative estimate for the land area 
    # Ref: McDowell, B., and D. Goodman. "Advanced Nuclear Reactor Plant Parameter Envelope and
    #Guidance." National Reactor Innovation Center (NRIC), NRIC-21-ENG-0001 (2021). 
    'Land Area': 18,  # acres
    'Escalation Year': 2024,
    'Excavation Volume': 412.605,  # m^3
    'Reactor Building Slab Roof Volume': (9750*6502.4*1500)/1e9,  # m^3
    'Reactor Building Basement Volume': (9750*6502.4*1500)/1e9,  # m^3
    'Reactor Building Exterior Walls Volume': ((2*9750*3500*1500)+(3502.4*3500*(1500+750)))/1e9,  # m^3
    'Reactor Building Superstructure Area': ((2*3500*3500)+(2*7500*3500))/1e6, # m^2
    
    # Connected to the Reactor Building (contains steel liner)
    'Integrated Heat Exchanger Building Slab Roof Volume': 0,  # m^3
    'Integrated Heat Exchanger Building Basement Volume': 0,  # m^3
    'Integrated Heat Exchanger Building Exterior Walls Volume': 0,  # m^3
    'Integrated Heat Exchanger Building Superstructure Area': 0, # m^2
    
    # Assumed to be High 40' CONEX Container with 20 cm wall thickness (including conex wall)
    'Turbine Building Slab Roof Volume': (12192*2438*200)/1e9,  # m^3
    'Turbine Building Basement Volume': (12192*2438*200)/1e9,  # m^3
    'Turbine Building Exterior Walls Volume': ((12192*2496*200)+(2038*2496*200))*2/1e9,  # m^3
    
    # Assumed to be High 40' CONEX Container with 20 cm wall thickness (including conex wall)
    'Control Building Slab Roof Volume': (12192*2438*200)/1e9,  # m^3
    'Control Building Basement Volume': (12192*2438*200)/1e9,  # m^3
    'Control Building Exterior Walls Volume': ((12192*2496*200)+(2038*2496*200))*2/1e9,  # m^3
    
    # Manipulator Building
    'Manipulator Building Slab Roof Volume': (4876.8*2438.4*400)/1e9, # m^3
    'Manipulator Building Basement Volume': (4876.8*2438.4*1500)/1e9, # m^3
    'Manipulator Building Exterior Walls Volume': ((4876.8*4445*400)+(2038.4*4445*400*2))/1e9, # m^3

    'Refueling Building Slab Roof Volume': 0,  # m^3
    'Refueling Building Basement Volume': 0,  # m^3
    'Refueling Building Exterior Walls Volume': 0,  # m^3
    
    'Spent Fuel Building Slab Roof Volume': 0,  # m^3
    'Spent Fuel Building Basement Volume': 0,  # m^3
    'Spent Fuel Building Exterior Walls Volume': 0,  # m^3
    
    'Emergency Building Slab Roof Volume': 0,  # m^3
    'Emergency Building Basement Volume': 0,  # m^3
    'Emergency Building Exterior Walls Volume': 0,  # m^3
    
    # Building to host operational spares (CO2, He, filters, etc.)
    'Storage Building Slab Roof Volume': (8400*3500*400)/1e9, # m^3
    'Storage Building Basement Volume': (8400*3500*400)/1e9, # m^3
    'Storage Building Exterior Walls Volume': ((8400*2700*400)+(3100*2700*400*2))/1e9, # m^3
    
    'Radwaste Building Slab Roof Volume': 0,  # m^3
    'Radwaste Building Basement Volume': 0,  # m^3
    'Radwaste Building Exterior Walls Volume': 0,  # m^3,
    
    'Interest Rate': 0.07,
    'Construction Duration': 12,  # months
    'Debt To Equity Ratio': 0.5,
    'Annual Return': 0.0475,  # Annual return on decommissioning costs
    'NOAK Unit Number': 100,
})
# **************************************************************************************************************************
#                                           Sec. 11: Post Processing
# **************************************************************************************************************************
params['Number of Samples'] = 100 # Accounting for cost uncertainties
# Estimate costs using the cost database file and save the output to an Excel file
estimate = detailed_bottom_up_cost_estimate('cost/Cost_Database.xlsx', params, "examples/output_HPMR.xlsx")
elapsed_time = (time.time() - time_start) / 60  # Calculate execution time
print('Execution time:', np.round(elapsed_time, 1), 'minutes')