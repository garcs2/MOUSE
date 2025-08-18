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
    'Reflector' : 'Be',
    'Moderator': 'monolith_graphite',
    'Secondary Coolant': 'Helium', # gap between the fuel and the moderator OR between heatpipe and moderator is filled with the secondary coolant (e.g. Helium)
    'Control Drum Absorber': 'B4C_natural',  # The absorber material in the control drums
    'Control Drum Reflector': 'Be',
    'Coolant': 'heatpipe', # The reactor is cooled by heatpipes which are modeled as a mixture of SS-316 and sodium
    'Common Temperature': 1000,  #K
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
params['Active Height'] = 160 #cm  
params['Reflector Thickness'] = 16 #cm
params['Axial Reflector Thickness'] = 20 #cm
params['Core Radius'] = params['Assembly FTF']* params['Number of Rings per Core'] +  params['Reflector Thickness']     
params['hexagonal Core Edge Length'] = (params['Assembly FTF'] * (params['Number of Rings per Core']-1)) + (params['Assembly FTF']/2) + 6.6  # The edge lenght is 86.6 as in the originial input so 6.6 is added based on this value
params['Fuel Pin Count per Assembly'] = calculate_number_fuel_elements_hpmr(params['Number of Rings per Assembly'])
params['Fuel Assemblies Count'] =  (3 * params['Number of Rings per Core']**2) - (3 * params['Number of Rings per Core'])
params['Fuel Pin Count'] = params['Fuel Assemblies Count'] * params['Fuel Pin Count per Assembly']
# **************************************************************************************************************************
#                                           Sec. 3: Control Drums
# ************************************************************************************************************************** 

update_params({
    'Drum Radius': 15, 
    'Drum Absorber Thickness': 1,  # cm
    'Drum Height': params['Active Height']
})

calculate_drums_volumes_and_masses(params)
calculate_reflector_mass_HPMR(params)

# **************************************************************************************************************************
#                                           Sec. 4: Overall System
# ************************************************************************************************************************** 
update_params({
    'Power MWt': 5, 
    'Thermal Efficiency': 0.36,
    'Heat Flux Criteria': 0.9,  # MW/m^2 
    'Time Steps': [t * 86400 for t in [0.01,   0.99,   3,   6,  20,  70, 100, 165, 365, 365, 365, 365, 365, 365, 365.00] ] # seconds
})
params['Power MWe'] = params['Power MWt'] * params['Thermal Efficiency']
params['Heat Flux'] =  calculate_heat_flux(params)

heat_flux_monitor = monitor_heat_flux(params)
run_openmc(build_openmc_model_HPMR, heat_flux_monitor, params)
fuel_calculations(params)  # calculate the fuel mass and SWU

params.show_summary(show_metadata=True, sort_by='time')