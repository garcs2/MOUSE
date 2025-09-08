import numpy as np
import watts  # Simulation workflows for one or multiple codes
from core_design.openmc_template_LTMR import *
from core_design.pins_arrangement import LTMR_pins_arrangement
from core_design.utils import *
from core_design.drums import *
from reactor_engineering_evaluation.fuel_calcs import fuel_calculations
from reactor_engineering_evaluation.BOP import *
from reactor_engineering_evaluation.vessels_calcs import *
from reactor_engineering_evaluation.tools import *
from cost.cost_estimation import detailed_bottom_up_cost_estimate,parametric_studies

import warnings
warnings.filterwarnings("ignore")

import time
time_start = time.time()

from pathlib import Path
import json
import string,random
import pandas as pd

# Misc functions
#Dean Price
def write_dict(data_dict, file_path):
    with open(file_path, 'w') as file:
        file.write('Parameter,Value\n')
        for key, value in data_dict.items():
            file.write(f'{key},{value}\n')

def rstring(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(length))
    return random_string

class OpenMC_LTMR:
    """
    Class for LTMR optimization
    
    :param working_dir: (str) woorking directory for optimzation
    """
    
    def __init__(self,working_dir = Path(".")):
        nominal_parameters = { 
            # Settings
            'plotting': "Y",  # "Y" or "N": Yes or No
            #'cross_sections_xml_location': '/projects/MRP_MOUSE/openmc_data/endfb-viii.0-hdf5/cross_sections.xml', # on INL HPC
            #'simplified_chain_thermal_xml': '/projects/MRP_MOUSE/openmc_data/simplified_thermal_chain11.xml',       # on INL HPC
            #'cost database':'/cost/',
            'cross_sections_xml_location': '/home/seurpr/Desktop/LDRD/LDRD_microreactor/Examples/libraries/endfb-viii.0-hdf5/cross_sections.xml',
            'simplified_chain_thermal_xml': '/home/seurpr/Desktop/LDRD/LDRD_microreactor/Examples/libraries/chain_casl_pwr.xml',
            'cost database':'/home/seurpr/Desktop/LDRD/LDRD_microreactor/MOUSE_dev/mouse/cost/',
            
            # Materials
            'reactor type': "LTMR", # LTMR or GCMR
            'TRISO Fueled': "No",
            'Fuel': 'TRIGA_fuel',
            'Enrichment': 0.1975,  # Fraction between 0 and 1
            "H_Zr_ratio": 1.6,  # Proportion of hydrogen to zirconium atoms
            'U_met_wo': 0.3,  # Weight ratio of Uranium to total fuel weight (less than 1)
            'Coolant': 'NaK',
            'Reflector': 'Graphite',
            'Moderator': 'ZrH',
            'Control Drum Absorber': 'B4C_enriched',
            'Control Drum Reflector': 'Graphite',
            'Common Temperature': 600,  # Kelvins
            'HX Material': 'SS316',

            # Geometry
            # Fuel pins detail
            'Fuel Pin Materials': ['Zr', None, 'TRIGA_fuel', None, 'SS304'],
            'Fuel Pin Radii': [0.28575, 0.3175, 1.5113, 1.5367, 1.5875],  # cm
            'Moderator Pin Materials': ['ZrH', 'SS304'],  
            'Moderator Pin Inner Radius': 1.5367,  # cm
            'Moderator Pin Radii': [1.5367, 1.5875],  # [params['Moderator Pin Inner Radius'], params['Fuel Pin Radii'][-1]]
            "Pin Gap Distance": 0.1,  # cm
            'Pins Arrangement': LTMR_pins_arrangement,
            'Number of Rings per Assembly': 12, # the number of rings can be 12 or lower as long as the heat flux criteria is not violated
            'Reflector Thickness': 14,  # cm

            # Overall System
            'Power MWt': 20,  # MWt
            'Thermal Efficiency': 0.31,
            'Heat Flux Criteria': 0.9,  # MW/m^2
            'Burnup Steps': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 140.0],  # MWd_per_Kg

            # If run multiple samples for uncertainty estimation in cost model
            'Number of Samples':500,
            # Number of particles per MC simulation
            'Particles':1000 
        }
        # Instantiate the parameter of the problem
        self.params = watts.Parameters()
        self.update_params(nominal_parameters)
        # Create running files
        self.folderindex = rstring(10)
        self.working_dir = working_dir
        if self.working_dir is not None:
            sdir = Path(self.working_dir / f"sample_{self.folderindex}")
            sdir.mkdir(parents = True, exist_ok = True)

    def update_nonnucisland(self):
        """
        Initialize all parameters that are independant from user-input
        May change for future scoping analysis

        """
        self.update_bop()
        self.update_shielding()
        self.update_vessels()
        self.update_operation()
        self.update_economic_parameters()

        return
    
    def update_params(self,updates):
        """
        Update parameters of the reactor model

        :param updates: (dict) new input for reactor
        
        return None
        """
        self.params.update(updates)

    def run_openmc(self,parameters):
        """
        Run openMC and update params
        
        """
        #TODO add multi-threading
        #TODO add directory
        if parameters is not None:
            self.update_params(parameters)
        self.update_new_reactor()
        
        heat_flux_monitor = monitor_heat_flux(self.params)
        run_openmc(build_openmc_model_LTMR, heat_flux_monitor, self.params)
        fuel_calculations(self.params)  # calculate the fuel mass and SWU
        self.update_nonnucisland()
        

    def update_new_reactor(self):
        """
        Update parameters that change depending on user-input parameter

        """

        self.update_core_geometry()
        self.update_control_drum()
        self.update_system()

    # Functions to update mics parameters related to geometry, materials, and operation 
    # ---
    def update_core_geometry(self):
        """
        Update core-level geometries

        """

        self.params['Lattice Radius'] = calculate_lattice_radius(self.params)
        self.params['Active Height']  =   78.4  # Or it is 2 * params['Lattice Radius']
        self.params['Axial Reflector Thickness'] = self.params['Reflector Thickness'] # cm
        self.params['Fuel Pin Count'] = calculate_pins_in_assembly(self.params, "FUEL")
        self.params['Moderator Pin Count'] =  calculate_pins_in_assembly(self.params, "MODERATOR")
        self.params['Moderator Mass'] = calculate_moderator_mass(self.params)
        self.params['Core Radius'] = self.params['Lattice Radius'] + self.params['Reflector Thickness'] 

    def update_control_drum(self):
        """
        Update control drums

        """
        self.update_params({
            'Drum Radius': 9.016, # or it is 0.23 * params['Lattice Radius'],  # cm
            'Drum Absorber Thickness': 1,  # cm
            'Drum Height': self.params['Active Height'] + 2*self.params['Axial Reflector Thickness']
        })

        calculate_drums_volumes_and_masses(self.params)
        calculate_reflector_mass_LTMR(self.params)

    def update_system(self):
        """
        Update power system parameters
        
        """

        self.params['Power MWe'] = self.params['Power MWt'] * self.params['Thermal Efficiency']
        self.params['Heat Flux'] =  calculate_heat_flux(self.params)

    def update_bop(self):
        """
        Update Balance-of-Plants parameters
        
        """

        self.update_params({
            'Secondary HX Mass': 0,
            'Primary Pump': 'Yes',
            'Secondary Pump': 'No',
            'Pump Isentropic Efficiency': 0.8,
            'Primary Loop Inlet Temperature': 430 + 273.15, # K
            'Primary Loop Outlet Temperature': 520 + 273.15, # K
            'Secondary Loop Inlet Temperature': 395 + 273.15, # K
            'Secondary Loop Outlet Temperature': 495 + 273.15, # K,
            'BoP Count': 2, # Number of BoP present in plant
            'BoP per loop load fraction': 0.5, # based on assuming that each BoP Handles the total load evenly (1/2)
        })
        self.params['Primary HX Mass'] = calculate_heat_exchanger_mass(self.params)  # Kg
        self.params['BoP Power kWe'] = 1000 * self.params['Power MWe'] * self.params['BoP per loop load fraction']
        # calculate coolant mass flow rate
        mass_flow_rate(self.params)
        calculate_primary_pump_mechanical_power(self.params) 
    
    def update_shielding(self):
        """
        Update radial shielding parameters

        """
        self.update_params({
            'In Vessel Shield Thickness': 10.16,  # cm
            'In Vessel Shield Inner Radius': self.params['Core Radius'],
            'In Vessel Shield Material': 'B4C_natural',
            'Out Of Vessel Shield Thickness': 39.37,  # cm
            'Out Of Vessel Shield Material': 'WEP',
            'Out Of Vessel Shield Effective Density Factor': 0.5 # The out of vessel shield is not fully made of the out of vessel material (e.g. WEP) so we use an effective density factor
        })

        self.params['In Vessel Shield Outer Radius'] =  self.params['Core Radius'] + self.params['In Vessel Shield Thickness']

    def update_vessels(self):
        """
        Update vessel parameters

        """
        self.update_params({
            'Vessel Radius': self.params['Core Radius'] +  self.params['In Vessel Shield Thickness'],
            'Vessel Thickness': 1,  # cm
            'Vessel Lower Plenum Height': 42.848 - 40,  # cm, based on Reflecting Barrel~RPV Liner (-Reflector Thickness, which is currently missing in CAD),  # cm
            'Vessel Upper Plenum Height': 47.152,  # cm
            'Vessel Upper Gas Gap': 0, 
            'Vessel Bottom Depth': 32.129,
            'Vessel Material': 'stainless_steel',
            'Gap Between Vessel And Guard Vessel': 2,  # cm
            'Guard Vessel Thickness': 0.5,  # cm
            'Guard Vessel Material': 'stainless_steel',
            'Gap Between Guard Vessel And Cooling Vessel': 5,  # cm
            'Cooling Vessel Thickness': 0.5,  # cm
            'Cooling Vessel Material': 'stainless_steel',
            'Gap Between Cooling Vessel And Intake Vessel': 3,  # cm
            'Intake Vessel Thickness': 0.5,  # cm
            'Intake Vessel Material': 'stainless_steel'
        })

        vessels_specs(self.params)  # calculate the volumes and masses of the vessels
        calculate_shielding_masses(self.params)  # calculate the masses of the shieldings

    def update_operation(self):
        """
        Update operational parameters

        """
        self.update_params({
            'Operation Mode': "Autonomous",
            'Number of Operators': 2,
            'Levelization Period': 60,  # years
            'Refueling Period': 7,
            'Emergency Shutdowns Per Year': 0.2,
            'Startup Duration after Refueling': 2,
            'Startup Duration after Emergency Shutdown': 14,
            'Reactors Monitored Per Operator': 10,
            'Security Staff Per Shift': 1
        })
        ## Calculated based on 1 tanks
        ## Density of NaK=855  kg/m3, Volume=8.2402 m3 (standard tank size)
        self.params['Onsite Coolant Inventory'] = 1 * 855 * 8.2402 # kg
        self.params['Replacement Coolant Inventory'] = 0 # assume that NaK does not need to be replaced.
        # params['Annual Coolant Supply Frequency']  # LTMR should not require frequent refilling

        total_refueling_period = self.params['Fuel Lifetime'] + self.params['Refueling Period'] + self.params['Startup Duration after Refueling'] # days
        total_refueling_period_yr = total_refueling_period/365
        self.params['A75: Vessel Replacement Period (cycles)']        = np.floor(10/total_refueling_period_yr) # change each 10 years similar to the ATR
        self.params['A75: Core Barrel Replacement Period (cycles)']   = np.floor(10/total_refueling_period_yr)
        self.params['A75: Reflector Replacement Period (cycles)']     = np.floor(10/total_refueling_period_yr)
        self.params['A75: Drum Replacement Period (cycles)']          = np.floor(10/total_refueling_period_yr)
        self.params['Mainenance to Direct Cost Ratio']                = 0.015
        # A78: Annualized Decommisioning Cost
        self.params['A78: CAPEX to Decommissioning Cost Ratio'] = 0.15
    
    def update_economic_parameters(self):
        """
        Update economic parameters
        
        """
        self.update_params({
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
            'NOAK Unit Number': 100
        })

    # Objective function
    def fitness(self,parameters,openmc_run=True):
        """
        Compute the objective function based on user-input parameters
        
        """
        
        if openmc_run:
            self.run_openmc(parameters=parameters)
            # Write the design parameter to a text file
            write_dict(self.params, self.working_dir / Path('sample_{}'.format(self.folderindex)) / Path("design_parameters.txt"))
        # Save parameters
        # ---
        if self.params['Number of Samples'] == 1:
            detailed_bottom_up_cost_estimate('{}Cost_Database.xlsx'.format(self.params['cost database']), self.params, "./{}/output_LTMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))
            # Post process the objectives
            # ---
            data = pd.read_excel("./{}/output_LTMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex),sheet_name='cost estimate')
            lcoe_foak,lcoe_noak = data.loc[data['Account Title'] == 'Levelized Cost Of Energy ($/MWh)'].values[0,-2:]
            ann_cost_foak,ann_cost_noak = data.loc[data['Account Title'] == 'Annualized Cost'].values[0,-2:]
            TCI_cost_foak,TCI_cost_noak = data.loc[data['Account Title'] == 'Total Capital Investment'].values[0,-2:]

            data = pd.read_excel("./{}/output_LTMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex),sheet_name='Parameters')
            heat_flux = data.loc[data['Parameter'] == 'Heat Flux'].values[0,1]
            fuel_lifetime = data.loc[data['Parameter'] == 'Fuel Lifetime'].values[0,1]
        
        else:
        #    tracked_params_list =     ["Packing Factor", "Lattice Radius", "Number Of TRISO Particles Per Compact Fuel",
        #    "Total Number of TRISO Particles","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass"]

            tracked_params_list =     ["U_met_wo","Moderator Pin Inner Radius","Moderator","Keff","Keff Std","Particles","Packing Factor", "Number Of TRISO Particles Per Compact Fuel",
            "Total Number of TRISO Particles","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass",
            'TRISO Fueled','Fuel','Enrichment','Lattice Pitch','Reflector','Power MWt','Number of Rings per Assembly','Reflector Thickness',
            'Operation Mode','Emergency Shutdowns Per Year']

            parametric_studies('{}Cost_Database.xlsx'.format(self.params['cost database']),  self.params, tracked_params_list, "./{}/output_LTMR_{}.csv".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))

            # Post process the objectives
            # ---
            data = pd.read_csv("./{}/output_LTMR_{}.csv".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))#,sheet_name='cost estimate')
            lcoe_foak,lcoe_noak = data['LCOE_FOAK Estimated Cost'].values[0],data['LCOE_NOAK Estimated Cost'].values[0]
            ann_cost_foak,ann_cost_noak = data['AC_FOAK Estimated Cost'].values[0],data['AC_NOAK Estimated Cost'].values[0]
            TCI_cost_foak,TCI_cost_noak = data['TCI_FOAK Estimated Cost'].values[0],data['TCI_NOAK Estimated Cost'].values[0]
            heat_flux = data['Heat Flux'].values[0]
            fuel_lifetime = data['Fuel Lifetime'].values[0]
        
        objectives={'LCOE_FOAK':lcoe_foak,'LCOE_NOAK':lcoe_noak,'ANN_COST_FOAK':ann_cost_foak,'ANN_COST_NOAK':ann_cost_noak,'TCI_FOAK':TCI_cost_foak,'TCI_COST_NOAK':TCI_cost_noak,'HEAT_FLUX':heat_flux,'FUEL_LIFETIME':fuel_lifetime}
        write_dict(objectives,'./{}/objectives_{}.csv'.format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))
        return 0