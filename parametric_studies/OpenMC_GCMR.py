import numpy as np
import watts  # Simulation workflows for one or multiple codes
from core_design.openmc_template_GCMR import *
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

class OpenMC_GCMR:
    """
    Class for GCMR optimization
    
    :param working_dir: (str) woorking directory for optimzation
    """
    
    def __init__(self,working_dir = Path(".")):
        nominal_parameters = {    
    # Settings
    'plotting': "Y",  # "Y" or "N": Yes or No
    'cross_sections_xml_location': '/projects/MRP_MOUSE/openmc_data/endfb-viii.0-hdf5/cross_sections.xml', # on INL HPC
    'simplified_chain_thermal_xml': '/projects/MRP_MOUSE/openmc_data/simplified_thermal_chain11.xml',       # on INL HPC
    'cost database':'/cost/',
    # Materials
    'reactor type': "GCMR",  # LTMR or GCMR
    'TRISO Fueled': "Yes",
    'Fuel': 'UN',
    'Enrichment': 0.1975,  # The enrichment is a fraction. It has to be between 0 and 1
    'UO2 atom fraction': 0.7,  # Mixing UO2 and UC by atom fraction
    'Reflector': 'Graphite',
    'Matrix Material': 'Graphite', # matrix material is a background material  within the compact fuel element between the TRISO particles
    'Moderator': 'Graphite', # The moderator is outside this compact fuel region 
    'Moderator Booster': 'ZrH',
    'Coolant': 'Helium',
    'Common Temperature': 850,  # Kelvins
    'Control Drum Absorber': 'B4C_enriched',  # The absorber material in the control drums
    'Control Drum Reflector': 'Graphite',  # The reflector material in the control drums
    'HX Material': 'SS316', 

    # Geometry
    # fuel pin details
    'Fuel Pin Materials': ['UN', 'buffer_graphite', 'PyC', 'SiC', 'PyC'],
    'Fuel Pin Radii': [0.025, 0.035, 0.039, 0.0425, 0.047],  # cm
    'Compact Fuel Radius': 0.6225,  # cm # The radius of the area that is occupied by the TRISO particles (fuel compact/ fuel element)
    'Packing Fraction': 0.3,
    
    # Coolant channel and booster dimensions
    'Coolant Channel Radius': 0.35,  # cm
    'Moderator Booster Radius': 0.55, # cm
    'Lattice Pitch'  : 2.25,
    'Assembly Rings' : 6,
    'Core Rings' : 5,
    # Overall system
    'Power MWt': 15,  # MWt
    
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
        assert mode in ['single_obj','multi_obj'], '---error: mode should either be single_obj or multi_obj not {}'.format(mode)
        self.mode=mode

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
        run_openmc(build_openmc_model_GCMR, heat_flux_monitor, self.params)#,directory=directory)
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
        self.params['Assembly FTF'] = self.params['Lattice Pitch']*(self.params['Assembly Rings']-1)*np.sqrt(3)
        if 'Reflector Thickness' not in self.params.keys():
            self.params['Reflector Thickness'] = 27.393 # cm # radial reflector
        self.params['Axial Reflector Thickness'] = self.params['Reflector Thickness'] # cm
        self.params['Core Radius'] = self.params['Assembly FTF']*self.params['Core Rings'] +  self.params['Reflector Thickness']
        self.params['Active Height'] = 250 
    
    def update_control_drum(self):
        """
        Update control drums

        """
        self.update_params({
            'Drum Radius' : 9, # cm   
            'Drum Absorber Thickness': 1, # cm
            'Drum Height': self.params['Active Height'] + 2*self.params['Axial Reflector Thickness'],
        })
        calculate_drums_volumes_and_masses(self.params)
        calculate_reflector_mass_GCMR(self.params)          
        calculate_moderator_mass_GCMR(self.params) 

    def update_system(self):
        """
        Update power system parameters
        
        """
        # Overall system
        self.update_params({
        #'Power MWt': 15,  # MWt
        'Thermal Efficiency': 0.4,
        'Heat Flux Criteria': 0.9,  # MW/m^2 (This one needs to be reviewed)
        'Burnup Steps': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0,
                                     30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0]  # MWd_per_Kg
        })
        self.params['Power MWe'] = self.params['Power MWt'] * self.params['Thermal Efficiency'] 
        self.params['Heat Flux'] =  calculate_heat_flux_TRISO(self.params) # MW/m^2
    
    # Parameters independant from user-inpout changes
    def update_bop(self):
        """
        Update Balance-of-Plants parameters
        
        """
        self.update_params({
        'Primary Loop Purification': True,
        'Secondary HX Mass': 0,
        'Compressor Pressure Ratio': 4,
        'Compressor Isentropic Efficiency': 0.8,
        'Primary Loop Count': 2, # Number of Primary Coolant Loops present in plant
        'Primary Loop per loop load fraction': 0.5, # based on assuming that each Primary Loop Handles the total load evenly (1/2)
        'Primary Loop Inlet Temperature': 300 + 273.15, # K
        'Primary Loop Outlet Temperature': 550 + 273.15, # K
        'Secondary Loop Inlet Temperature': 290 + 273.15, # K
        'Secondary Loop Outlet Temperature': 500 + 273.15, # K,
        'Primary Loop Pressure Drop': 50e3, # Pa. Assumption based on Enrique's estimate
        })
        self.params['Primary HX Mass'] = calculate_heat_exchanger_mass(self.params)  # Kg
        # calculate coolant mass flow rate
        mass_flow_rate(self.params)
        compressor_power(self.params)

        # Update BoP Parameters
        self.params.update({
            'BoP Count': 2, # Number of BoP present in plant
            'BoP per loop load fraction': 0.5, # based on assuming that each BoP Handles the total load evenly (1/2)
            })
        self.params['BoP Power kWe'] = 1000 * self.params['Power MWe'] * self.params['BoP per loop load fraction']

        # Integrated Heat Transfer Vessel
        # Assumed no Integrated Heat Transfer Vessel in this design
        self.params.update({
            'Integrated Heat Transfer Vessel Thickness': 0, # cm
            'Integrated Heat Transfer Vessel Material': 'SA508',
        })
        GCMR_integrated_heat_transfer_vessel(self.params)


    def update_shielding(self):
        """
        Update radial shielding parameters

        """
        self.update_params({
    'In Vessel Shield Thickness': 0,  # cm (no shield in vessel for GCMR)
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
        # Assume to be the Core Barrel
        'Vessel Radius': self.params['Core Radius'] +  self.params['In Vessel Shield Thickness'],
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

        vessels_specs(self.params)  # calculate the volumes and masses of the vessels
        calculate_shielding_masses(self.params)  # calculate the masses of the shieldings


    def update_operation(self):
        """
        Update operational parameters

        """
        self.update_params({
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
        ## Calculated based on 10 tanks w/ 291 cuft ea @ 2400psi, 30°C
        ## Density=24.417 kg/m3, Volume=8.2402 m3 (standard tank size?)
        ## Refill Frequency: 1 /yr if purified, 6 /yr if not purified
        self.params['Onsite Coolant Inventory'] = 10 * 24.417 * 8.2402 # kg
        self.params['Replacement Coolant Inventory'] = self.params['Onsite Coolant Inventory'] / 4
        self.params['Annual Coolant Supply Frequency'] = 1 if self.params['Primary Loop Purification'] else 6

        # A75: Annualized Capital Expenditures
        ## Input for replacement of large capital equipments. Replacements are made during refueling cycles
        ## Components to be replaced:
        ## If the period is 0, it is assumed to never be replaced throughout Levelization period
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

    # Objective function
    def fitness(self,parameters,openmc_run=True):
        """
        Compute the objective function based on user-input parameters
        ---

        :params parameter: (dict) parameters of the design to run
        :params openmc_run: (bool) inform if openmc must be ran or not (useful for analysis where the design parameters are related to cost only)
        
        Returns
        0:
        """
        
        if openmc_run:
            self.run_openmc(parameters=parameters)
            # Write the design parameter to a text file
            write_dict(self.params, self.working_dir / Path('sample_{}'.format(self.folderindex)) / Path("design_parameters.txt"))
        # Save parameters
        # ---
        if self.params['Number of Samples'] == 1:
            detailed_bottom_up_cost_estimate('{}Cost_Database.xlsx'.format(self.params['cost database']), self.params, "./{}/output_GCMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))
            # Post process the objectives
            # ---
            data = pd.read_excel("./{}/output_GCMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex),sheet_name='cost estimate')
            lcoe_foak,lcoe_noak = data.loc[data['Account Title'] == 'Levelized Cost Of Energy ($/MWh)'].values[0,-2:]
            ann_cost_foak,ann_cost_noak = data.loc[data['Account Title'] == 'Annualized Cost'].values[0,-2:]
            TCI_cost_foak,TCI_cost_noak = data.loc[data['Account Title'] == 'Total Capital Investment'].values[0,-2:]

            data = pd.read_excel("./{}/output_GCMR_{}.xlsx".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex),sheet_name='Parameters')
            heat_flux = data.loc[data['Parameter'] == 'Heat Flux'].values[0,1]
            fuel_lifetime = data.loc[data['Parameter'] == 'Fuel Lifetime'].values[0,1]
        
        else:
        #    tracked_params_list =     ["Packing Factor", "Lattice Radius", "Number Of TRISO Particles Per Compact Fuel",
        #    "Total Number of TRISO Particles","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass"]

            tracked_params_list =     ["Keff","Keff Std","Packing Fraction", "Number Of TRISO Particles Per Compact Fuel",
            "Total Number of TRISO Particles","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass",
            'TRISO Fueled','Fuel','Enrichment','Lattice Pitch','Control Drum Reflector','Reflector','Power MWt','Assembly Rings','Reflector Thickness',
            'Operation Mode','Emergency Shutdowns Per Year','Moderator Booster','Moderator Booster Radius','Particles']

            parametric_studies('{}Cost_Database.xlsx'.format(self.params['cost database']),  self.params, tracked_params_list, "./{}/output_GCMR_{}.csv".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))

            # Post process the objectives
            # ---
            data = pd.read_csv("./{}/output_GCMR_{}.csv".format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))#,sheet_name='cost estimate')
            lcoe_foak,lcoe_noak = data['LCOE_FOAK Estimated Cost'].values[0],data['LCOE_NOAK Estimated Cost'].values[0]
            ann_cost_foak,ann_cost_noak = data['AC_FOAK Estimated Cost'].values[0],data['AC_NOAK Estimated Cost'].values[0]
            TCI_cost_foak,TCI_cost_noak = data['TCI_FOAK Estimated Cost'].values[0],data['TCI_NOAK Estimated Cost'].values[0]
            heat_flux = data['Heat Flux'].values[0]
            fuel_lifetime = data['Fuel Lifetime'].values[0]
        
        objectives={'LCOE_FOAK':lcoe_foak,'LCOE_NOAK':lcoe_noak,'ANN_COST_FOAK':ann_cost_foak,'ANN_COST_NOAK':ann_cost_noak,'TCI_FOAK':TCI_cost_foak,'TCI_COST_NOAK':TCI_cost_noak,'HEAT_FLUX':heat_flux,'FUEL_LIFETIME':fuel_lifetime}
        write_dict(objectives,'./{}/objectives_{}.csv'.format(self.working_dir / Path('sample_{}'.format(self.folderindex)),self.folderindex))

        return 0
