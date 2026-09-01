# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import numpy as np 

DEFAULT_TRANSPORT_MASS_LIMIT_KG = 22200.0
# Mass groups (label, params key). Edit these lists to change what rides where.
#   CORE     : reactor structure that always travels as the "reactor" load
#   SHIELD   : in/out-of-vessel shielding + ISO container
#   FUEL     : the fuel elements (ship separately in Semi-Mobile / Stationary)

_CORE_KEYS = [
    ('Moderator',           'Moderator Mass'),
    ('Moderator Booster',   'Moderator Booster Mass'),
    ('Control Drums',       'Control Drums Mass'),
    ('Radial Reflector',    'Radial Reflector Mass'),
    ('Axial Reflector',     'Axial Reflector Mass'),
    ('Vessels (all)',       'Total Vessels Mass'),
]
_SHIELD_KEYS = [
    ('In-Vessel Shield',      'In Vessel Shield Mass'),
    ('Out-of-Vessel Shield',  'Out Of Vessel Shield Mass'),
    ('ISO Container',         'Isocontainer Mass'),
]
_FUEL_KEYS = [
    ('Fuel Element',          'Fuel Element Mass'),
]

def _group_mass(params, keys, missing):
    total = 0.0
    for label, key in keys:
        value = params.get(key)
        if value is None:
            missing.append((label, key))
        else:
            total += float(value)
    return total

def ellipsoid_shell(a, b, c):
    return 4*np.pi*np.power(((a*b)**1.6 + (a*c)**1.6 + (b*c)**1.6)/3, 1/1.6)

def circle_area(r):
    return (np.pi) * r **2


def materials_densities(material):
    material_densities = {
    "carbon_steel": 7.82,
    "stainless_steel": 8.0,  # Approximate density of stainless steel
    "SS316": 8.0,            # Approximate density of SS316
    "SS304": 7.93,           # Approximate density of SS304
    "low_alloy_steel": 7.85, # Approximate density of SA508 Gr3 Cls 1
    "SA508": 7.85,           # Approximate density of SA508 Gr3 Cls 1
    "B4C_enriched": 2.52,    # Approximate density of boron carbide
    "B4C_natural": 2.52,     # Approximate density of boron carbide
    "WEP": 1.1,              # WEP density (water extended polymer)
    "WB": 15.43,
    "W2B": 16.75,
    "WB4": 8.23,
    "WC": 15.32,
    }
    return material_densities[material] # in gram/cm^3

def material_specific_heat(material):
    material_cp = {
        "Helium": 5193,  # J/(kg·K)
        "NaK": 982.      # J/(kg·K)
    }
    return material_cp[material]  # J/(kg·K)

def cylinder_annulus_mass(outer_radius , inner_radius,height, material ):

    volume = 3.14* (outer_radius**2 - inner_radius**2) * height
    mass = volume* materials_densities(material)/1000  # kg
    return mass # in kg

def concentric_rectangular_prism_mass(thickness, material, params):
    # Standard ISO container interior dimensions — now read from params
    # (Isocontainer Interior Height/Width/Length) so the mass calculation
    # and the transport geometry (create_shielding_annuli) always agree,
    # rather than each having its own hardcoded copy of these numbers.
    height = params['Isocontainer Interior Height']
    width  = params['Isocontainer Interior Width']
    length = params['Isocontainer Interior Length']
 
    inner_volume = height * width * length
    outer_volume = (height + 2 * thickness) * (width + 2 * thickness) * (length + 2 * thickness)
    volume = outer_volume - inner_volume
    mass = volume * materials_densities(material) / 1000
    return mass  # in kg


def calculate_shielding_masses(params):
    params['In Vessel Shield Mass'] = cylinder_annulus_mass(params['In Vessel Shield Outer Radius'],\
    params['In Vessel Shield Inner Radius'], params['Vessel Height'], params['In Vessel Shield Material'] )
    params['Outer Shield Outer Radius'] = params['Out Of Vessel Shield Thickness'] + params['Vessels Total Radius']
    params['Outer Shield Inner Radius'] = params['Outer Shield Outer Radius'] - params['Out Of Vessel Shield Thickness']

    outer_shield_mass = cylinder_annulus_mass(params['Outer Shield Outer Radius'], params['Outer Shield Inner Radius'],\
    params['Vessels Total Height'], params['Out Of Vessel Shield Material']) 
    params['Out Of Vessel Shield Mass'] = params['Out Of Vessel Shield Effective Density Factor'] * outer_shield_mass
    
    params['Isocontainer Mass'] = concentric_rectangular_prism_mass(
        params['Isocontainer Steel Thickness'],
        params['Isocontainer Steel Material'],
        params,
    ) if params.get('Mobile', False) else 0

def mass_flow_rate(params):
    loop_factor = 1
    thermal_power_MW = params['Power MWt']
    if 'Primary Loop per loop load fraction' in params.keys():
        loop_factor = params['Primary Loop per loop load fraction']
        thermal_power_MW = params['Power MWt'] * loop_factor
        
    deltaT =  params['Primary Loop Outlet Temperature'] - params['Primary Loop Inlet Temperature']
    if params['reactor type'] == "HPMR":
        coolant = params['Secondary Coolant']
    else:    
        coolant = params['Coolant']
    coolant_specific_heat = material_specific_heat(coolant)
    m_dot = 1e6 * thermal_power_MW/ (deltaT * coolant_specific_heat)
    params['Coolant Mass Flow Rate']  = m_dot / loop_factor # For Reactor Mass Flow Rate
    params['Primary Loop Mass Flow Rate'] = m_dot # For individual Primary Loop Mass Flow Rate
    
def compressor_power(params):
    # Estimates the required compressor power based on a simplified
    # model using pressure drop and compressor isentropic efficiency

    rho_he = 3.3297  # kg/m3. TODO: Consider importing CoolProp to estimate density based on cold-leg temperature and pressure
    power = params['Primary Loop Pressure Drop']*params['Primary Loop Mass Flow Rate']/params['Compressor Isentropic Efficiency']/rho_he
    params['Primary Loop Compressor Power'] = power # W
    return

def compressor_wheel_diameter(params):
    # Estimates the approximate compressor size based on its specific
    # diameter, matched to the MIGHTR horizontal HTGR design.
    # Ref for specific diameter:
    #  https://www.dropbox.com/scl/fi/fnqdg2hyi6y4ozu9p7nyu/final-report-str-mech-ARDP-redacted-V3.pdf?rlkey=h97dii28tvf0bxtffo8q62tn5&st=zsls1bs2&dl=0
    ref_specific_diameter = 3.6  # dimensionless
    rho_He = 3.330  # kg/m3 for He at 4 MPa, 300 °C. TODO: use a He density correlation or CoolProp to estimate density based on cold-leg temperature and pressure
    Vdot_gcmr = params['Primary Loop Mass Flow Rate'] / rho_He  # m3/s — volumetric flow rate
    dP = params['Primary Loop Pressure Drop']
    diameter = ref_specific_diameter/1.054 / (dP/rho_He)**0.25 * np.sqrt(Vdot_gcmr) # m
    return diameter

def GCMR_integrated_heat_transfer_vessel(params):
    # Calculates the required parameters for the
    # GCMR Integrated Heat Transfer Vessel that houses:
    #   circulator, PCHE, piping, valves, insulation

    contingency = 0.3  # accounts for the volume/mass of valves, fittings, and connections
    PCHE_volume = (params['Primary HX Mass'] / (materials_densities(params['HX Material'])*1e3) / 0.4)  # accounts for assumed 60% coolant channel void fraction
    compressor_volume = (compressor_wheel_diameter(params))**3  # approximated as a cube with side = wheel diameter. TODO: improve compressor sizing

    vessel_inner_volume = (1+contingency)*(PCHE_volume + compressor_volume)  # assumes a cube-like structure
    vessel_outer_volume = (vessel_inner_volume**(1/3)+ 1e-2*params['Integrated Heat Transfer Vessel Thickness'])**3  # m3
    vessel_volume = vessel_outer_volume - vessel_inner_volume
    vessel_density = materials_densities(params['Integrated Heat Transfer Vessel Material'])*1e3
    params['Integrated Heat Transfer Vessel Outer Volume'] = vessel_outer_volume
    params['Integrated Heat Transfer Vessel Mass'] = vessel_volume * vessel_density
    
    if params['Integrated Heat Transfer Vessel Thickness'] == 0:
        params['Integrated Heat Transfer Vessel Outer Volume'] = 0
        params['Integrated Heat Transfer Vessel Mass'] = 0

    # Rough estimate of the mass supported by the support structure:
    # Primary HX + Integrated Heat Transfer Vessel + Compressor + Valves/Fittings/Bolts/etc.
    # params['Integrated Heat Transfer System Mass'] = params['Primary HX Mass'] + (vessel_volume * vessel_density) + compressor_volume*8000



def evaluate_transport_mass(params, verbose=True):
    """
    Auto-determines the required Deployment Mode based on component mass limits,
    or falls back to the least-assembled mode if limits are exceeded.
    """
    limit = float(params.get('Transport Mass Limit (kg)', DEFAULT_TRANSPORT_MASS_LIMIT_KG))

    missing = []
    core   = _group_mass(params, _CORE_KEYS, missing)
    shield = _group_mass(params, _SHIELD_KEYS, missing)
    fuel   = _group_mass(params, _FUEL_KEYS, missing)

    # 1. Define load configurations in order of preference (most to least integrated)
    candidate_modes = {
        'Mobile': {
            'Full unit (core + shield + fuel)': core + shield + fuel
        },
        'Semi-Mobile': {
            'Reactor (core + shield, unfueled)': core + shield,
            'Fuel': fuel
        },
        'Stationary': {
            'Core': core,
            'Shielding': shield,
            'Fuel': fuel
        }
    }

    # 2. Automatically select the most integrated mode that satisfies the limit
    selected_mode = None
    loads = {}

    for mode, candidate_loads in candidate_modes.items():
        heaviest_in_candidate = max(candidate_loads.values()) if candidate_loads else 0.0
        if heaviest_in_candidate <= limit:
            selected_mode = mode
            loads = candidate_loads
            break

    # 3. Fallback: If no single mode stays under the limit, default to Stationary
    if selected_mode is None:
        selected_mode = 'Stationary'
        loads = candidate_modes['Stationary']

    heaviest = max(loads.values()) if loads else 0.0
    within = heaviest <= limit

    # Update parameters dict with determined mode and results
    params['Deployment Mode'] = selected_mode
    params['Transport Loads (kg)'] = loads
    params['Transport Heaviest Load (kg)'] = heaviest
    params['Transport Mass Limit (kg)'] = limit
    params['Within Transport Limit'] = within

    if verbose:
        _print_transport_mass_report(selected_mode, loads, missing, heaviest, limit, within)

    return within
 
 
def _print_transport_mass_report(mode, loads, missing, heaviest, limit, within):
    green, red, yellow, reset = '\033[92m', '\033[91m', '\033[93m', '\033[0m'
 
    print('\n' + '=' * 66)
    print(f' TRANSPORT MASS EVALUATION  -  Deployment Mode: {mode}')
    print('=' * 66)
    for name, m in loads.items():
        ok = m <= limit
        tag = f'{green}OK  {reset}' if ok else f'{red}OVER{reset}'
        print(f'  [{tag}] {name:<35s} {m:>11,.1f} kg ({m/1000:>6.3f} t)')
    print('-' * 66)
    print(f'  Heaviest load: {heaviest:>11,.1f} kg ({heaviest/1000:.3f} t)     '
          f'limit {limit:,.0f} kg ({limit/1000:.1f} t)')
 
    pct = 100.0 * heaviest / limit if limit else float('nan')
    if within:
        print(f'{green}  PASS - heaviest load is {limit - heaviest:,.0f} kg '
              f'({(limit-heaviest)/1000:.3f} t) under the limit ({pct:.1f}% of limit){reset}')
    else:
        print(f'{red}  FAIL - heaviest load is {heaviest - limit:,.0f} kg '
              f'({(heaviest-limit)/1000:.3f} t) OVER the limit ({pct:.1f}% of limit).{reset}')
        print(f'{red}         A less-consolidated Deployment Mode (Semi-Mobile / Stationary) '
              f'splits the load and may bring it within limit.{reset}')
 
    if missing:
        print(f'{yellow}  Note: not found in params (counted as 0):{reset}')
        for label, key in missing:
            print(f'{yellow}        - {label} -> params[{key!r}]{reset}')
    print('=' * 66 + '\n')