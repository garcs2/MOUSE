import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/garcsamu/OpenMC/MOUSE')

from reactor_engineering_evaluation.pin_temperatures import (
    LTMRPinGeometry, LTMRThermalProperties,
    LTMRCoolantConditions, compute_pin_temperatures_abc,
)

# --- Geometry and properties ---
geom  = LTMRPinGeometry()
props = LTMRThermalProperties()
props.set_fuel_k_from_name('UZrH_alloy')

# --- Coolant conditions ---
cond = LTMRCoolantConditions(
    T_inlet     = 703.15,   # K
    DeltaT_coolant = 90,   # K
    P_total_W   = 20.0e6,   # W
    N_fuel_pins = 300,
)

q_nom = 20.0e6 / (300 * geom.L_active)

# --- Mock per_step_data with PF = 1.0 ---
mock = {
    1: pd.DataFrame([{
        'Region_ID':      0,
        'Peaking_Factor': 1.0,
        'Step':           1,
    }])
}

summary, per_step = compute_pin_temperatures_abc(
    per_step_data = mock,
    geom          = geom,
    props         = props,
    cond          = cond,
    q_prime_nom   = q_nom,
)