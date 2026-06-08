"""
compute_pin_temperatures.py
============================
Derives axial temperature profiles and  temperature
differences from OpenMC radial peaking factors, using:
  - A cosine axial power shape whose peak amplitude equals the radial PF
    from compute_pin_peaking_factors() (MOUSE)
  - The closed-form radial heat conduction solution at each axial node

Axial model
-----------
For a pin with radial peaking factor PF_i, the local linear heat rate at
axial position z (measured from core bottom, 0 <= z <= L) is:

    q'(z) = q'_peak_i * cos(pi * (z - L/2) / L)

where q'_peak_i is chosen so that integrating q'(z) over the active length
recovers the total pin power implied by PF_i:

    integral_0^L q'(z) dz = PF_i * q'_nom * L
    => (2L/pi) * q'_peak_i = PF_i * q'_nom * L
    => q'_peak_i = PF_i * q'_nom * pi/2

The coolant temperature at axial position z follows from an energy balance
(integrating the heat addition from inlet to z):

    T_coolant(z) = T_inlet + q'_peak_i / (m_dot_pin * cp) * L/pi * (1 - cos(pi*z/L))

  At z=0 (bottom): T_coolant = T_inlet
  At z=L (top):    T_coolant = T_inlet + 2*q'_peak_i*L / (pi * m_dot_pin * cp)
                             = T_inlet + PF_i * q'_nom * L / (m_dot_pin * cp)
                               [same as lumped energy balance — consistent]

The radial solve (_solve_pin_temperatures) is then applied at each axial
node using the local q'(z) and T_coolant(z), giving T_fuel_max(z).

Key outputs per pin
-------------------
  DeltaT_fuel    = T_fuel_max(z=L) - T_fuel_max(z=0)   [top minus bottom]
  DeltaT_coolant = T_coolant(z=L)  - T_coolant(z=0)    [= T_outlet - T_inlet for this pin]

Pin cross-section (from watts_exec_LTMR_fuel_study.py)
-------------------------------------------------------
  Layer        Material    r_inner [cm]   r_outer [cm]
  ──────────── ─────────── ────────────── ──────────────
  Center rod   Zr          0              0.28575
  Inner bond   NaK/gap     0.28575        0.3175
  Fuel         UO2/UN/UC   0.3175         1.5113
  Outer gap    NaK/gap     1.5113         1.5367
  Cladding     SS304       1.5367         1.5875
  Coolant      NaK bulk    (outside)

Notation (matching classroom notes)
------------------------------------
   r_i  = outer radius of Zr rod = inner radius of fuel   [= geom.r_fi]
   r_f  = outer radius of fuel                            [= geom.r_fo]
   r_g  = outer radius of gap (= inner radius of clad)    [= geom.r_ci]
   r_c  = outer radius of cladding                        [= geom.r_co]
   q''' = volumetric heat generation rate in fuel [W/m³]
        = q'(z) / (pi * (r_f^2 - r_i^2))

Radial temperature distribution (at a given axial node with local q'(z))
--------------------------------------------------------------------------
  T_clad(r_c) = T_coolant(z) + q'''(r_f^2-r_i^2) / (2 h r_c)

  T_c(r) = -q'''(r_f^2-r_i^2)/(2 k_c) * ln(r/r_c) + T_clad(r_c)
           [valid for r_g <= r <= r_c]

  T_f(r) = q'''/(4 k_f) * (r_f^2 - r^2 + 2*r_i^2*ln(r/r_f))
           - q'''(r_f^2-r_i^2)/(2 k_g) * ln(r_f/r_g)
           - q'''(r_f^2-r_i^2)/(2 k_c) * ln(r_g/r_c)
           + T_clad(r_c)
           [valid for r_i <= r <= r_f; maximum at r = r_i]

Inner surface / Zr rod
-----------------------
Zero-flux BC at r = r_i (Zr rod generates no power) means no heat crosses
the inner fuel surface in steady state:
   T_Zr_surface = T_fi = T_fuel_max   (no drop across inner gap or Zr rod)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Pin geometry dataclass — defaults match LTMR from watts_exec_LTMR_fuel_study
# ============================================================================

@dataclass
class LTMRPinGeometry:
    """
    Radial pin geometry in metres.

    Defaults match the LTMR design from watts_exec_LTMR_fuel_study.py:
        'Fuel Pin Radii'  : [0.28575, 0.3175, 1.5113, 1.5367, 1.5875]  # cm
        'Fuel Pin Materials': ['Zr', None, Fuel, None, 'SS304']
        'Pin Gap Distance': 0.1  # cm

    The pin pitch and P/D ratio are derived from the cladding outer radius
    and the gap exactly as in the OpenMC template:
        pin_pitch = 2 * r_co + pin_gap
        P/D       = pin_pitch / (2 * r_co)
    """
    r_Zr_o:   float = 0.28575e-2   # outer radius of Zr center rod [m]
    r_fi:     float = 0.3175e-2    # inner radius of fuel annulus   [m]  (= r_i)
    r_fo:     float = 1.5113e-2    # outer radius of fuel annulus   [m]  (= r_f)
    r_ci:     float = 1.5367e-2    # inner radius of cladding       [m]  (= r_g)
    r_co:     float = 1.5875e-2    # outer radius of cladding       [m]  (= r_c)
    pin_gap:  float = 0.1e-2       # centre-to-surface gap between adjacent pins [m]
                                   # = params['Pin Gap Distance'] * 1e-2
    L_active: float = 0.784        # active (heated) height [m]

    @property
    def D_co(self) -> float:
        """Cladding outer diameter [m]."""
        return 2.0 * self.r_co

    @property
    def pin_pitch(self) -> float:
        """Centre-to-centre pin pitch [m].
        Matches openmc_template_LTMR: pin_pitch = 2*r_co + pin_gap"""
        return 2.0 * self.r_co + self.pin_gap

    @property
    def P_D(self) -> float:
        """Pin pitch-to-diameter ratio (dimensionless)."""
        return self.pin_pitch / self.D_co


@dataclass
class LTMRThermalProperties:
    """
    Thermal properties for each layer.

    Defaults:
      k_fuel      : UO2 at ~800 K ≈ 3.5 W/m·K  (conservative; UN ~ 20, UC ~ 20)
      k_clad      : SS304 at 700 K ≈ 19 W/m·K
      k_NaK_gap   : NaK thermal conductivity in the outer gap annulus ≈ 24 W/m·K.
                    Used in the hollow-cylinder gap conduction formula:
                      ΔT_gap = q'/(2π k_NaK_gap) · ln(r_g/r_f)
      k_NaK_coolant : bulk NaK thermal conductivity for the Kazimi-Carelli HTC
                      correlation. Typically the same value as k_NaK_gap.
    """
    k_fuel:         float = 3.5    # W/(m·K)  — UO2 default; override for UN/UC
    k_clad:         float = 19.0   # W/(m·K)  — SS304
    k_NaK_gap:      float = 24.0   # W/(m·K)  — NaK in the outer gap annulus
    # NOTE: no h_gap_inner — zero-flux BC at r_i means T_Zr = T_fi exactly.
    k_NaK_coolant:  float = 24.0   # W/(m·K)  — bulk NaK (for Nu correlation)

    FUEL_K_DEFAULTS: dict = field(default_factory=lambda: {
        "UO2":        3.5,
        "UC":         20.0,
        "UN":         20.0,
        "TRIGA_fuel": 14.0,   # ZrH-U fuel (General Atomics TRIGA data)
        "UZrH_alloy": 14.0,   # U-ZrHx alloy — same family as TRIGA_fuel.
                               # k ~ 14-18 W/(m·K) at operating temperatures.
                               # Conservative mid-range; override via
                               # props.k_fuel if composition-specific data available.
    })

    def set_fuel_k_from_name(self, fuel_name: str) -> None:
        """Automatically set k_fuel from fuel material name if known."""
        k = self.FUEL_K_DEFAULTS.get(fuel_name)
        if k is not None:
            self.k_fuel = k
        else:
            print(f"[ThermalProps] Unknown fuel '{fuel_name}'; "
                  f"keeping k_fuel = {self.k_fuel} W/(m·K)")


@dataclass
class LTMRCoolantConditions:
    """
    System-level coolant operating conditions.

    Specify T_inlet, DeltaT_coolant, and P_total_W as design inputs.
    The required mass-flow rate and outlet temperature are derived:

        m_dot_total = P_total_W / (cp_NaK * DeltaT_coolant)
        T_outlet    = T_inlet + DeltaT_coolant

    Attributes
    ----------
    T_inlet          : Coolant inlet temperature [K]
    DeltaT_coolant   : Design core-average coolant temperature rise [K]
                       (= T_outlet - T_inlet). This is the primary thermal-
                       hydraulic design target; m_dot is sized to achieve it.
    P_total_W        : Total reactor thermal power [W]
    N_fuel_pins      : Total number of fuel pins in the core
    cp_NaK           : NaK specific heat capacity [J/(kg·K)]
    rho_NaK          : NaK density at mean coolant temperature [kg/m³]
    h_NaK_override   : If set, bypasses the Nu correlation entirely [W/(m²·K)]
    """
    T_inlet:        float = 703.15   # K  (430 °C — LTMR default)
    DeltaT_coolant: float = 90.0     # K  (design target temperature rise)
    P_total_W:      float = 20.0e6   # W  (20 MWt — LTMR default)
    N_fuel_pins:    int   = 1
    cp_NaK:         float = 880.0    # J/(kg·K)
    rho_NaK:        float = 830.0    # kg/m³
    h_NaK_override: Optional[float] = None

    def __post_init__(self):
        if self.DeltaT_coolant <= 0:
            raise ValueError(
                f"DeltaT_coolant ({self.DeltaT_coolant} K) must be positive."
            )

    @property
    def T_outlet(self) -> float:
        """Coolant outlet temperature derived from inlet + ΔT [K]."""
        return self.T_inlet + self.DeltaT_coolant

    @property
    def m_dot_total(self) -> float:
        """Total mass-flow rate required to achieve DeltaT_coolant [kg/s]."""
        return self.P_total_W / (self.cp_NaK * self.DeltaT_coolant)

    @property
    def m_dot_per_pin(self) -> float:
        """Per-pin mass-flow rate [kg/s] under uniform-flow assumption."""
        return self.m_dot_total / self.N_fuel_pins


# ============================================================================
# NaK convective HTC — Kazimi-Carelli / modified Schad correlation
# ============================================================================

def _nusselt_liquid_metal_bundle(Pe: float, P_D: float) -> float:
    """
    Nu = 4.0 + 0.33*(P/D)^3.8 * (Pe/100)^0.86 + 0.16*(P/D)^5
    Valid ~1.1 <= P/D <= 1.4, Pe >= 10.
    """
    return 4.0 + 0.33 * (P_D**3.8) * ((Pe / 100.0)**0.86) + 0.16 * (P_D**5)


def compute_h_NaK(
    geom:  LTMRPinGeometry,
    props: LTMRThermalProperties,
    cond:  LTMRCoolantConditions,
) -> float:
    """
    Convective HTC [W/(m²·K)] at the cladding outer surface.
    Computed once per pin (flow conditions are axially uniform under the
    uniform-flow assumption).
    """
    if cond.h_NaK_override is not None:
        return cond.h_NaK_override

    P      = geom.pin_pitch
    A_flow = (np.sqrt(3) / 4.0) * P**2 - (np.pi / 2.0) * geom.r_co**2
    u  = cond.m_dot_per_pin / (cond.rho_NaK * A_flow)
    D_h = geom.D_co
    Pe  = cond.rho_NaK * u * D_h * cond.cp_NaK / props.k_NaK_coolant
    Nu  = _nusselt_liquid_metal_bundle(Pe, geom.P_D)
    return Nu * props.k_NaK_coolant / D_h


def subchannel_velocity(
    geom: LTMRPinGeometry,
    cond: LTMRCoolantConditions,
) -> float:
    """
    Mean coolant velocity in the fuel bundle subchannel [m/s].

    Uses the triangular-lattice unit-cell flow-area:
        A_flow_per_pin = sqrt(3)/4 * P² - pi/2 * r_co²

    where P = geom.pin_pitch = 2*r_co + pin_gap. This is the correct
    per-pin flow area for a close-packed hexagonal bundle (each
    equilateral-triangle unit cell contains half a pin cross-section).
    """
    P      = geom.pin_pitch
    A_flow = (np.sqrt(3) / 4.0) * P**2 - (np.pi / 2.0) * geom.r_co**2
    return cond.m_dot_per_pin / (cond.rho_NaK * A_flow)


def check_velocity(
    geom: LTMRPinGeometry,
    cond: LTMRCoolantConditions,
) -> float:
    """
    Compute the subchannel velocity and issue design guidance.

    NaK velocity limits depend heavily on the specific geometry, material,
    and operating temperature. Broad guidelines from liquid-metal reactor
    experience (EBR-II, FFTF, IAEA TECDOC-1677) for SS304/316:

      Recommended operating range : < 6 m/s
      Caution (elevated erosion)   : 6 – 12 m/s
      Avoid if possible            : > 12 m/s

    Note: compact liquid-metal microreactors (small P/D, high power density)
    commonly operate in the 6–15 m/s range due to tight subchannels. The
    binding thermal-hydraulic constraint for these designs is typically the
    cladding surface heat flux, not velocity alone. Use this output as one
    input to a broader erosion-corrosion assessment.

    Parameters
    ----------
    geom : LTMRPinGeometry
    cond : LTMRCoolantConditions

    Returns
    -------
    u : float
        Subchannel velocity [m/s]
    """
    u = subchannel_velocity(geom, cond)

    # Subchannel area and Re for context
    P      = geom.pin_pitch
    A_flow = (np.sqrt(3) / 4.0) * P**2 - (np.pi / 2.0) * geom.r_co**2
    Re     = cond.rho_NaK * u * geom.D_co / 3.5e-4  # mu_NaK ~ 3.5e-4 Pa·s

    if u > 12.0:
        level = "WARNING — above 12 m/s: elevated erosion risk for SS304/316. " \
                "Review cladding material selection and lifetime."
    elif u > 6.0:
        level = "CAUTION — 6–12 m/s: acceptable for compact LM microreactors " \
                "but warrants erosion-corrosion lifetime assessment."
    else:
        level = "OK — within conventional SFR operating range (< 6 m/s)."

    print(
        f"\n[Velocity Check]\n"
        f"  Subchannel velocity : {u:.2f} m/s  ({level})\n"
        f"  Reynolds number     : {Re:.0f}  (turbulent: Re >> 2300)\n"
        f"  A_flow per pin      : {A_flow * 1e4:.4f} cm²\n"
        f"  m_dot_per_pin       : {cond.m_dot_per_pin:.4f} kg/s\n"
        f"  m_dot_total         : {cond.m_dot_total:.2f} kg/s\n"
    )

    return u


# ============================================================================
# Radial heat conduction solver — called at each axial node
# ============================================================================

def _solve_radial(
    q_prime: float,
    T_bulk:  float,
    h_NaK:   float,
    geom:    LTMRPinGeometry,
    props:   LTMRThermalProperties,
) -> dict:
    """
    Closed-form radial temperature distribution at one axial location.

    Uses classroom notation: r_i, r_f, r_g, r_c.

    Parameters
    ----------
    q_prime : local linear heat rate at this axial node [W/m]
    T_bulk  : local coolant bulk temperature at this axial node [K]
    h_NaK   : convective HTC at cladding outer surface [W/(m²·K)]

    Returns
    -------
    dict with T_bulk, T_co, T_ci, T_fo, T_fi (= T_fuel_max), T_Zr_surface,
    q_prime, q_tpp, q_pp, h_NaK
    """
    r_i = geom.r_fi;  r_f = geom.r_fo
    r_g = geom.r_ci;  r_c = geom.r_co
    k_f = props.k_fuel;  k_c = props.k_clad;  k_g = props.k_NaK_gap

    q_tpp = q_prime / (np.pi * (r_f**2 - r_i**2))   # q''' [W/m³]
    q_pp  = q_prime / (2.0 * np.pi * r_c)            # q'' at clad OD [W/m²]

    # Cladding outer surface
    T_co = T_bulk + q_tpp * (r_f**2 - r_i**2) / (2.0 * h_NaK * r_c)

    # Cladding inner surface
    T_ci = T_co - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_c) * np.log(r_g / r_c)

    # Outer gap (hollow-cylinder conduction, no heat source)
    T_fo = T_ci + q_prime / (2.0 * np.pi * k_g) * np.log(r_g / r_f)

    # Fuel inner surface (peak fuel temperature)
    T_fi = (
        q_tpp / (4.0 * k_f) * (r_f**2 - r_i**2 + 2.0 * r_i**2 * np.log(r_i / r_f))
        - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_g) * np.log(r_f / r_g)
        - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_c) * np.log(r_g / r_c)
        + T_co
    )

    return {
        "q_prime":   q_prime,
        "q_tpp":     q_tpp,
        "q_pp":      q_pp,
        "h_NaK":     h_NaK,
        "T_bulk":    T_bulk,
        "T_co":      T_co,
        "T_ci":      T_ci,
        "T_fo":      T_fo,
        "T_fi":      T_fi,           # = T_fuel_max at this axial node
        "T_Zr":      T_fi,           # = T_fi (zero-flux BC at r_i)
    }


# ============================================================================
# Axial model — uniform linear heat rate, linear coolant temperature rise
# ============================================================================

def _axial_profiles(
    PF:          float,
    q_prime_nom: float,
    h_NaK:       float,
    geom:        LTMRPinGeometry,
    props:       LTMRThermalProperties,
    cond:        LTMRCoolantConditions,
    n_nodes:     int = 50,
) -> dict:
    """
    Compute axial temperature profiles for one pin assuming a uniform
    (flat) axial power distribution.

    Uniform power shape
    -------------------
    The local linear heat rate is constant along the entire active length:

        q'(z) = PF * q'_nom   for all z in [0, L]

    Coolant temperature (linear rise from inlet to outlet)
    -------------------------------------------------------
    With uniform heat deposition, the coolant temperature rises linearly:

        T_coolant(z) = T_inlet + (q' * z) / (m_dot_pin * cp_NaK)

    At z = L:
        T_coolant(L) = T_inlet + PF * q'_nom * L / (m_dot_pin * cp_NaK)
                     = T_inlet + PF * DeltaT_coolant_nom

    The radial solve (_solve_radial) is applied at each axial node using
    the local T_coolant(z) and the constant q'(z) = PF * q'_nom.
    The fuel peak temperature is therefore highest at the top of the core
    (z = L) where the coolant is hottest.

    Parameters
    ----------
    PF          : radial peaking factor for this pin (from MOUSE)
    q_prime_nom : nominal (core-average) linear heat rate [W/m]
    h_NaK       : convective HTC [W/(m²·K)]
    n_nodes     : number of axial nodes (default 50)

    Returns
    -------
    dict with:
        z              : axial positions [m], shape (n_nodes,)
        q_prime_z      : local linear heat rate [W/m]  (constant array)
        T_coolant_z    : coolant bulk temperature [K]
        T_fuel_max_z   : peak fuel temperature (at r_i) [K]
        T_co_z         : cladding outer surface temperature [K]
        DeltaT_fuel    : T_fuel_max(top) - T_fuel_max(bottom) [K]
        DeltaT_coolant : T_coolant(top)  - T_coolant(bottom)  [K]
        q_prime_pin    : pin linear heat rate [W/m]  (= PF * q'_nom)
    """
    L       = geom.L_active
    q_prime = PF * q_prime_nom   # uniform linear heat rate for this pin [W/m]

    z = np.linspace(0.0, L, n_nodes)

    # Uniform heat rate — same at every axial node
    q_prime_z = np.full(n_nodes, q_prime)

    # Coolant temperature: linear rise from inlet
    T_coolant_z = (cond.T_inlet
                   + q_prime * z / (cond.m_dot_per_pin * cond.cp_NaK))

    # Radial solve at each axial node
    T_fuel_max_z = np.zeros(n_nodes)
    T_co_z       = np.zeros(n_nodes)

    for j in range(n_nodes):
        sol = _solve_radial(q_prime_z[j], T_coolant_z[j], h_NaK, geom, props)
        T_fuel_max_z[j] = sol["T_fi"]
        T_co_z[j]       = sol["T_co"]

    return {
        "z":               z,
        "q_prime_z":       q_prime_z,
        "T_coolant_z":     T_coolant_z,
        "T_fuel_max_z":    T_fuel_max_z,
        "T_co_z":          T_co_z,
        "DeltaT_fuel":     float(T_fuel_max_z[-1] - T_fuel_max_z[0]),
        "DeltaT_coolant":  float(T_coolant_z[-1]  - T_coolant_z[0]),
        "q_prime_pin":     float(q_prime),
    }


# ============================================================================
# Public API
# ============================================================================

def compute_pin_temperatures(
    per_step_data: dict,
    geom:          LTMRPinGeometry,
    props:         LTMRThermalProperties,
    cond:          LTMRCoolantConditions,
    q_prime_nom:   float,
    n_axial:       int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute per-pin axial temperature profiles and top-minus-bottom differences.

    Parameters
    ----------
    per_step_data : dict
        Output of compute_pin_peaking_factors() (MOUSE).
        Keys: depletion step (int); Values: DataFrame with columns
        ['Region_ID', 'Peaking_Factor', 'Step'].

    geom : LTMRPinGeometry
        Pin radial geometry (LTMR defaults loaded automatically).

    props : LTMRThermalProperties
        Thermal conductivities.

    cond : LTMRCoolantConditions
        Coolant operating conditions.

    q_prime_nom : float
        Nominal (core-average) linear heat rate [W/m].
        = P_total / (N_fuel_pins * L_active)

    n_axial : int
        Number of axial nodes along the pin (default 50).

    Returns
    -------
    summary : pd.DataFrame
        One row per depletion step with columns:
            Step, Max_PF,
            Max_DeltaT_fuel [K], Max_DeltaT_coolant [K],
            Max_T_fuel_top [K], Max_T_coolant_top [K],
            Region_ID_hottest_fuel, Region_ID_hottest_coolant

    per_step_data_out : dict
        Keys: depletion step (int).
        Values: DataFrame with one row per pin:
            Region_ID, Peaking_Factor, Step,
            q_prime_pin [W/m],
            T_fuel_bottom [K], T_fuel_top [K], DeltaT_fuel [K],
            T_coolant_bottom [K], T_coolant_top [K], DeltaT_coolant [K],
            axial_profiles  (dict with full z, T arrays — for plotting)
    """
    summary_rows     = []
    per_step_data_out = {}

    print("\n========== LTMR PIN TEMPERATURE ANALYSIS (Axial + Radial) ==========\n")
    print(f"  Nominal q'_nom          : {q_prime_nom:.1f} W/m")
    print(f"  Coolant inlet T         : {cond.T_inlet:.2f} K")
    print(f"  Coolant outlet T        : {cond.T_outlet:.2f} K")
    print(f"  Per-pin mass-flow rate  : {cond.m_dot_per_pin:.4f} kg/s")
    print(f"  Total mass-flow rate    : {cond.m_dot_total:.2f} kg/s")
    print(f"  k_fuel                  : {props.k_fuel:.2f} W/(m·K)")
    print(f"  k_clad                  : {props.k_clad:.2f} W/(m·K)")
    print(f"  k_NaK_gap               : {props.k_NaK_gap:.2f} W/(m·K)")
    print(f"  Axial nodes             : {n_axial}")
    print(f"  Axial shape             : cosine  (peak = PF * q'_nom * pi/2)\n")

    check_velocity(geom, cond)

    for step, df in per_step_data.items():

        # HTC is the same for all pins (uniform flow assumption)
        h = compute_h_NaK(geom, props, cond)

        rows = []
        for _, pin in df.iterrows():
            PF = float(pin["Peaking_Factor"])

            ax = _axial_profiles(PF, q_prime_nom, h, geom, props, cond, n_axial)

            rows.append({
                "Region_ID":           pin["Region_ID"],
                "Peaking_Factor":      PF,
                "Step":                step,
                "q_prime_pin [W/m]":   ax["q_prime_pin"],
                # Bottom of core (z = 0, coolant inlet side)
                "T_fuel_bottom [K]":   float(ax["T_fuel_max_z"][0]),
                "T_coolant_bottom [K]": float(ax["T_coolant_z"][0]),
                # Top of core (z = L, coolant outlet side)
                "T_fuel_top [K]":      float(ax["T_fuel_max_z"][-1]),
                "T_coolant_top [K]":   float(ax["T_coolant_z"][-1]),
                # Top-minus-bottom differences
                "DeltaT_fuel [K]":     ax["DeltaT_fuel"],
                "DeltaT_coolant [K]":  ax["DeltaT_coolant"],
                # Full axial profiles stored for plotting
                "axial_profiles":      ax,
            })

        out = pd.DataFrame(rows)
        per_step_data_out[step] = out

        idx_hot_fuel    = out["DeltaT_fuel [K]"].abs().idxmax()
        idx_hot_coolant = out["DeltaT_coolant [K]"].abs().idxmax()

        print(f"--- Step {step} ---")
        print(out[["Region_ID", "Peaking_Factor",
                   "T_fuel_bottom [K]", "T_fuel_top [K]", "DeltaT_fuel [K]",
                   "T_coolant_bottom [K]", "T_coolant_top [K]",
                   "DeltaT_coolant [K]"]].to_string(index=False))
        print()

        summary_rows.append({
            "Step":                    step,
            "Max_PF":                  float(out["Peaking_Factor"].max()),
            "Max_DeltaT_fuel [K]":     float(out["DeltaT_fuel [K]"].max()),
            "Max_DeltaT_coolant [K]":  float(out["DeltaT_coolant [K]"].max()),
            "Max_T_fuel_top [K]":      float(out["T_fuel_top [K]"].max()),
            "Max_T_coolant_top [K]":   float(out["T_coolant_top [K]"].max()),
            "Region_ID_hottest_fuel":  out.loc[idx_hot_fuel,    "Region_ID"],
            "Region_ID_hottest_cool":  out.loc[idx_hot_coolant, "Region_ID"],
        })

    summary = pd.DataFrame(summary_rows).sort_values("Step")

    print("========== Summary (worst pin per step) ==========")
    print(summary.to_string(index=False))
    print("==================================================\n")

    return summary, per_step_data_out


# ============================================================================
# Standalone demo
# ============================================================================

if __name__ == "__main__":

    geom  = LTMRPinGeometry()
    props = LTMRThermalProperties()
    props.set_fuel_k_from_name("UN")

    P_total = 20.0e6; T_in = 703.15; T_out = 793.15; cp = 880.0
    N_pins  = 397

    cond = LTMRCoolantConditions(
        T_inlet        = T_in,
        DeltaT_coolant = T_out - T_in,
        P_total_W   = P_total,
        N_fuel_pins = N_pins,
        cp_NaK      = cp,
    )

    q_nom = P_total / (N_pins * geom.L_active)
    print(f"m_dot_total = {cond.m_dot_total:.1f} kg/s,  q'_nom = {q_nom:.1f} W/m\n")

    rng    = np.random.default_rng(42)
    n_pins = 20
    pf_raw = rng.uniform(0.7, 1.4, size=n_pins)
    pf_nom = pf_raw / pf_raw.mean()

    mock = {
        1: pd.DataFrame({"Region_ID": list(range(n_pins)),
                         "Peaking_Factor": pf_nom, "Step": 1}),
        2: pd.DataFrame({"Region_ID": list(range(n_pins)),
                         "Peaking_Factor": pf_nom * rng.uniform(0.97, 1.03, n_pins),
                         "Step": 2}),
    }

    summary, per_step = compute_pin_temperatures(
        per_step_data = mock,
        geom          = geom,
        props         = props,
        cond          = cond,
        q_prime_nom   = q_nom,
        n_axial       = 50,
    )

    # --- Quick sanity check: verify coolant ΔT matches lumped energy balance ---
    print("=== Sanity check: DeltaT_coolant vs lumped energy balance ===")
    step1 = per_step[1]
    for _, row in step1.iterrows():
        PF       = row["Peaking_Factor"]
        dT_axial = row["DeltaT_coolant [K]"]
        dT_lumped = PF * q_nom * geom.L_active / (cond.m_dot_per_pin * cp)
        match = np.isclose(dT_axial, dT_lumped, rtol=1e-6)
        print(f"  Pin {int(row['Region_ID']):2d}  PF={PF:.3f}  "
              f"DeltaT_axial={dT_axial:.4f}  lumped={dT_lumped:.4f}  match={match}")

# Use compute_pin_temperatures_abc() below which reports these clearly.

def compute_pin_temperatures_abc(
    per_step_data: dict,
    geom:          LTMRPinGeometry,
    props:         LTMRThermalProperties,
    cond:          LTMRCoolantConditions,
    q_prime_nom:   float,
    n_axial:       int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    ABC-analysis-ready version of compute_pin_temperatures.

    Reports per pin, per depletion step:
      DeltaT_coolant   : T_coolant(top) - T_coolant(bottom)
                         = axial coolant temperature rise  [K]
      DeltaT_fuel_peak : T_fuel_max(top) - T_inlet
                         = fuel temperature rise above inlet at the core outlet [K]
      T_fuel_peak      : absolute peak fuel temperature [K]
      T_coolant_top    : coolant outlet temperature for this pin [K]

    These are the inputs typically needed for ABC (hot-channel) analysis.
    """
    h = compute_h_NaK(geom, props, cond)

    summary_rows      = []
    per_step_data_out = {}

    print("\n========== ABC TEMPERATURE ANALYSIS ==========\n")
    print(f"  q'_nom = {q_prime_nom:.1f} W/m  |  T_inlet = {cond.T_inlet:.2f} K  "
          f"|  DeltaT_coolant = {cond.DeltaT_coolant:.2f} K  |  T_outlet = {cond.T_outlet:.2f} K  |  m_dot_total = {cond.m_dot_total:.2f} kg/s  "
          f"|  m_dot_per_pin = {cond.m_dot_per_pin:.4f} kg/s\n")

    check_velocity(geom, cond)

    for step, df in per_step_data.items():
        rows = []
        for _, pin in df.iterrows():
            PF = float(pin["Peaking_Factor"])
            ax = _axial_profiles(PF, q_prime_nom, h, geom, props, cond, n_axial)

            # With uniform power, fuel peak is at the top (hottest coolant)
            DeltaT_coolant   = float(ax["T_coolant_z"][-1] - ax["T_coolant_z"][0])
            DeltaT_fuel_peak = float(ax["T_fuel_max_z"][-1] - cond.T_inlet)
            T_fuel_peak      = float(ax["T_fuel_max_z"][-1])
            T_coolant_top    = float(ax["T_coolant_z"][-1])

            rows.append({
                "Region_ID":            pin["Region_ID"],
                "Peaking_Factor":       PF,
                "Step":                 step,
                "DeltaT_coolant [K]":   DeltaT_coolant,
                "DeltaT_fuel_peak [K]": DeltaT_fuel_peak,
                "T_fuel_peak [K]":      T_fuel_peak,
                "T_coolant_top [K]":    T_coolant_top,
                "axial_profiles":       ax,
            })

        out = pd.DataFrame(rows)
        per_step_data_out[step] = out

        print(f"--- Step {step} ---")
        print(out[["Region_ID", "Peaking_Factor",
                   "DeltaT_coolant [K]", "DeltaT_fuel_peak [K]",
                   "T_fuel_peak [K]", "T_coolant_top [K]"]].to_string(index=False))
        print()

        summary_rows.append({
            "Step":                        step,
            "Max_PF":                      float(out["Peaking_Factor"].max()),
            "Max_DeltaT_coolant [K]":      float(out["DeltaT_coolant [K]"].max()),
            "Max_DeltaT_fuel_peak [K]":    float(out["DeltaT_fuel_peak [K]"].max()),
            "Max_T_fuel_peak [K]":         float(out["T_fuel_peak [K]"].max()),
            "Max_T_coolant_top [K]":       float(out["T_coolant_top [K]"].max()),
            "Region_ID_hottest":           out.loc[out["T_fuel_peak [K]"].idxmax(), "Region_ID"],
        })

    summary = pd.DataFrame(summary_rows).sort_values("Step")
    print("========== ABC Summary ==========")
    print(summary.to_string(index=False))
    print("=================================\n")
    return summary, per_step_data_out


def run_thermal_analysis(params) -> None:
    """
    Run pin temperature ABC analysis after the OpenMC depletion run.

    Reads pf_per_step from params['PF Per Step'], which is stored there
    by openmc_depletion() in utils.py while the statepoints are still
    local. This avoids any dependency on the watts database path.

    Required params keys
    --------------------
    Geometry:
        'Fuel Pin Radii'    : [r_Zr, r_fi, r_fo, r_ci, r_co]  in cm
        'Pin Gap Distance'  : cm
        'Active Height'     : cm
        'Fuel Pin Count'    : int
    Power / coolant:
        'Power MWt'                      : MWt
        'Primary Loop Inlet Temperature' : K
        'Primary Loop Outlet Temperature': K
    Fuel:
        'Fuel' : string  e.g. 'UZrH_alloy', 'UN', 'UC', 'UO2'
    Neutronics (set by openmc_depletion):
        'PF Summary' : pf_summary.to_dict(orient='list') from compute_pin_peaking_factors()
                       Must contain columns: Step, Max_PF, Region_ID_Max
    """
    pf_summary_dict = params.get('PF Summary')

    if not pf_summary_dict:
        print("[Thermal] 'PF Summary' not found in params — skipping thermal analysis.")
        return

    # --- Reconstruct worst-case (max PF) pin per step from pf_summary ---
    # pf_summary was stored as a dict of lists via .to_dict(orient='list').
    # We only solve for the hottest pin at each step — this is a safety analysis.
    pf_summary = pd.DataFrame(pf_summary_dict)
    worst_case_per_step = {}
    for _, row in pf_summary.iterrows():
        step = row['Step']
        worst_case_per_step[step] = pd.DataFrame([{
            'Region_ID':      row['Region_ID_Max'],
            'Peaking_Factor': row['Max_PF'],
            'Step':           step,
        }])

    # --- Build geometry from params (all cm → m) ---
    radii = params['Fuel Pin Radii']   # [r_Zr, r_fi, r_fo, r_ci, r_co] in cm
    geom  = LTMRPinGeometry(
        r_Zr_o   = radii[0] * 1e-2,
        r_fi     = radii[1] * 1e-2,
        r_fo     = radii[2] * 1e-2,
        r_ci     = radii[3] * 1e-2,
        r_co     = radii[4] * 1e-2,
        pin_gap  = params['Pin Gap Distance'] * 1e-2,
        L_active = params['Active Height']    * 1e-2,
    )

    # --- Thermal properties: fuel conductivity from material name ---
    props = LTMRThermalProperties()
    props.set_fuel_k_from_name(params['Fuel'])

    # --- Coolant conditions: inlet/outlet T set by user; m_dot derived ---
    cond = LTMRCoolantConditions(
        T_inlet     = params['Primary Loop Inlet Temperature'],
        DeltaT_coolant = (params['Primary Loop Outlet Temperature']
                          - params['Primary Loop Inlet Temperature']),
        P_total_W   = params['Power MWt'] * 1e6,
        N_fuel_pins = params['Fuel Pin Count'],
    )

    # Mirror the params that mass_flow_rate() in tools.py would have set.
    # For the LTMR there is no 'Primary Loop per loop load fraction' key, so
    # loop_factor = 1 and both quantities equal cond.m_dot_total.
    loop_factor = params.get('Primary Loop per loop load fraction', 1)
    params['Coolant Mass Flow Rate']      = cond.m_dot_total / loop_factor
    params['Primary Loop Mass Flow Rate'] = cond.m_dot_total

    # --- Nominal linear heat rate ---
    q_nom = (params['Power MWt'] * 1e6) / (params['Fuel Pin Count'] * geom.L_active)

    # --- Run ABC temperature analysis (worst-case pin per step only) ---
    abc_summary, abc_per_step = compute_pin_temperatures_abc(
        per_step_data = worst_case_per_step,
        geom          = geom,
        props         = props,
        cond          = cond,
        q_prime_nom   = q_nom,
    )

    # --- Store results back into params ---
    params['ABC Worst Case Summary']     = abc_summary
    params['ABC Worst Case Per Step']    = abc_per_step
    params['Max DeltaT Fuel [K]']        = float(
        abc_summary['Max_DeltaT_fuel_peak [K]'].max())
    params['Max DeltaT Coolant [K]']     = float(
        abc_summary['Max_DeltaT_coolant [K]'].max())
    params['Max T Fuel Peak [K]']        = float(
        abc_summary['Max_T_fuel_peak [K]'].max())

    print(f"\n[Thermal] Max fuel peak temperature : "
          f"{params['Max T Fuel Peak [K]']:.1f} K")
    print(f"[Thermal] Max fuel peak ΔT (above inlet) : "
          f"{params['Max DeltaT Fuel [K]']:.1f} K")
    print(f"[Thermal] Max coolant axial ΔT : "
          f"{params['Max DeltaT Coolant [K]']:.1f} K\n")