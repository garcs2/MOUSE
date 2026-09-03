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

    These field defaults are only a convenience for direct instantiation
    (quick tests, notebooks). On the production path the geometry is built
    from the watts params dict via LTMRPinGeometry.from_params(params), which
    is the single source of truth for the params -> geometry mapping.

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
    annular_fuel: Optional[bool] = None
                                   # Fuel geometry selector:
                                   #   None  -> auto-detect from r_fi (annular if
                                   #            r_fi > 0, i.e. a central Zr rod /
                                   #            inner void with a zero-flux inner
                                   #            boundary; solid if r_fi == 0)
                                   #   True  -> force annular treatment
                                   #   False -> force solid-pellet treatment

    @property
    def has_central_rod(self) -> bool:
        """True for annular fuel (central Zr rod or inner void — peak fuel
        temperature sits at the inner surface r_fi, zero-flux BC there).
        False for solid fuel (fuel fills to the centerline r=0, peak at r=0).

        Auto-detected from r_fi unless `annular_fuel` is set explicitly. To
        model solid fuel, pass r_Zr_o = r_fi = 0 (e.g. 'Fuel Pin Radii' =
        [0.0, 0.0, r_fo, r_ci, r_co])."""
        if self.annular_fuel is not None:
            return self.annular_fuel
        return self.r_fi > 1e-9   # metres; any real inner radius is >> 1 nm

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

    @classmethod
    def from_params(cls, params) -> "LTMRPinGeometry":
        """Build the pin geometry directly from the watts params dict.

        This is the authoritative params -> geometry mapping; the dataclass
        field defaults are only used when constructing without params.

        Reads (all lengths in cm, converted to metres):
            'Fuel Pin Radii'   : [r_Zr, r_fi, r_fo, r_ci, r_co]
            'Pin Gap Distance' : centre-to-surface gap between adjacent pins
            'Active Height'    : active (heated) height
            'Fuel Pin Materials' (optional) : decides annular vs solid fuel
                (see _fuel_is_annular). For solid fuel the inner fuel radius is
                forced to 0 so that q''' = q'/(pi*r_fo^2) uses the correct area;
                the radii[1] entry (an internal split of a solid fuel region)
                is then ignored.
        """
        radii = params['Fuel Pin Radii']
        if len(radii) < 5:
            raise ValueError(
                "'Fuel Pin Radii' must have 5 entries "
                f"[r_Zr, r_fi, r_fo, r_ci, r_co]; got {radii!r}"
            )
        for key in ('Pin Gap Distance', 'Active Height'):
            if key not in params:
                raise KeyError(f"run_thermal_analysis: missing required param '{key}'")

        is_annular = _fuel_is_annular(params)
        r_fi_cm    = radii[1] if is_annular else 0.0
        return cls(
            r_Zr_o   = radii[0] * 1e-2,
            r_fi     = r_fi_cm  * 1e-2,
            r_fo     = radii[2] * 1e-2,
            r_ci     = radii[3] * 1e-2,
            r_co     = radii[4] * 1e-2,
            pin_gap  = params['Pin Gap Distance'] * 1e-2,
            L_active = params['Active Height']    * 1e-2,
            annular_fuel = is_annular,
        )


@dataclass
class LTMRThermalProperties:
    """
    Thermal properties for each layer.

    Defaults:
      k_fuel      : UO2 at ~800 K ≈ 3.5 W/m·K  (conservative; UN ~ 20, UC ~ 20)
      k_clad      : SS304 at 700 K ≈ 19 W/m·K
      k_gap       : conductivity of the fuel–clad (outer) gap fill, used in the
                    hollow-cylinder gap conduction term
                      ΔT_gap = q'/(2π k_gap) · ln(r_ci/r_fo)
                    Set from the gap fill material via set_gap_k_from_name:
                    sodium-bonded pins ≈ 70 W/m·K, helium-bonded pins ≈ 0.15-0.3
                    W/m·K. This is the gap BOND, distinct from the NaK coolant.
      k_NaK_coolant : bulk NaK thermal conductivity for the Kazimi-Carelli HTC
                      correlation (the coolant, not the gap).
    """
    k_fuel:         float = 3.5    # W/(m·K)  — UO2 default; override for UN/UC
    k_clad:         float = 19.0   # W/(m·K)  — SS304
    k_gap:          float = 70.0   # W/(m·K)  — Na bond default; He gap ~0.15-0.3
    # NOTE: no h_gap_inner — zero-flux BC at r_i means T_Zr = T_fi exactly.
    k_NaK_coolant:  float = 24.0   # W/(m·K)  — bulk NaK coolant (for Nu correlation)

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
    GAP_K_DEFAULTS: dict = field(default_factory=lambda: {
        # Liquid-metal bonds (~ high conductivity)
        "Na":     70.0,     # sodium bond
        "Sodium": 70.0,
        "NaK":    24.0,     # sodium-potassium, if used as a bond
        # Gas bonds (~ low conductivity). He value is a cold/conservative
        # figure (~0.15); helium at 700-900 K is nearer 0.3 W/m·K. A lower k
        # overestimates the gap ΔT, i.e. it is conservative (hotter fuel) for a
        # safety screen. Override props.k_gap directly for a temperature-
        # specific value.
        "He":     0.151,    # helium bond
        "Helium": 0.151,
    })

    def set_fuel_k_from_name(self, fuel_name: str) -> None:
        """Automatically set k_fuel from fuel material name if known."""
        k = self.FUEL_K_DEFAULTS.get(fuel_name)
        if k is not None:
            self.k_fuel = k
        else:
            print(f"[ThermalProps] Unknown fuel '{fuel_name}'; "
                  f"keeping k_fuel = {self.k_fuel} W/(m·K)")

    def set_gap_k_from_name(self, gap_name) -> None:
        """Set k_gap from the fuel–clad gap fill material (case-insensitive).

        Accepts e.g. 'Na'/'Sodium' or 'He'/'Helium'. If gap_name is None or
        unrecognised, k_gap is left at its current value and a note is printed
        so a silent fallback is visible in the run log.
        """
        if gap_name is None:
            print(f"[ThermalProps] Gap fill material unspecified; "
                  f"keeping k_gap = {self.k_gap} W/(m·K). Set params['Gap "
                  f"Material'] (e.g. 'He' or 'Na') to select it explicitly.")
            return
        # case-insensitive lookup against the defaults table
        lookup = {name.lower(): k for name, k in self.GAP_K_DEFAULTS.items()}
        k = lookup.get(str(gap_name).strip().lower())
        if k is not None:
            self.k_gap = k
        else:
            print(f"[ThermalProps] Unknown gap fill '{gap_name}'; "
                  f"keeping k_gap = {self.k_gap} W/(m·K)")
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
    k_f = props.k_fuel;  k_c = props.k_clad;  k_g = props.k_gap

    # q''' uses the actual fuel cross-section: pi*(r_f^2 - r_i^2) for annular
    # fuel, pi*r_f^2 for solid fuel (r_i = 0). The clad/gap/film drops below
    # depend only on the total linear power q', so they are identical for both
    # geometries; only the fuel-region terms (peak and volume average) branch.
    q_tpp = q_prime / (np.pi * (r_f**2 - r_i**2))   # q''' [W/m³]
    q_pp  = q_prime / (2.0 * np.pi * r_c)            # q'' at clad OD [W/m²]

    # Cladding outer surface
    T_co = T_bulk + q_tpp * (r_f**2 - r_i**2) / (2.0 * h_NaK * r_c)

    # Cladding inner surface
    T_ci = T_co - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_c) * np.log(r_g / r_c)

    # Outer gap (hollow-cylinder conduction, no heat source)
    T_fo = T_ci + q_prime / (2.0 * np.pi * k_g) * np.log(r_g / r_f)

    if geom.has_central_rod:
        # ---- Annular fuel: zero-flux inner BC at r_i, peak at r_i ----
        # Fuel inner surface (peak fuel temperature)
        T_peak = (
            q_tpp / (4.0 * k_f) * (r_f**2 - r_i**2 + 2.0 * r_i**2 * np.log(r_i / r_f))
            - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_g) * np.log(r_f / r_g)
            - q_tpp * (r_f**2 - r_i**2) / (2.0 * k_c) * np.log(r_g / r_c)
            + T_co
        )

        # Volume (area) averaged fuel temperature over the annulus r_i..r_f.
        # Profile relative to the fuel outer surface (all heat flows outward):
        #     T(r) - T_fo = q'''/(2 k_f) * [ r_i^2 ln(r/r_f) - (r^2 - r_f^2)/2 ]
        # Area-averaging with weight 2*pi*r dr over [r_i, r_f] gives:
        #     T_f_avg - T_fo = q'''/k_f * [ (r_f^2 - r_i^2)/8 - r_i^2/4
        #                                 + r_i^4/(2(r_f^2 - r_i^2)) * ln(r_f/r_i) ]
        # Verified vs quadrature; reduces to q'/(8 pi k_f) as r_i -> 0.
        dA = r_f**2 - r_i**2
        T_f_avg = T_fo + q_tpp / k_f * (
            dA / 8.0
            - r_i**2 / 4.0
            + r_i**4 / (2.0 * dA) * np.log(r_f / r_i)
        )
        T_center = T_peak   # peak is at the inner surface for annular fuel
    else:
        # ---- Solid fuel: symmetry (zero-flux) at r = 0, peak at centerline ----
        # Profile relative to the fuel outer surface:
        #     T(r) - T_fo = q'''/(4 k_f) * (r_f^2 - r^2)
        # Centerline (r=0):     T_center - T_fo = q''' r_f^2 /(4 k_f) = q'/(4 pi k_f)
        # Area-average [0,r_f]: T_f_avg  - T_fo = q''' r_f^2 /(8 k_f) = q'/(8 pi k_f)
        # (the annular closed form has ln(r_i) terms that go NaN at r_i=0, so the
        #  solid case is handled with its own finite closed form.)
        T_center = T_fo + q_tpp * r_f**2 / (4.0 * k_f)
        T_f_avg  = T_fo + q_tpp * r_f**2 / (8.0 * k_f)
        T_peak   = T_center

    return {
        "q_prime":   q_prime,
        "q_tpp":     q_tpp,
        "q_pp":      q_pp,
        "h_NaK":     h_NaK,
        "T_bulk":    T_bulk,
        "T_co":      T_co,
        "T_ci":      T_ci,
        "T_fo":      T_fo,
        "T_fi":      T_peak,         # = T_fuel_max at this node (inner surface
                                     #   for annular, centerline for solid)
        "T_f_avg":   T_f_avg,        # = volume-averaged fuel temp at this node
        "T_center":  T_center,       # = centerline temp (solid) / inner-surf (annular)
        "T_Zr":      T_peak if geom.has_central_rod else float("nan"),
                                     #   Zr-rod surface temp (annular only; nan for solid)
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
# Internal helper: solve and print a single PF case
# ============================================================================

def _solve_and_print_case(
    label:       str,
    PF:          float,
    q_prime_nom: float,
    h_NaK:       float,
    geom:        LTMRPinGeometry,
    props:       LTMRThermalProperties,
    cond:        LTMRCoolantConditions,
    n_axial:     int = 50,
) -> dict:
    """
    Run _axial_profiles for a given PF, print a detailed radial summary
    at the outlet (top of core, worst-case location), and return the
    axial profile dict.
    """
    ax  = _axial_profiles(PF, q_prime_nom, h_NaK, geom, props, cond, n_axial)
    sol = _solve_radial(ax["q_prime_z"][-1], ax["T_coolant_z"][-1], h_NaK, geom, props)

    print(f"\n  {'─'*55}")
    print(f"  {label}  (PF = {PF:.4f})")
    print(f"  {'─'*55}")
    print(f"  q' (pin linear heat rate)  : {ax['q_prime_pin']:.1f} W/m")
    print(f"  T_coolant (outlet / top)   : {ax['T_coolant_z'][-1]:.2f} K  "
          f"({ax['T_coolant_z'][-1]-273.15:.1f} °C)")
    print(f"  T_co  (clad outer, top)    : {sol['T_co']:.2f} K  "
          f"({sol['T_co']-273.15:.1f} °C)")
    print(f"  T_ci  (clad inner, top)    : {sol['T_ci']:.2f} K  "
          f"({sol['T_ci']-273.15:.1f} °C)")
    print(f"  T_fo  (fuel outer, top)    : {sol['T_fo']:.2f} K  "
          f"({sol['T_fo']-273.15:.1f} °C)")
    print(f"  T_fi  (fuel peak, top)     : {sol['T_fi']:.2f} K  "
          f"({sol['T_fi']-273.15:.1f} °C)")
    print(f"  DeltaT_coolant (top-bot)   : {ax['DeltaT_coolant']:.2f} K")
    print(f"  DeltaT_fuel    (top-bot)   : {ax['DeltaT_fuel']:.2f} K")
    print(f"  DeltaT_fuel_peak (T_fi - T_inlet) : "
          f"{sol['T_fi'] - cond.T_inlet:.2f} K")

    return ax


# ============================================================================
# Public API
# ============================================================================

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

    For each depletion step, evaluates TWO cases:
      1) The worst-case pin at the actual PF from OpenMC (safety case)
      2) A reference pin at PF = 1.0 (nominal / validation case)

    Reports per pin, per depletion step:
      DeltaT_coolant   : T_coolant(top) - T_coolant(bottom) [K]
      DeltaT_fuel_peak : T_fuel_max(top) - T_inlet [K]
      T_fuel_peak      : absolute peak fuel temperature [K]
      T_coolant_top    : coolant outlet temperature for this pin [K]

    The PF=1 reference case is printed for validation and stored under
    params['ABC Nominal Reference'].
    """
    h = compute_h_NaK(geom, props, cond)

    summary_rows      = []
    per_step_data_out = {}

    print("\n========== ABC TEMPERATURE ANALYSIS ==========\n")
    print(f"  q'_nom = {q_prime_nom:.1f} W/m  |  T_inlet = {cond.T_inlet:.2f} K  "
          f"|  DeltaT_coolant = {cond.DeltaT_coolant:.2f} K  "
          f"|  T_outlet = {cond.T_outlet:.2f} K  "
          f"|  m_dot_total = {cond.m_dot_total:.2f} kg/s  "
          f"|  m_dot_per_pin = {cond.m_dot_per_pin:.4f} kg/s\n")

    check_velocity(geom, cond)

    # --- PF = 1 reference case (printed once, same for all steps) ---
    print("\n========== NOMINAL REFERENCE CASE (PF = 1.0) ==========")
    ax_nom = _solve_and_print_case(
        label       = "Nominal pin (PF = 1.0)",
        PF          = 1.0,
        q_prime_nom = q_prime_nom,
        h_NaK       = h,
        geom        = geom,
        props       = props,
        cond        = cond,
        n_axial     = n_axial,
    )
    print()

    for step, df in per_step_data.items():
        rows = []

        print(f"========== STEP {step} — WORST-CASE PIN ==========")

        for _, pin in df.iterrows():
            PF = float(pin["Peaking_Factor"])

            ax = _solve_and_print_case(
                label       = f"Worst-case pin (Region {pin['Region_ID']})",
                PF          = PF,
                q_prime_nom = q_prime_nom,
                h_NaK       = h,
                geom        = geom,
                props       = props,
                cond        = cond,
                n_axial     = n_axial,
            )

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

        print()

        out = pd.DataFrame(rows)
        per_step_data_out[step] = out

        summary_rows.append({
            "Step":                        step,
            "Max_PF":                      float(out["Peaking_Factor"].max()),
            "Max_DeltaT_coolant [K]":      float(out["DeltaT_coolant [K]"].max()),
            "Max_DeltaT_fuel_peak [K]":    float(out["DeltaT_fuel_peak [K]"].max()),
            "Max_T_fuel_peak [K]":         float(out["T_fuel_peak [K]"].max()),
            "Max_T_coolant_top [K]":       float(out["T_coolant_top [K]"].max()),
            "Region_ID_hottest":           out.loc[out["T_fuel_peak [K]"].idxmax(), "Region_ID"],
            # PF=1 reference values (same for all steps)
            "Ref_PF1_T_fuel_peak [K]":     float(ax_nom["T_fuel_max_z"][-1]),
            "Ref_PF1_T_coolant_top [K]":   float(ax_nom["T_coolant_z"][-1]),
            "Ref_PF1_DeltaT_coolant [K]":  float(ax_nom["DeltaT_coolant"]),
            "Ref_PF1_DeltaT_fuel_peak [K]": float(ax_nom["T_fuel_max_z"][-1] - cond.T_inlet),
        })

    summary = pd.DataFrame(summary_rows).sort_values("Step")

    print("\n========== ABC Summary ==========")
    print(summary.to_string(index=False))
    print("=================================\n")

    return summary, per_step_data_out


def _fuel_is_annular(params) -> bool:
    """Decide annular vs solid fuel from the already-built params.

    Reads 'Fuel Pin Materials' (layer materials ordered centre -> clad) and the
    fuel name in params['Fuel']. Fuel is treated as annular when the innermost
    layer is a distinct non-fuel solid — e.g. a 'Zr' central rod that generates
    no power, giving a zero-flux boundary at the fuel inner radius. It is solid
    when the fuel material fills the centre:

        annular : ['Zr',   None,   <fuel>, None, 'SS304']
        solid   : [<fuel>, <fuel>, <fuel>, None, 'SS304']

    An explicit params['Annular Fuel'] (True/False) overrides the material
    inspection. If neither the flag nor the materials list is available, falls
    back to a radius test (inner fuel radius radii[1] > 0).
    """
    override = params.get('Annular Fuel')
    if override is not None:
        return bool(override)

    mats = params.get('Fuel Pin Materials')
    if mats:
        fuel_name = params.get('Fuel')
        # Annular iff the central layer is a real material other than the fuel.
        return mats[0] not in (None, fuel_name)

    # Fallback: infer from the inner fuel radius.
    radii = params.get('Fuel Pin Radii')
    return bool(radii) and radii[1] > 0.0


def _resolve_gap_material(params):
    """Determine the fuel–clad (outer) gap fill material from params.

    The gap bond is distinct from the NaK coolant and is typically helium
    (gas-bonded oxide pins) or sodium (metal/liquid-metal-bonded pins).

    Preference order:
      1. An explicit param naming the fill:
         'Gap Material' / 'Bond Material' / 'Fuel-Clad Gap Material' / 'Gap Fill'
      2. The outer-gap entry in 'Fuel Pin Materials' (index -2), if it names a
         material rather than None. NOTE: in the LTMR arrays the gap slots are
         written as None, so this usually yields nothing and an explicit param
         is required.

    Returns the material name (str) or None if it cannot be resolved.
    """
    for key in ('Gap Material', 'Bond Material',
                'Fuel-Clad Gap Material', 'Gap Fill'):
        val = params.get(key)
        if val:
            return val

    mats = params.get('Fuel Pin Materials')
    if mats and len(mats) >= 2 and mats[-2] not in (None, ''):
        return mats[-2]

    return None


def run_thermal_analysis(params) -> None:
    """
    Run pin temperature ABC analysis after the OpenMC depletion run.

    Reads pf_per_step from params['PF Summary'], which is stored there
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
    pf_summary = pd.DataFrame(pf_summary_dict)
    worst_case_per_step = {}
    for _, row in pf_summary.iterrows():
        step = row['Step']
        worst_case_per_step[step] = pd.DataFrame([{
            'Region_ID':      row['Region_ID_Max'],
            'Peaking_Factor': row['Max_PF'],
            'Step':           step,
        }])

    # --- Build geometry from params (single source of truth) ---
    geom = LTMRPinGeometry.from_params(params)
    params['Fuel Geometry'] = 'annular' if geom.has_central_rod else 'solid'
    _mats = params.get('Fuel Pin Materials')
    print(f"[Thermal] Fuel geometry: {params['Fuel Geometry']}"
          + (f"  (materials: {_mats})" if _mats else "")
          + f"\n[Thermal]   r_fi = {geom.r_fi*1e2:.4f} cm, "
            f"r_fo = {geom.r_fo*1e2:.4f} cm, "
            f"r_ci = {geom.r_ci*1e2:.4f} cm, r_co = {geom.r_co*1e2:.4f} cm"
          + f"\n[Thermal]   L_active = {geom.L_active*1e2:.2f} cm, "
            f"pin_gap = {geom.pin_gap*1e2:.4f} cm")

    # --- Thermal properties ---
    props = LTMRThermalProperties()
    props.set_fuel_k_from_name(params['Fuel'])

    gap_mat = _resolve_gap_material(params)
    props.set_gap_k_from_name(gap_mat)
    params['Gap Material'] = gap_mat
    params['Gap Conductivity [W/m-K]'] = props.k_gap
    print(f"[Thermal]   Gap fill: {gap_mat if gap_mat else 'unspecified'} "
          f"-> k_gap = {props.k_gap:.3f} W/(m·K)")
    # --- Coolant conditions ---
    cond = LTMRCoolantConditions(
        T_inlet        = params['Primary Loop Inlet Temperature'],
        DeltaT_coolant = (params['Primary Loop Outlet Temperature']
                          - params['Primary Loop Inlet Temperature']),
        P_total_W      = params['Power MWt'] * 1e6,
        N_fuel_pins    = params['Fuel Pin Count'],
    )

    loop_factor = params.get('Primary Loop per loop load fraction', 1)
    params['Coolant Mass Flow Rate']      = cond.m_dot_total / loop_factor
    params['Primary Loop Mass Flow Rate'] = cond.m_dot_total

    # --- Nominal linear heat rate ---
    q_nom = (params['Power MWt'] * 1e6) / (params['Fuel Pin Count'] * geom.L_active)

    # --- Run ABC temperature analysis ---
    abc_summary, abc_per_step = compute_pin_temperatures_abc(
        per_step_data = worst_case_per_step,
        geom          = geom,
        props         = props,
        cond          = cond,
        q_prime_nom   = q_nom,
    )

    # --- Store results ---
    params['ABC Worst Case Summary']      = abc_summary
    params['ABC Worst Case Per Step']     = abc_per_step
    params['Max DeltaT Fuel [K]']         = float(abc_summary['Max_DeltaT_fuel_peak [K]'].max())
    params['Max DeltaT Coolant [K]']      = float(abc_summary['Max_DeltaT_coolant [K]'].max())
    params['Max T Fuel Peak [K]']         = float(abc_summary['Max_T_fuel_peak [K]'].max())
    params['Ref PF1 T Fuel Peak [K]']     = float(abc_summary['Ref_PF1_T_fuel_peak [K]'].iloc[0])
    params['Ref PF1 T Coolant Top [K]']   = float(abc_summary['Ref_PF1_T_coolant_top [K]'].iloc[0])

    # --- Average fuel temperature and fuel-to-coolant dT for the ABC screen ---
    # The A integral parameter wants <T_fuel> - <T_coolant> for the core-average
    # (PF = 1) pin, not the hot-pin peak. Under this file's uniform axial-power
    # model, q'(z) is constant along z, so every radial temperature drop
    # (film, clad, gap, fuel-radial-average) is the same at every axial node.
    # The fuel-to-coolant offset is therefore constant in z, and
    #     <T_fuel> - <T_coolant> = T_f_avg(node) - T_bulk(node)
    # exactly, independent of the axial coolant rise. A single radial solve at
    # q'_nom evaluated at the mean coolant temperature gives it directly.
    h_NaK      = compute_h_NaK(geom, props, cond)
    T_cool_avg = cond.T_inlet + 0.5 * cond.DeltaT_coolant
    sol_avg    = _solve_radial(q_nom, T_cool_avg, h_NaK, geom, props)

    params['Coolant Average Temperature']  = T_cool_avg
    params['Fuel Average Temperature']     = float(sol_avg['T_f_avg'])
    params['Fuel Outer Temperature']       = float(sol_avg['T_fo'])
    params['Fuel Peak Temp at Mean Coolant [K]'] = float(sol_avg['T_fi'])
    # Key consumed by _evaluate_abc_criteria for the A integral parameter:
    params['Fuel-Coolant dT']              = float(sol_avg['T_f_avg'] - T_cool_avg)

    print(f"\n[Thermal] ── Nominal reference (PF=1) ──")
    print(f"[Thermal]   T_fuel_peak      : {params['Ref PF1 T Fuel Peak [K]']:.1f} K  "
          f"({params['Ref PF1 T Fuel Peak [K]']-273.15:.1f} °C)")
    print(f"[Thermal]   T_coolant_top    : {params['Ref PF1 T Coolant Top [K]']:.1f} K  "
          f"({params['Ref PF1 T Coolant Top [K]']-273.15:.1f} °C)")
    print(f"\n[Thermal] ── Worst-case pin (max PF) ──")
    print(f"[Thermal]   Max T_fuel_peak  : {params['Max T Fuel Peak [K]']:.1f} K  "
          f"({params['Max T Fuel Peak [K]']-273.15:.1f} °C)")
    print(f"[Thermal]   Max DeltaT_fuel  : {params['Max DeltaT Fuel [K]']:.1f} K")
    print(f"[Thermal]   Max DeltaT_cool  : {params['Max DeltaT Coolant [K]']:.1f} K\n")

    print(f"[Thermal] ── ABC screen inputs (core-average, PF = 1) ──")
    print(f"[Thermal]   T_fuel_avg       : {params['Fuel Average Temperature']:.1f} K  "
          f"({params['Fuel Average Temperature']-273.15:.1f} °C)")
    print(f"[Thermal]   T_coolant_avg    : {params['Coolant Average Temperature']:.1f} K  "
          f"({params['Coolant Average Temperature']-273.15:.1f} °C)")
    print(f"[Thermal]   Fuel-Coolant dT  : {params['Fuel-Coolant dT']:.1f} K\n")


def plot_peaking_factor_map(per_step_data: dict, statepoint_path: str, params) -> None:
    """
    Generate a spatial peaking factor map for each depletion step.
    Saves peaking_factor_map_step{N}.png in the current directory.
    """
    from core_design.peaking_factor import get_pin_positions
    import matplotlib.pyplot as plt

    pin_pitch = (2 * params['Fuel Pin Radii'][-1] + params['Pin Gap Distance'])  # cm
    positions = get_pin_positions(statepoint_path, pin_pitch)

    for step, df in per_step_data.items():
        pf = df.rename(columns={'Region_ID': 'region_id'}).copy()
        pf['region_id'] = pf['region_id'].astype(int)
        merged = positions.merge(pf, on='region_id')

        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(merged['x [cm]'], merged['y [cm]'],
                        c=merged['Peaking_Factor'],
                        cmap='hot_r', s=300, vmin=0.8, vmax=1.4)
        plt.colorbar(sc, ax=ax, label='Peaking Factor')

        hot = merged.loc[merged['Peaking_Factor'].idxmax()]
        ax.annotate(f"Max PF\n{hot['Peaking_Factor']:.3f}\nID={int(hot['region_id'])}",
                    xy=(hot['x [cm]'], hot['y [cm]']), fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', fc='yellow', alpha=0.8))

        ax.set_aspect('equal')
        ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]')
        ax.set_title(f'Pin Peaking Factor Map — Step {step}')
        plt.tight_layout()
        plt.savefig(f'peaking_factor_map_step{step}.png', dpi=150)
        plt.close()
        print(f"[Thermal] Saved peaking_factor_map_step{step}.png")