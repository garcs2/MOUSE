# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
# Importing libraries
import openmc


# ==================================================================================
#  Library-version dispatcher
# ==================================================================================

# Supported values for params['XS_type']:
SUPPORTED_XS_TYPES = ('endf8.0', 'endf8.1')


def collect_materials_data(params):
    """Public entry point. Routes to the correct library-specific builder based
    on params['XS_type'].

    params['XS_type'] must be one of: 'endf8.0', 'endf8.1'
    """
    xs_type = params.get('XS_type', '').lower().strip()

    if xs_type == 'endf8.1':
        print(f"XS_type = '{xs_type}' → loading ENDF/B-VIII.1 material definitions")
        return _collect_materials_endf81(params)
    elif xs_type == 'endf8.0':
        print(f"XS_type = '{xs_type}' → loading ENDF/B-VIII.0 material definitions")
        return _collect_materials_endf80(params)
    else:
        raise ValueError(
            f"Unrecognised XS_type '{xs_type}'. "
            f"Must be one of: {SUPPORTED_XS_TYPES}"
        )


# ==================================================================================
#  Shared helpers
# ==================================================================================

def _enrich_tsl_suffix(enrichment_frac):
    """Return the ENDF/B-VIII.1 enrichment-specific TSL suffix for uranium fuel
    materials based on the U-235 atom/weight fraction.

    Matches the enrichment-specific TSL families in the ENDF/B-VIII.1 library:
      _5p    ->  ~5%  (LEU)
      _10p   -> ~10%  (LEU+)
      _HALEU -> ~19.75% (HALEU)
      _HEU   -> >=50% (HEU)
      _100p  -> 100%  (pure U-235)
      ''     -> natural / no suffix (fallback for other enrichments)
    """
    e = enrichment_frac
    if abs(e - 1.00) < 0.01:
        return "_100p"
    elif e >= 0.50:
        return "_HEU"
    elif abs(e - 0.1975) < 0.03:   # HALEU band ~17-22%
        return "_HALEU"
    elif abs(e - 0.10) < 0.03:     # 10% band ~7-13%
        return "_10p"
    elif abs(e - 0.05) < 0.03:     # 5% band ~2-8%
        return "_5p"
    else:
        return ""  # generic (natural) table fallback


def _build_base_materials(params):
    """Build all OpenMC material objects WITHOUT any S(α,β) tables attached.
    Returns a dict of named materials ready for TSL assignment.

    Both ENDF/B-VIII.0 and VIII.1 collectors call this first, then add
    their library-specific S(α,β) tables on top.
    """
    mats = {}

    # ------------------------------------------------------------------
    # Sec. 1.1 : Fuels
    # ------------------------------------------------------------------

    # TRIGA fuel components
    try:
        U_met = openmc.Material(name="U_met")
        U_met.set_density("g/cm3", 19.05)
        U_met.add_nuclide("U235", params['Enrichment'])
        U_met.add_nuclide("U238", 1 - params['Enrichment'])

        ZrH_fuel = openmc.Material(name="ZrH_fuel")
        ZrH_fuel.set_density("g/cm3", 5.63)
        ZrH_fuel.add_element("zirconium", 1.0)
        ZrH_fuel.add_nuclide("H1", params["H_Zr_ratio"])

        Er_bp = openmc.Material(name="Er_bp")
        Er_bp.set_density("g/cm3", 9.07)
        Er_bp.add_element("erbium", 1.0)

        er_wo = params['er_wo']

        UZrH_alloy = openmc.Material.mix_materials(
            [U_met, ZrH_fuel, Er_bp],
            [params['U_met_wo'], 1 - params['U_met_wo'] - er_wo, er_wo],
            "wo", name="UZrH")
        UZrH_alloy.temperature = params['Common Temperature']
        mats['U_met']      = U_met
        mats['UZrH_alloy'] = UZrH_alloy

    except KeyError as e:
        print(f"Skipping UZrH_alloy due to missing parameter: {e}")

    # UO2
    try:
        UO2 = openmc.Material(name='UO2')
        UO2.set_density('g/cm3', 10.41)
        UO2.add_element('U', 1.0, enrichment=100 * params['Enrichment'])
        UO2.add_nuclide('O16', 2.0)
        mats['UO2'] = UO2
    except KeyError as e:
        print(f"Skipping UO2 due to missing parameter: {e}")

    # Uranium Carbide
    try:
        UC = openmc.Material(name='UC')
        UC.set_density('g/cm3', 13.0)
        UC.add_element('U', 1.0, enrichment=100 * params['Enrichment'])
        UC.add_element('C', 1.0)
        mats['UC'] = UC
    except KeyError as e:
        print(f"Skipping UC due to missing parameter: {e}")

    # UCO (UO2 + UC mixture)
    # mix_materials cannot accept materials that already have S(α,β) tables,
    # so clean intermediate copies are used here; TSLs are added post-mix.
    try:
        UO2_for_mix = openmc.Material(name='UO2_for_mix')
        UO2_for_mix.set_density('g/cm3', 10.41)
        UO2_for_mix.add_element('U', 1.0, enrichment=100 * params['Enrichment'])
        UO2_for_mix.add_nuclide('O16', 2.0)

        UC_for_mix = openmc.Material(name='UC_for_mix')
        UC_for_mix.set_density('g/cm3', 13.0)
        UC_for_mix.add_element('U', 1.0, enrichment=100 * params['Enrichment'])
        UC_for_mix.add_element('C', 1.0)

        UCO = openmc.Material.mix_materials(
            [UO2_for_mix, UC_for_mix],
            [params['UO2 atom fraction'], 1 - params['UO2 atom fraction']],
            'ao', name='UCO')
        UCO.temperature = params['Common Temperature']
        mats['UCO'] = UCO
    except (KeyError, NameError) as e:
        print(f"Skipping UCO due to missing parameter/material: {e}")

    # Uranium Nitride
    try:
        UN = openmc.Material(name='UN')
        UN.set_density('g/cm3', 14.0)
        UN.add_element('U', 1.0, enrichment=100 * params['Enrichment'])
        UN.add_element('N', 1.0)
        mats['UN'] = UN
    except KeyError as e:
        print(f"Skipping UN due to missing parameter: {e}")

    # U-10Zr
    try:
        UZr = openmc.Material(name='UZr')
        UZr.set_density('g/cm3', 16.0)
        UZr.add_element('U', 10, 'wo', enrichment=100 * params['Enrichment'])
        UZr.add_element('Zr', 90, 'wo')
        mats['UZr'] = UZr
    except KeyError as e:
        print(f"Skipping U-10Zr due to missing parameter: {e}")

    # Homogenized TRISO fuel
    try:
        U_total  = 0.00130037929
        density  = 8.08250295E-02
        U235_frac = params['Enrichment'] * U_total
        U238_frac = (1 - params['Enrichment']) * U_total
        homog_TRISO = openmc.Material(name='homog_TRISO')
        homog_TRISO.set_density('atom/b-cm', density)
        homog_TRISO.temperature = params['Common Temperature']
        homog_TRISO.add_nuclide('U235', U235_frac, 'ao')
        homog_TRISO.add_nuclide('U238', U238_frac, 'ao')
        homog_TRISO.add_nuclide('O16',  2.59371545E-03, 'ao')
        homog_TRISO.add_nuclide('O17',  1.05004397E-06, 'ao')
        homog_TRISO.add_nuclide('O18',  5.99797186E-06, 'ao')
        homog_TRISO.add_nuclide('Si28', 2.76954169E-03, 'ao')
        homog_TRISO.add_nuclide('Si29', 1.40694868E-04, 'ao')
        homog_TRISO.add_nuclide('Si30', 9.28556098E-05, 'ao')
        homog_TRISO.add_nuclide('C12',  7.31619752E-02, 'ao')
        homog_TRISO.add_nuclide('C13',  7.58819416E-04, 'ao')
        mats['homog_TRISO'] = homog_TRISO
    except KeyError as e:
        print(f"Skipping homog_TRISO due to missing parameter: {e}")

    # ------------------------------------------------------------------
    # Sec. 1.2 : Hydrides
    # ------------------------------------------------------------------

    ZrH = openmc.Material(name="ZrH", temperature=params['Common Temperature'])
    ZrH.set_density("g/cm3", 5.6)
    ZrH.add_nuclide("H1", 1.85)
    ZrH.add_element("zirconium", 1.0)
    mats['ZrH'] = ZrH

    YHx = openmc.Material(name="YHx")
    YHx.set_density("g/cm3", 4.28)
    YHx.add_nuclide("H1", 1.5)
    YHx.add_element("yttrium", 1.0)
    YHx.temperature = params['Common Temperature']
    mats['YHx'] = YHx

    # ------------------------------------------------------------------
    # Sec. 1.3 : Coolants
    # ------------------------------------------------------------------

    NaK = openmc.Material(name="NaK", temperature=params['Common Temperature'])
    NaK.set_density("g/cm3", 0.85)
    NaK.add_nuclide("Na23", 2.20000e-01)
    NaK.add_nuclide("K39",  7.27413e-01)
    NaK.add_nuclide("K41",  5.24956e-02)
    mats['NaK'] = NaK

    Helium = openmc.Material(name='Helium')
    Helium.set_density('g/cm3', 0.000166)
    Helium.temperature = params['Common Temperature']
    Helium.add_element('He', 1.0)
    mats['Helium'] = Helium

    # ------------------------------------------------------------------
    # Sec. 1.4 : Beryllium and Beryllium Oxide
    # ------------------------------------------------------------------

    Be = openmc.Material(name="Be")
    Be.add_element("beryllium", 1.0)
    Be.set_density("g/cm3", 1.84)
    Be.temperature = params['Common Temperature']
    mats['Be'] = Be

    BeO = openmc.Material(name="BeO", temperature=params['Common Temperature'])
    BeO.set_density("g/cm3", 3.01)
    BeO.add_element("beryllium", 1.0)
    BeO.add_element("oxygen", 1.0)
    mats['BeO'] = BeO

    # ------------------------------------------------------------------
    # Sec. 1.5 : Zirconium
    # ------------------------------------------------------------------

    Zr = openmc.Material(name="Zr", temperature=params['Common Temperature'])
    Zr.set_density("g/cm3", 6.49)
    Zr.add_element("zirconium", 1.0)
    mats['Zr'] = Zr

    # ------------------------------------------------------------------
    # Sec. 1.6 : SS304
    # ------------------------------------------------------------------

    SS304 = openmc.Material(name="SS304", temperature=params['Common Temperature'])
    SS304.set_density("g/cm3", 7.98)
    SS304.add_element("carbon",     0.04,   "wo")
    SS304.add_element("silicon",    0.50,   "wo")
    SS304.add_element("phosphorus", 0.023,  "wo")
    SS304.add_element("sulfur",     0.015,  "wo")
    SS304.add_element("chromium",   19.00,  "wo")
    SS304.add_element("manganese",  1.00,   "wo")
    SS304.add_element("iron",       70.173, "wo")
    SS304.add_element("nickel",     9.25,   "wo")
    mats['SS304'] = SS304

    # ------------------------------------------------------------------
    # Sec. 1.7 : Carbides
    # ------------------------------------------------------------------

    B4C_natural = openmc.Material(name="B4C_natural", temperature=params['Common Temperature'])
    B4C_natural.add_element("boron",  4)
    B4C_natural.add_element("carbon", 1)
    B4C_natural.set_density("g/cm3", 2.52)
    mats['B4C_natural'] = B4C_natural

    B4C_enriched = openmc.Material(name="B4C_enriched", temperature=params['Common Temperature'])
    B4C_enriched.add_element("boron", 4, enrichment=95.0,
                             enrichment_target='B10', enrichment_type='ao')
    B4C_enriched.add_element("carbon", 1)
    B4C_enriched.set_density("g/cm3", 2.52)
    mats['B4C_enriched'] = B4C_enriched

    SiC = openmc.Material(name='SiC')
    SiC.set_density('g/cm3', 3.18)
    SiC.add_element('Si', 0.5)
    SiC.add_element('C',  0.5)
    mats['SiC'] = SiC

    ZrC = openmc.Material(name='ZrC')
    ZrC.set_density('g/cm3', 6.73)
    ZrC.add_element('Zr', 1.0)
    ZrC.add_element('C',  1.0)
    mats['ZrC'] = ZrC

    # ------------------------------------------------------------------
    # Sec. 1.8 : Carbon-based materials
    # ------------------------------------------------------------------

    Graphite = openmc.Material(name='Graphite')
    Graphite.set_density('g/cm3', 1.60)
    Graphite.add_element('C', 1.0)
    mats['Graphite'] = Graphite

    buffer_graphite = openmc.Material(name='Buffer')
    buffer_graphite.set_density('g/cm3', 0.95)
    buffer_graphite.add_element('C', 1.0)
    mats['buffer_graphite'] = buffer_graphite

    PyC = openmc.Material(name='PyC')
    PyC.set_density('g/cm3', 1.9)
    PyC.add_element('C', 1.0)
    mats['PyC'] = PyC

    # ------------------------------------------------------------------
    # Sec. 1.9 : Magnesium Oxide
    # ------------------------------------------------------------------

    MgO = openmc.Material(name='MgO')
    MgO.set_density('g/cm3', 3.58)
    MgO.add_element('Mg', 1.0)
    MgO.add_element('O',  1.0)
    mats['MgO'] = MgO

    # ------------------------------------------------------------------
    # Sec. 1.10 : Tungsten-based materials
    # ------------------------------------------------------------------

    WB = openmc.Material(name='WB')
    WB.set_density('g/cm3', 15.43)
    WB.add_element('W', 1.0)
    WB.add_element('B', 1.0)
    mats['WB'] = WB

    W2B = openmc.Material(name='W2B')
    W2B.set_density('g/cm3', 16.75)  # doi.org/10.1016/j.jnucmat.2020.152062
    W2B.add_element('W', 2.0)
    W2B.add_element('B', 1.0)
    mats['W2B'] = W2B

    WB4 = openmc.Material(name='WB4')
    WB4.set_density('g/cm3', 8.23)
    WB4.add_element('W', 1.0)
    WB4.add_element('B', 4.0)
    mats['WB4'] = WB4

    WC = openmc.Material(name='WC')
    WC.set_density('g/cm3', 15.32)
    WC.add_element('W', 1.0)
    WC.add_element('C', 1.0)
    mats['WC'] = WC

    # ------------------------------------------------------------------
    # Sec. 1.11 : Heat Pipe Microreactor
    # ------------------------------------------------------------------

    heatpipe = openmc.Material(name='heatpipe')
    heatpipe.set_density('atom/b-cm', 2.74917E-02)
    heatpipe.temperature = params['Common Temperature']
    heatpipe.add_nuclide('Si28',  1.49701E-02, 'ao')
    heatpipe.add_nuclide('Si29',  7.60143E-04, 'ao')
    heatpipe.add_nuclide('Si30',  5.01090E-04, 'ao')
    heatpipe.add_nuclide('Cr50',  6.46763E-03, 'ao')
    heatpipe.add_nuclide('Cr52',  1.24724E-01, 'ao')
    heatpipe.add_nuclide('Cr53',  1.41423E-02, 'ao')
    heatpipe.add_nuclide('Cr54',  3.52029E-03, 'ao')
    heatpipe.add_nuclide('Mn55',  1.66133E-02, 'ao')
    heatpipe.add_nuclide('Fe54',  3.12186E-02, 'ao')
    heatpipe.add_nuclide('Fe56',  4.90061E-01, 'ao')
    heatpipe.add_nuclide('Fe57',  1.13180E-02, 'ao')
    heatpipe.add_nuclide('Fe58',  1.50617E-03, 'ao')
    heatpipe.add_nuclide('Ni58',  6.33738E-02, 'ao')
    heatpipe.add_nuclide('Ni60',  2.44119E-02, 'ao')
    heatpipe.add_nuclide('Ni61',  1.06115E-03, 'ao')
    heatpipe.add_nuclide('Ni62',  3.38338E-03, 'ao')
    heatpipe.add_nuclide('Ni64',  8.61654E-04, 'ao')
    heatpipe.add_nuclide('Mo92',  1.75699E-03, 'ao')
    heatpipe.add_nuclide('Mo94',  1.09514E-03, 'ao')
    heatpipe.add_nuclide('Mo95',  1.88484E-03, 'ao')
    heatpipe.add_nuclide('Mo96',  1.97478E-03, 'ao')
    heatpipe.add_nuclide('Mo97',  1.13066E-03, 'ao')
    heatpipe.add_nuclide('Mo98',  2.85681E-03, 'ao')
    heatpipe.add_nuclide('Mo100', 1.14011E-03, 'ao')
    heatpipe.add_nuclide('Na23',  1.79266E-01, 'ao')
    mats['heatpipe'] = heatpipe

    monolith_graphite = openmc.Material(name='monolith_graphite')
    monolith_graphite.set_density('g/cm3', 1.63)
    monolith_graphite.temperature = params['Common Temperature']
    monolith_graphite.add_nuclide('C12', 0.9893, 'ao')
    monolith_graphite.add_nuclide('C13', 0.0107, 'ao')
    mats['monolith_graphite'] = monolith_graphite

    return mats


# ==================================================================================
#  ENDF/B-VIII.0 builder
# ==================================================================================

def _collect_materials_endf80(params):
    """Build materials with ENDF/B-VIII.0 S(α,β) tables.

    TSL coverage in VIII.0:
      - c_H_in_ZrH, c_H_in_YH2               (hydrides, H side only)
      - c_U_in_UO2, c_O_in_UO2               (UO2)
      - c_U_in_UN,  c_N_in_UN                (UN)
      - c_C_in_SiC                            (SiC, C side only)
      - c_Graphite                            (all graphite variants)
      - c_Be, c_Be_in_BeO, c_O_in_BeO        (beryllium)
    """
    print("Reading the Materials Database")
    mats = _build_base_materials(params)
    materials = openmc.Materials()

    # --- UZrH alloy ---
    if 'UZrH_alloy' in mats:
        mats['UZrH_alloy'].add_s_alpha_beta("c_H_in_ZrH")
        materials.append(mats['UZrH_alloy'])

    # --- UO2 ---
    if 'UO2' in mats:
        mats['UO2'].add_s_alpha_beta("c_U_in_UO2")
        mats['UO2'].add_s_alpha_beta("c_O_in_UO2")
        materials.append(mats['UO2'])

    # --- UC (no TSL in VIII.0) ---
    if 'UC' in mats:
        materials.append(mats['UC'])

    # --- UCO (approximation: UO2 TSLs only) ---
    if 'UCO' in mats:
        mats['UCO'].add_s_alpha_beta("c_U_in_UO2")
        mats['UCO'].add_s_alpha_beta("c_O_in_UO2")
        materials.append(mats['UCO'])

    # --- UN ---
    if 'UN' in mats:
        mats['UN'].add_s_alpha_beta("c_U_in_UN")
        mats['UN'].add_s_alpha_beta("c_N_in_UN")
        materials.append(mats['UN'])

    # --- U-10Zr (no TSL) ---
    if 'UZr' in mats:
        materials.append(mats['UZr'])

    # --- homog_TRISO ---
    if 'homog_TRISO' in mats:
        mats['homog_TRISO'].add_s_alpha_beta('c_Graphite')
        materials.append(mats['homog_TRISO'])

    # --- ZrH (H side only in VIII.0) ---
    mats['ZrH'].add_s_alpha_beta("c_H_in_ZrH")
    materials.append(mats['ZrH'])

    # --- YHx (H side only in VIII.0) ---
    mats['YHx'].add_s_alpha_beta("c_H_in_YH2")
    materials.append(mats['YHx'])

    # --- Coolants (no TSL) ---
    materials.extend([mats['NaK'], mats['Helium']])

    # --- Beryllium ---
    mats['Be'].add_s_alpha_beta("c_Be")
    mats['BeO'].add_s_alpha_beta("c_Be_in_BeO")
    mats['BeO'].add_s_alpha_beta("c_O_in_BeO")
    materials.extend([mats['Be'], mats['BeO']])

    # --- Zr, SS304 (no TSL) ---
    materials.extend([mats['Zr'], mats['SS304']])

    # --- Carbides ---
    mats['SiC'].add_s_alpha_beta("c_C_in_SiC")   # C side only in VIII.0
    # ZrC: no TSL in VIII.0
    materials.extend([mats['B4C_natural'], mats['B4C_enriched'], mats['SiC']])

    # --- Graphite family ---
    for key in ('Graphite', 'buffer_graphite', 'PyC'):
        mats[key].add_s_alpha_beta('c_Graphite')
        materials.append(mats[key])

    # --- MgO (no TSL in VIII.0) ---
    # c_Mg_in_MgO and c_O_in_MgO are new in VIII.1

    # --- Tungsten (no TSL) ---
    # --- Heatpipe + monolith graphite ---
    mats['monolith_graphite'].add_s_alpha_beta('c_Graphite')
    materials.extend([mats['heatpipe'], mats['monolith_graphite']])

    return mats


# ==================================================================================
#  ENDF/B-VIII.1 builder
# ==================================================================================

def _collect_materials_endf81(params):
    """Build materials with ENDF/B-VIII.1 S(α,β) tables.

    New/updated TSLs versus VIII.0:
      - c_Zr_in_ZrH                           (ZrH, Zr sublattice)
      - c_Y_in_YH2                            (YHx, Y sublattice)
      - c_U_metal[_suffix]                    (U metal, new)
      - c_U_in_UC[_suffix], c_C_in_UC[_suffix](UC, new)
      - c_Zr_in_ZrC, c_C_in_ZrC              (ZrC, new)
      - c_Si_in_SiC                           (SiC, Si sublattice)
      - c_Mg_in_MgO, c_O_in_MgO              (MgO, new)
      - enrichment-specific variants for UO2, UC, UN, U_metal
    """
    print("Reading the Materials Database")
    mats = _build_base_materials(params)
    materials = openmc.Materials()

    # Enrichment-specific suffix for uranium fuel TSLs
    sfx = _enrich_tsl_suffix(params.get('Enrichment', 0.0))

    # --- UZrH alloy ---
    if 'UZrH_alloy' in mats:
        # Note: c_U_metal TSL is on U_met before mixing; post-mix we add ZrH TSLs.
        # mix_materials cannot accept materials with S(α,β) tables, so U_met TSL
        # is intentionally skipped here — the mixed material picks up ZrH TSLs only.
        mats['UZrH_alloy'].add_s_alpha_beta("c_H_in_ZrH")
        mats['UZrH_alloy'].add_s_alpha_beta("c_Zr_in_ZrH")       # new in VIII.1
        materials.append(mats['UZrH_alloy'])

    # --- UO2 ---
    if 'UO2' in mats:
        mats['UO2'].add_s_alpha_beta("c_U_in_UO2" + sfx)         # enrichment-specific
        mats['UO2'].add_s_alpha_beta("c_O_in_UO2" + sfx)         # enrichment-specific
        materials.append(mats['UO2'])

    # --- UC ---
    if 'UC' in mats:
        mats['UC'].add_s_alpha_beta("c_U_in_UC" + sfx)           # new in VIII.1, enrichment-specific
        mats['UC'].add_s_alpha_beta("c_C_in_UC" + sfx)           # new in VIII.1, enrichment-specific
        materials.append(mats['UC'])

    # --- UCO (approximation: UO2 TSLs for the oxide fraction) ---
    if 'UCO' in mats:
        mats['UCO'].add_s_alpha_beta("c_U_in_UO2" + sfx)
        mats['UCO'].add_s_alpha_beta("c_O_in_UO2" + sfx)
        materials.append(mats['UCO'])

    # --- UN ---
    if 'UN' in mats:
        mats['UN'].add_s_alpha_beta("c_U_in_UN" + sfx)           # enrichment-specific
        mats['UN'].add_s_alpha_beta("c_N_in_UN" + sfx)           # enrichment-specific
        materials.append(mats['UN'])

    # --- U-10Zr (no TSL available) ---
    if 'UZr' in mats:
        materials.append(mats['UZr'])

    # --- homog_TRISO ---
    if 'homog_TRISO' in mats:
        mats['homog_TRISO'].add_s_alpha_beta('c_Graphite')
        materials.append(mats['homog_TRISO'])

    # --- ZrH ---
    mats['ZrH'].add_s_alpha_beta("c_H_in_ZrH")
    mats['ZrH'].add_s_alpha_beta("c_Zr_in_ZrH")                  # new in VIII.1
    materials.append(mats['ZrH'])

    # --- YHx ---
    mats['YHx'].add_s_alpha_beta("c_H_in_YH2")
    mats['YHx'].add_s_alpha_beta("c_Y_in_YH2")                   # new in VIII.1
    materials.append(mats['YHx'])

    # --- Coolants (no TSL) ---
    materials.extend([mats['NaK'], mats['Helium']])

    # --- Beryllium ---
    mats['Be'].add_s_alpha_beta("c_Be")
    mats['BeO'].add_s_alpha_beta("c_Be_in_BeO")
    mats['BeO'].add_s_alpha_beta("c_O_in_BeO")
    materials.extend([mats['Be'], mats['BeO']])

    # --- Zr, SS304 (no TSL) ---
    materials.extend([mats['Zr'], mats['SS304']])

    # --- Carbides ---
    mats['SiC'].add_s_alpha_beta("c_C_in_SiC")
    mats['SiC'].add_s_alpha_beta("c_Si_in_SiC")                  # new in VIII.1
    mats['ZrC'].add_s_alpha_beta("c_Zr_in_ZrC")                  # new in VIII.1
    mats['ZrC'].add_s_alpha_beta("c_C_in_ZrC")                   # new in VIII.1
    materials.extend([mats['B4C_natural'], mats['B4C_enriched'], mats['SiC']])

    # --- Graphite family ---
    for key in ('Graphite', 'buffer_graphite', 'PyC'):
        mats[key].add_s_alpha_beta('c_Graphite')
        materials.append(mats[key])

    # --- MgO ---
    mats['MgO'].add_s_alpha_beta("c_Mg_in_MgO")                  # new in VIII.1
    mats['MgO'].add_s_alpha_beta("c_O_in_MgO")                   # new in VIII.1

    # --- Tungsten (no TSL) ---
    # --- Heatpipe + monolith graphite ---
    mats['monolith_graphite'].add_s_alpha_beta('c_Graphite')
    materials.extend([mats['heatpipe'], mats['monolith_graphite']])

    return mats