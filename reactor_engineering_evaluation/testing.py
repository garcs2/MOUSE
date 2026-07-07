from core_design_3D.openmc_materials_database_3D import collect_materials_data
p = {'XS_type':'endf8.0','Enrichment':0.1975,
     'Common Temperature':600,'Reference Temperature':293.15,
     'Thermal Expansion':False,'Per-Region Temperatures':True,
     'Fuel':'UO2','Reflector':'BeO','Coolant':'NaK','Moderator':'ZrH',
     'UO2 atom fraction':0.75,
     'Fuel Temperature':600,'Reflector Temperature':700,'Coolant Temperature':600}
mats = collect_materials_data(p)
for n in ('UO2','BeO','NaK','ZrH'):
    if n in mats: print(f'{n:5s} T={mats[n].temperature}  rho={mats[n].density}')