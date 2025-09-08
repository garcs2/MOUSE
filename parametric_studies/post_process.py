import numpy as np
np.random.seed(23)
import pandas as pd
import joblib
from pathlib import Path
import os,sys
import matplotlib.pyplot as plt
# Add relative path to sys.path to be able to call the external scripts as import
# ---
sys.path.append(os.path.dirname(os.path.abspath('.')))
#print(os.path.dirname(os.path.abspath('.')),sys.path)
#
from OpenMC_GCMR import OpenMC_GCMR


if __name__ == "__main__":

    if sys.argv[1] == 'GCMR':
        tracked_params_list =     ["Particles", "Number Of TRISO Particles Per Compact Fuel",
            "Total Number of TRISO Particles","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass",
            'TRISO Fueled','Fuel','Enrichment','Lattice Pitch','Reflector','Power MWt','Assembly Rings','Reflector Thickness',
            'Operation Mode','Emergency Shutdowns Per Year','Moderator Booster','Moderator Booster Radius',#'Particles',
            'LCOE_FOAK Estimated Cost','LCOE_NOAK Estimated Cost','AC_FOAK Estimated Cost','AC_NOAK Estimated Cost','TCI_FOAK Estimated Cost','TCI_NOAK Estimated Cost',
            'LCOE_FOAK Estimated Cost std','LCOE_NOAK Estimated Cost std','AC_FOAK Estimated Cost std','AC_NOAK Estimated Cost std','TCI_FOAK Estimated Cost std','TCI_NOAK Estimated Cost std']
    elif sys.argv[1] == 'LTMR':
        tracked_params_list =     ['U_met_wo',"Number of Rings per Assembly","Moderator","Moderator Pin Inner Radius","Core Radius", "Heat Flux","Fuel Lifetime", "Mass U235", "Mass U238", "Uranium Mass",
            'TRISO Fueled','Fuel','Enrichment','Reflector','Power MWt','Reflector Thickness',
            'Operation Mode','Emergency Shutdowns Per Year',#'Particles',
            'LCOE_FOAK Estimated Cost','LCOE_NOAK Estimated Cost','AC_FOAK Estimated Cost','AC_NOAK Estimated Cost','TCI_FOAK Estimated Cost','TCI_NOAK Estimated Cost',
            'LCOE_FOAK Estimated Cost std','LCOE_NOAK Estimated Cost std','AC_FOAK Estimated Cost std','AC_NOAK Estimated Cost std','TCI_FOAK Estimated Cost std','TCI_NOAK Estimated Cost std']
    else:
        print('---error: Option is either "GCMR" or "LTRM" not {}'.format(sys.argv[1]))
        sys.exit()
    sub_dir = sys.argv[2]
    finished = []
    finished_tags = []
    for f in Path(sub_dir).iterdir():
        folderindex = f.name[7:]
        if (f / Path('output_{}_{}.csv'.format(sys.argv[1],folderindex))).exists():
            finished.append(f)
            finished_tags.append(f.name.split("_")[1])
        elif f.is_dir():
            print(f, "unfinished or failed")        
    df = pd.DataFrame(np.zeros((len(finished_tags), len(tracked_params_list))), index = finished_tags, columns = tracked_params_list)
    for f,t in zip(finished, finished_tags):
        #fill in design inputs
        folderindex = f.name[7:]
        df_params = pd.read_csv(f / Path('output_{}_{}.csv'.format(sys.argv[1],folderindex)))    
        design_inputs = df_params.loc[:,tracked_params_list]
        df.loc[t, :] = design_inputs.values[0]
    
    df['Core Height'] = 2*df['Core Radius']
    df['Core diameter'] = 2*df['Core Radius']
    df.to_csv(Path('./{}'.format(sub_dir)) / Path("postproc.csv"))

    