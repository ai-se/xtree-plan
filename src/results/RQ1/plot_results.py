import re
import pandas as pd
import os
import sys
from pdb import set_trace
import numpy
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

from utils.plot_util_clean import plot_catplot

if __name__ == "__main__":
    cur_path = root.joinpath("results/RQ1/")
    projects = [d for d in cur_path.iterdir() if d.is_dir()]
    for proj in projects:
        for data in proj.glob("*.csv"):
            dframe = pd.read_csv(data, index_col=False)
            dframe.loc[dframe['Overlap'] == 50, 'Overlap'] = 77
            dframe.loc[dframe['Overlap'] == 75, 'Overlap'] = 50
            dframe.loc[dframe['Overlap'] == 77, 'Overlap'] = 75
            fname = re.sub(" ", "_", data.name[: -4])
            if 'dec' in fname:
                # -- Decreased --
                plot_catplot(dframe, save_path=proj, title=fname,
                             y_lbl="# Defects Removed", postfix="")
            else:
                # -- Increased --
                plot_catplot(dframe, save_path=proj, title=fname,
                             y_lbl="# Defects Added", postfix="")
