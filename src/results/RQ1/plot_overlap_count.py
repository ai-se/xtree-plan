import os
import sys
import numpy
from pathlib import Path

from pdb import set_trace
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

import pandas as pd
from utils.plot_util_clean import plot_violin, plot_catplot
import re

if __name__ == "__main__":
    cur_path = root.joinpath("results/RQ1/")
    projects = [d for d in cur_path.iterdir() if d.is_dir()]
    for proj in projects:
        for data in proj.glob("*_counts.csv"):
            dframe = pd.read_csv(data, index_col=False)
            dframe.loc[dframe['Overlap'] == 50, 'Overlap'] = 77
            dframe.loc[dframe['Overlap'] == 75, 'Overlap'] = 50
            dframe.loc[dframe['Overlap'] == 77, 'Overlap'] = 75
            fname = re.sub(" ", "_", data.name[: -4])
            plot_catplot(dframe, save_path=proj, title=fname, y_lbl="Counts", postfix="")
