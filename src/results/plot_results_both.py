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


def plot_results(path_postfix):
    cur_path = root.joinpath("results", path_postfix)
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


def plot_overlap_counts(path_postfix):
    cur_path = root.joinpath("results", path_postfix)
    projects = [d for d in cur_path.iterdir() if d.is_dir()]
    for proj in projects:
        for data in proj.glob("*_counts.csv"):
            dframe = pd.read_csv(data, index_col=False)
            dframe.loc[dframe['Overlap'] == 50, 'Overlap'] = 77
            dframe.loc[dframe['Overlap'] == 75, 'Overlap'] = 50
            dframe.loc[dframe['Overlap'] == 77, 'Overlap'] = 75
            fname = re.sub(" ", "_", data.name[: -4])
            plot_catplot(dframe, save_path=proj, title=fname,
                         y_lbl="Counts", postfix="")


if __name__ == "__main__":
    for postfix in ['RQ1', 'RQ2']:
        plot_results(path_postfix=postfix)
        plot_overlap_counts(path_postfix=postfix)
