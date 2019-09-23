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


def plot_inc_dec_ratio(path_postfix):
    cur_path = root.joinpath("results", path_postfix)
    projects = [d for d in cur_path.iterdir() if d.is_dir()]

    def reshape_dframe(name):
        # -- dframe --
        dframe = pd.read_csv(name, index_col=False)
        # -- Sort the values --
        dframe = dframe.sort_values(['Overlap', 'Method'])
        # -- Group by mean --
        dframe = dframe.groupby(['Overlap', 'Method']).mean()
        # -- Covert to integer --
        dframe['Num'] = dframe['Num'].astype('int')
        df = pd.DataFrame(dframe.values.reshape(
            (1, -1)), columns=dframe.index)
        return df[100].values[0]

    all_results = []
    for proj in projects:
        for dec_data in proj.glob("*_dec.csv"):
            inc_data = next(proj.glob(dec_data.name[:-7] + 'inc.csv'))

            #
            dframe_inc = reshape_dframe(inc_data)
            dframe_dec = reshape_dframe(dec_data)

            #
            fname = dec_data.name[:-8]
            fname = re.sub(" v", "-", fname)
            all_results.append([fname, dframe_dec[-1], dframe_inc[-1]])
    all_results = pd.DataFrame(all_results, columns=['Name', 'Dec', 'Inc'])
    all_results.set_index('Name')
    all_results.to_csv('ALL_RES.csv', index=False)


if __name__ == "__main__":
    for postfix in ['RQ1']:
        plot_inc_dec_ratio(path_postfix=postfix)

        # plot_results(path_postfix=postfix)
    #     plot_overlap_counts(path_postfix=postfix)
