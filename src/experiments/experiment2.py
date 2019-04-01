"""
Experiment 1: Measure Overlap of XTREE with Developer changes
"""
import os
import sys
from pdb import set_trace
import numpy
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

import pandas as pd
from data.get_data import get_all_projects
from planners.XTREE import XTREE
from planners.alves import alves
from planners.shatnawi import shatnawi
from planners.oliveira import oliveira
from utils.file_util import list2dataframe
from utils.rq_utils import measure_overlap, reshape
from utils.plot_util_clean import plot_violin, plot_catplot
from utils.stats_utils.auec import compute_auec
from utils.rq_utils import measure_overlap, reshape, reshape_overlap
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class Experiment2:
    def __init__(self, bellwether, plot_results=True, decrease=True, verbose=True):
        self.bellwether = bellwether
        self.plot_results = plot_results
        self.decrease = decrease
        self.verbose = verbose

    def main(self, proj, paths):
        i = 0
        train = self.bellwether
        for test, validation in zip(paths.data[:-1], paths.data[1:]):
            i += 1

            # -- Save names and save paths --
            plot_title = "{} v{}".format(proj, i)
            save_path = os.path.join(root, "results", "RQ2", proj)
            if self.verbose:
                print(plot_title)

            # -- Create dataframes to hold the results --
            columns = ["Overlap", "Num", "Method"]
            decrease = pd.DataFrame(columns=columns)
            increase = pd.DataFrame(columns=columns)
            counts = pd.DataFrame(columns=columns)

            # -- Create a dataframe for test and validation --
            test_df_full = pd.read_csv(test)
            train_df = list2dataframe(train.data)
            valdn_df = pd.read_csv(validation)

            # -- Binarize training data labels --
            train_df.loc[train_df[train_df.columns[-1]] >
                         0, train_df.columns[-1]] = 1

            # ------------------------------------------------------------------
            # -- Repeat 10 times with 90% samples --
            for repeats in range(10):
                # -- Split the test data --
                test_df, __ = train_test_split(
                    test_df_full, test_size=0.1, random_state=1729)

                # -- Build an XTREE Model --
                xtree_arplan = XTREE(strategy="itemset")
                xtree_arplan = xtree_arplan.fit(train_df)

                # -- Generate Plans --
                patched_xtree = xtree_arplan.predict(test_df)
                patched_alves = alves(train_df[train_df.columns[1:]], test)
                patched_shatw = shatnawi(train_df[train_df.columns[1:]], test)
                patched_olive = oliveira(train_df[train_df.columns[1:]], test)

                # -- Compute overlap with developers changes --
                overlap_xtree, res_xtree = measure_overlap(
                    test_df, patched_xtree, valdn_df)
                overlap_alves, res_alves = measure_overlap(
                    test_df, patched_alves, valdn_df)
                overlap_shatw, res_shatw = measure_overlap(
                    test_df, patched_shatw, valdn_df)
                overlap_olive, res_olive = measure_overlap(
                    test_df, patched_olive, valdn_df)

                # -- Summary of defects decreased/increased --
                res_dec, res_inc = reshape(
                    res_xtree, res_alves, res_shatw, res_olive)

                # -- Summary of Overlap counts --
                overlap_counts = reshape_overlap(
                    overlap_xtree, overlap_alves, overlap_shatw, overlap_olive)

                decrease = decrease.append(res_dec, ignore_index=True)
                increase = increase.append(res_inc, ignore_index=True)
                counts = counts.append(overlap_counts, ignore_index=True)

            decrease.to_csv(os.path.join(
                save_path, plot_title + "_dec.csv"), index=False)
            increase.to_csv(os.path.join(
                save_path, plot_title + "_inc.csv"), index=False)
            counts.to_csv(os.path.join(
                save_path, plot_title + "_counts.csv"), index=False)

            # ------------------------------------------------------------------
            # -- Plot the results --
            if self.plot_results:
                # --- Violin plots ---
                # -- Decreased --
                plot_catplot(decrease, save_path=save_path, title=plot_title,
                             y_lbl="# Defects Removed", postfix="dec")
                # -- Increased --
                plot_catplot(increase, save_path=save_path, title=plot_title,
                             y_lbl="# Defects Added", postfix="inc")


if __name__ == "__main__":
    exp = Experiment1()
    exp.main()
