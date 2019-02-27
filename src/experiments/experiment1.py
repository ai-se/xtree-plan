"""
Experiment 1: Measure Overlap of XTREE with Developer changes
"""
import os
import sys
from pdb import set_trace
import pandas as pd
import unittest
import numpy
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

from data.get_data import get_all_projects
from planners.XTREE import XTREE
from planners.alves import alves
from planners.shatnawi import shatnawi
from planners.oliveira import oliveira
from utils.rq_utils import measure_overlap, reshape
from utils.plot_util import plot_bar
from utils.stats_utils.auec import compute_auec
from utils.rq_utils import measure_overlap, reshape


class Experiment1:
    def __init__(self, plot_results=True, decrease=True, verbose=True):
        self.plot_results = plot_results
        self.decrease = decrease
        self.verbose = verbose

    def main(self):
        data = get_all_projects()
        for proj, paths in data.items():
            i = 0
            for train, test, validation in zip(paths.data[:-2], paths.data[1:-1], paths.data[2:]):
                i += 1
                # -- Create a dataframe for train and test --
                test_df = pd.read_csv(test)
                train_df = pd.read_csv(train)
                valdn_df = pd.read_csv(validation)

                # -- Binarize training data labels --
                train_df.loc[train_df[train_df.columns[-1]]
                             > 0, train_df.columns[-1]] = 1

                # -- Build an XTREE Model --
                xtree = XTREE()
                xtree = xtree.fit(train_df)

                # -- Generate Plans --
                patched_xtree = xtree.predict(test_df)
                patched_alves = alves(train_df[train_df.columns[1:]], test)
                patched_shatw = shatnawi(train_df[train_df.columns[1:]], test)
                patched_olive = oliveira(train_df[train_df.columns[1:]], test)

                # -- Compute overlap with developers changes --
                res_xtree = measure_overlap(test_df, patched_xtree, valdn_df)
                res_alves = measure_overlap(test_df, patched_alves, valdn_df)
                res_shatw = measure_overlap(test_df, patched_shatw, valdn_df)
                res_olive = measure_overlap(test_df, patched_olive, valdn_df)

                # -- AUPEC of defects decreased/increased --
                res_dec, res_inc = reshape(
                    res_xtree, res_alves, res_shatw, res_olive)

                # -- Plot the results --
                if self.plot_results:
                    plot_bar(res_inc, res_dec, save_path=os.path.join(
                        root, "results", "RQ1", proj), title="{} v{}".format(proj, i), y_lbl="Defects",
                        postfix="")

                # -- Max/Min to normalize AUPEC --
                y_max = max(res_dec.max(axis=0).values)
                y_min = max(res_dec.min(axis=0).values)

                if self.decrease:
                    # -- Decrease AUC --
                    xtree_dec_auc = compute_auec(
                        res_dec[["Overlap", "XTREE"]], y_max, y_min)
                    alves_dec_auc = compute_auec(
                        res_dec[["Overlap", "Alves"]], y_max, y_min)
                    shatw_dec_auc = compute_auec(
                        res_dec[["Overlap", "Shatnawi"]], y_max, y_min)
                    olive_dec_auc = compute_auec(
                        res_dec[["Overlap", "Oliveira"]], y_max, y_min)

                    if self.verbose:
                        print("{}-{}\t{}\t{}\t{}\t{}".format(
                            proj[:3], i, xtree_dec_auc, alves_dec_auc, shatw_dec_auc, olive_dec_auc))

                else:
                    # -- Increase AUC --
                    xtree_inc_auc = compute_auec(
                        res_inc[["Overlap", "XTREE"]], y_max, y_min)
                    alves_inc_auc = compute_auec(
                        res_inc[["Overlap", "Alves"]], y_max, y_min)
                    shatw_inc_auc = compute_auec(
                        res_inc[["Overlap", "Shatnawi"]], y_max, y_min)
                    olive_inc_auc = compute_auec(
                        res_inc[["Overlap", "Oliveira"]], y_max, y_min)

                    if self.verbose:
                        print("{}-{}\t{}\t{}\t{}\t{}".format(
                            proj[:3], i, xtree_inc_auc, alves_inc_auc, shatw_inc_auc, olive_inc_auc))


if __name__ == "__main__":
    exp = Experiment1()
    exp.main()
