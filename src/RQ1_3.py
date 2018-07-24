"""
Compare XTREE with other threshold based learners.
"""

from __future__ import print_function, division

import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from planners.XTREE import xtree
from planners.alves import alves
from utils.plot_util import plot_bar
from planners.shatnawi import shatnawi
from planners.oliveira import oliveira
from data.get_data import get_all_projects
from utils.file_util import list2dataframe
from utils.stats_utils.auec import compute_auec
from utils.rq_utils import measure_overlap, reshape


def research_question_1_3(decrease=True, verbose=True, plot_results=True):
    """
    RQ1: How effective is XTREE?
    RQ3: How does XTREE compare with BELLTREE? (The XTREE part of this RQ is answered here)

    Parameters
    ----------
    decrease: Bool
        Compute AUPEC for defects reduced.
    verbose: Bool
        Display results on the console
    plot_results: Bool
        Save barcharts of overlap vs. defects increased/decreased
    """

    data = get_all_projects()
    if verbose:
        print("Data \tXTREE\tAlves\tShatw\tOlive")

    for proj, paths in data.iteritems():
        i = 0

        for train, test, validation in zip(paths.data[:-2], paths.data[1:-1], paths.data[2:]):
            i += 1

            "Convert to pandas type dataframe"
            train = list2dataframe(train)
            test = list2dataframe(test)
            validation = list2dataframe(validation)

            "Recommend changes with XTREE"
            patched_xtree = xtree(train[train.columns[1:]], test)
            patched_alves = alves(train[train.columns[1:]], test)
            patched_shatw = shatnawi(train[train.columns[1:]], test)
            patched_olive = oliveira(train[train.columns[1:]], test)

            "Compute overlap with developers changes"
            res_xtree = measure_overlap(test, patched_xtree, validation)
            res_alves = measure_overlap(test, patched_alves, validation)
            res_shatw = measure_overlap(test, patched_shatw, validation)
            res_olive = measure_overlap(test, patched_olive, validation)

            "AUPEC of defects decreased/increased"
            res_dec, res_inc = reshape(res_xtree, res_alves, res_shatw, res_olive)

            "Plot the results"
            if plot_results:
                plot_bar(res_inc, res_dec, save_path=os.path.join(
                    root, "results", "RQ1", proj), title="{} v{}".format(proj, i), y_lbl="Defects",
                             postfix="")


            "Max/Min to normalize AUPEC"
            y_max = max(res_dec.max(axis=0).values)
            y_min = max(res_dec.min(axis=0).values)

            if decrease:
                "Decrease AUC"
                xtree_dec_auc = compute_auec(res_dec[["Overlap", "XTREE"]], y_max, y_min)
                alves_dec_auc = compute_auec(res_dec[["Overlap", "Alves"]], y_max, y_min)
                shatw_dec_auc = compute_auec(res_dec[["Overlap", "Shatnawi"]], y_max, y_min)
                olive_dec_auc = compute_auec(res_dec[["Overlap", "Oliveira"]], y_max, y_min)

                if verbose:
                    print("{}-{}\t{}\t{}\t{}\t{}".format(proj[:3], i, xtree_dec_auc, alves_dec_auc, shatw_dec_auc, olive_dec_auc))

            else:
                "Increase AUC"
                xtree_inc_auc = compute_auec(res_inc[["Overlap", "XTREE"]], y_max, y_min)
                alves_inc_auc = compute_auec(res_inc[["Overlap", "Alves"]], y_max, y_min)
                shatw_inc_auc = compute_auec(res_inc[["Overlap", "Shatnawi"]], y_max, y_min)
                olive_inc_auc = compute_auec(res_inc[["Overlap", "Oliveira"]], y_max, y_min)

                if verbose:
                    print("{}-{}\t{}\t{}\t{}\t{}".format(proj[:3], i, xtree_inc_auc, alves_inc_auc, shatw_inc_auc, olive_inc_auc))


if __name__ == "__main__":
    print("AUPEC: Defects Reduced\n{}".format(22*"-"))
    research_question_1_3(decrease=True, verbose=True, plot_results=False)
    print("\n"+40*"="+"\nAUPEC: Defects Increased\n"+24*"-")
    research_question_1_3(decrease=False, verbose=True, plot_results=False)
