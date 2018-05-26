"""
Compare Bellwether XTREEs with other threshold based learners.
"""

from __future__ import print_function, division

import os
import sys
from pdb import set_trace

# Update path
root = os.path.join(os.getcwd().split(
    'src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from planners.XTREE import xtree
from planners.alves import alves
from planners.shatnawi import shatnawi
from planners.oliveira import oliveira
from data.get_data import get_all_projects
from utils.file_util import list2dataframe
from utils.plot_util import plot_compare, plot_bar
from utils.stats_utils.auec import compute_auec
from utils.stats_utils.ScottKnott import sk_chart
import random
import warnings

warnings.filterwarnings("ignore")

TAXI_CAB = 1729
OVERLAP_RANGE = range(25, 101, 25)
random.seed(TAXI_CAB)


def effectiveness(dframe, thresh):
    overlap = dframe['Overlap']
    heeded = dframe['Heeded']

    a, b, c, d = 0, 0, 0, 0

    for over, heed in zip(overlap, heeded):
        if 0 < over <= thresh and heed >= 0:
            a += 1
        if 0 < over <= thresh and heed < 0:
            b += 1
        if over == 0 and heed < 0:
            c += 1
        if over == 0 and heed >= 0:
            d += 1

    return a, b, c, d


def measure_overlap(test, new, validation):
    results = pd.DataFrame()

    "Find modules that appear both in test and validation datasets"
    common_modules = pd.merge(
        test, validation, how='inner', on=['Name'])['Name']

    "Intitialize variables to hold information"
    improve_heeded = []
    overlap = []

    for module_name in common_modules:
        "For every common class, do the following ... "
        same = 0  # Keep track of features that follow XTREE's recommendations
        # Metric values of classes in the test set
        test_value = test.loc[test['Name'] == module_name]
        # Metric values of classes in the XTREE's planned changes
        plan_value = new.loc[new['Name'] == module_name]
        validate_value = validation.loc[validation['Name']  # Actual metric values the developer's changes yielded
                                        == module_name]

        for col in test_value.columns[1:-1]:
            "For every metric (except the class name and the bugs), do the following ... "
            try:
                if isinstance(plan_value[col].values[0], str):
                    "If the change recommended by XTREE lie's in a range of values"
                    if eval(plan_value[col].values[0])[0] <= validate_value[col].values[0] <= \
                            eval(plan_value[col].values[0])[1]:
                        "If the actual change lies withing the recommended change, then increment the count of heeded values by 1"
                        same += 1
                if isinstance(plan_value[col].values[0], tuple):
                    "If the change recommended by XTREE lie's in a range of values"
                    if plan_value[col].values[0][0] <= validate_value[col].values[0] <= plan_value[col].values[0][1]:
                        "If the actual change lies withing the recommended change, then increment the count of heeded values by 1"
                        same += 1
                elif plan_value[col].values[0] == validate_value[col].values[0]:
                    "If the XTREE recommends no change and developers didn't change anything, that also counts as an overlap"
                    same += 1

            except IndexError:
                "Catch instances where classes don't match"
                pass

        # There are 20 metrics, so, find % of overlap for the class.
        overlap.append(int(same / 20 * 100))
        heeded = test_value['<bug'].values[0] - \
            validate_value['<bug'].values[
            0]  # Find the change in the number of bugs between the test version and the validation version for that class.

        improve_heeded.append(heeded)

    "Save the results ... "

    results['Overlap'] = overlap
    results['Heeded'] = improve_heeded

    return [effectiveness(results, thresh=t) for t in OVERLAP_RANGE]


def reshape_to_plot(res_xtree, res_bellw):
    bugs_increased = []
    bugs_decreased = []

    for thresh, every_res_xtree, every_res_bellw in zip(OVERLAP_RANGE, res_xtree, res_bellw):
        bugs_decreased.append(
            [thresh, every_res_xtree[0], every_res_bellw[0]])

        bugs_increased.append(
            [thresh, every_res_xtree[1], every_res_bellw[1]])

    bugs_decreased = pd.DataFrame(bugs_decreased)
    bugs_increased = pd.DataFrame(bugs_increased)
    bugs_decreased.columns = ["Overlap", "XTREE", "Bellw"]
    bugs_increased.columns = ["Overlap", "XTREE", "Bellw"]

    return bugs_decreased, bugs_increased


def planning(n_reps=1, verbose=True):
    data = get_all_projects()

    bellw = data.pop('lucene').data

    if verbose:
        print("Dataset, Within-project, Cross-project")

    for proj, paths in data.iteritems():
        i = 0

        for train, test, validation in zip(paths.data[:-2], paths.data[1:-1], paths.data[2:]):
            i += 1

            "Convert to pandas type dataframe"
            bell_ = list2dataframe(bellw)
            train = list2dataframe(train)
            test = list2dataframe(test)
            validation = list2dataframe(validation)

            "10, 90% samples"
            # test = test.sample(frac=0.9, replace=True)

            for rep in xrange(n_reps):
                "Recommend changes with XTREE"
                patched_xtree = xtree(train[train.columns[1:]], test)
                patched_bellw = xtree(bell_[bell_.columns[1:]], test)
                
                res_xtree = measure_overlap(test, patched_xtree, validation)
                res_bellw = measure_overlap(test, patched_bellw, validation)

                res_dec, res_inc = reshape_to_plot(res_xtree, res_bellw)

                y_max = max(res_dec.max(axis=0).values)
                y_min = max(res_dec.min(axis=0).values)
                
                "Decrease AUC"
                xtree_dec_auc = compute_auec(res_dec[["Overlap", "XTREE"]], y_max, y_min)
                bellw_dec_auc = compute_auec(res_dec[["Overlap", "Bellw"]], y_max, y_min)
                
                if verbose:
                    print("{}-{},{},{}".format(proj, i, xtree_dec_auc, bellw_dec_auc))

    set_trace()


if __name__ == "__main__":
    planning()
