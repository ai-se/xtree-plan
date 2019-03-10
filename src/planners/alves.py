"""
Deriving Metric Thresholds from Benchmark Data

Alves, T. L., Ypma, C., & Visser, J. (2010). Deriving metric thresholds from
benchmark data. In ICSM'10 (pp. 1-10). http://doi.org/10.1109/ICSM.2010.5609747
"""

from __future__ import print_function, division

import os
import sys

# Update path
root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from pdb import set_trace
from random import random as rand
from utils.file_util import list2dataframe
from sklearn.feature_selection import f_classif
from utils.experiment_utils import apply2, Changes
from oracle.smote import SMOTE


def _ent_weight(X, scale):
    try:
        loc = X["loc"].values  # LOC is the 10th index position.
    except KeyError:
        try:
            loc = X["$WCHU_numberOfLinesOfCode"].values
        except KeyError:
            loc = X["$CountLineCode"]

    return X.multiply(loc, axis="index") / scale


def alves(train, test):
    if isinstance(test, list):
        test = list2dataframe(test)

    if isinstance(test, str):
        test = list2dataframe([test])

    if isinstance(train, list):
        train = list2dataframe(train)

    train.loc[train[train.columns[-1]] == 1, train.columns[-1]] = True
    train.loc[train[train.columns[-1]] == 0, train.columns[-1]] = False
    metrics = [met[1:] for met in train[train.columns[:-1]]]

    X = train[train.columns[:-1]]  # Independent Features (CK-Metrics)
    changes = []

    """
    As weight we will consider
    the source lines of code (LOC) of the entity.
    """

    loc_key = "loc"
    tot_loc = train.sum()["loc"]
    X = _ent_weight(X, scale=tot_loc)

    """
    Divide the entity weight by the sum of all weights of the same system.
    """
    denom = pd.DataFrame(X).sum().values
    norm_sum = pd.DataFrame(pd.DataFrame(X).values / denom, columns=X.columns)

    """
    Find Thresholds
    """
    y = train[train.columns[-1]]  # Dependent Feature (Bugs)
    pVal = f_classif(X, y)[1]  # P-Values
    cutoff = []
    def cumsum(vals): return [sum(vals[:i]) for i, __ in enumerate(vals)]

    def point(array):
        for idx, val in enumerate(array):
            if val > 0.95:
                return idx

    for idx in range(len(train.columns[:-1])):
        # Setup Cumulative Dist. Func.
        name = train.columns[idx]
        loc = train[loc_key].values
        vals = norm_sum[name].values
        sorted_ids = np.argsort(vals)
        cumulative = [sum(vals[:i]) for i, __ in enumerate(sorted(vals))]
        cutpoint = point(cumulative)
        cutoff.append(vals[sorted_ids[cutpoint]] * tot_loc / loc[
            sorted_ids[cutpoint]] * denom[idx])

    """
    Apply Plans Sequentially
    """

    modified = []
    for n in range(test.shape[0]):
        if test.iloc[n][-1] > 0 and rand() > 0.5:
            new_row = apply2(cutoff, test.iloc[n].values.tolist())
            modified.append(new_row)

    return pd.DataFrame(modified, columns=test.columns)
