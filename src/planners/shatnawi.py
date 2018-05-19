"""
XTREE
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
# from pdb import set_trace
from random import random as rand
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from utils.experiment_utils import apply2, Changes
from utils.file_util import list2dataframe
from oracle.smote import SMOTE


def VARL(coef, inter, p0=0.05):
    """
    :param coef: Slope of   (Y=aX+b)
    :param inter: Intercept (Y=aX+b)
    :param p0: Confidence Interval. Default p=0.05 (95%)
    :return: VARL threshold
  
              1   /     /  p0   \             \
    VARL = ----- | log | ------ | - intercept |
           slope \     \ 1 - p0 /             /
  
    """
    return (np.log(p0 / (1 - p0)) - inter) / coef


def shatnawi(train, test):
    """
    Implements shatnavi's threshold based planner.
    :param train: 
    :param test: 
    :param rftrain: 
    :param tunings: 
    :param verbose: 
    :return: 
    """
    "Compute Thresholds"


    if isinstance(test, list):
        test = list2dataframe(test)

    if isinstance(test, basestring):
        test = list2dataframe([test])

    if isinstance(train, list):
        train = list2dataframe(train)

    changed =[]
    metrics = [str[1:] for str in train[train.columns[:-1]]]
    ubr = LogisticRegression()  # Init LogisticRegressor
    X = train[
        train.columns[:-1]]  # Independent Features (CK-Metrics)
    y = train[train.columns[-1]]  # Dependent Feature (Bugs)

    ubr.fit(X, y.values.tolist())  # Fit Logit curve
    inter = ubr.intercept_[0]  # Intercepts
    coef = ubr.coef_[0]  # Slopes
    pVal = f_classif(X, y)[1]  # P-Values
    changes = len(metrics) * [-1]

    "Find Thresholds using VARL"
    for Coeff, P_Val, idx in zip(coef, pVal,
                                 range(len(metrics))):  # xrange(len(metrics)):
        thresh = VARL(Coeff, inter, p0=0.005)  # VARL p0=0.05 (95% CI)
        if P_Val < 0.05:
            changes[idx] = thresh

    # set_trace()

    """
    Apply Plans Sequentially
    """

    modified = []
    for n in xrange(test.shape[0]):
        if test.iloc[n][-1] > 0 or test.iloc[n][-1] == True:
            new_row = apply2(changes, test.iloc[n].values.tolist())
            modified.append(new_row)

        # Disable the next two line if you're measuring the number of changes.
        else:
            if rand() > 0.7:
                modified.append(test.iloc[n].tolist())

    return pd.DataFrame(modified, columns=test.columns)