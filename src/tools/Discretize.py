"""
An instance filter that discretizes a range of numeric attributes in the dataset into nominal attributes. Discretization is by Fayyad & Irani's MDL method (the default).

For more information, see:

Usama M. Fayyad, Keki B. Irani: Multi-interval discretization of continuous valued attributes for classification learning. In: Thirteenth International Joint Conference on Artificial Intelligence, 1022-1027, 1993.

Igor Kononenko: On Biases in Estimating Multi-Valued Attributes. In: 14th International Joint Conference on Articial Intelligence, 1034-1040, 1995.

Dougherty, James, Ron Kohavi, and Mehran Sahami. "Supervised and unsupervised discretization of continuous features." Machine learning: proceedings of the twelfth international conference. Vol. 12. 1995.
"""
from __future__ import division, print_function

from .misc import *
import numpy as np
import pandas as pd
from pdb import set_trace
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as CART


def fWeight(tbl):
    """
    Sort features based on entropy
    """
    clf = CART(criterion='entropy')
    features = tbl.columns[1:-1]
    klass = tbl[tbl.columns[-1]]
    clf.fit(tbl[features], [k == True for k in klass])
    lbs = clf.feature_importances_

    return [tbl.columns[i] for i in np.argsort(lbs)[::-1]]


def discretize(feature, klass, atleast=-1, discrete=False):
    """
    Recursive Minimal Entropy Discretization
    ````````````````````````````````````````
    Inputs:
      feature: A list or a numpy array of continuous attributes
      klass: A list, or a numpy array of discrete class labels.
      atleast: minimum splits.
    Outputs:
      splits: A list containing suggested spilt locations
    """

    def measure(x):
        def ent(x):
            C = Counter(x)
            N = len(x)
            return sum([-C[n] / N * np.log(C[n] / N) for n in C.keys()])

        def stdev(x):
            if np.isnan(np.var(x) ** 0.5):
                return 0
            return np.var(x) ** 0.5

        if not discrete:
            return ent(x)
        else:
            return stdev(x)

    # Sort features and klass
    feature, klass = sorted(feature), [k for (f, k) in
                                       sorted(zip(feature, klass))]
    splits = []
    gain = []
    lvl = 0

    def redo(feature, klass, lvl):
        if len(feature) > 0:
            E = measure(klass)
            N = len(klass)
            T = []  # Record boundaries of splits
            for k in range(len(feature)):
                west, east = feature[:k], feature[k:]
                k_w, k_e = klass[:k], klass[k:]
                N_w, N_e = len(west), len(east)
                T += [N_w / N * measure(k_w) + N_e / N * measure(k_e)]

            T_min = np.argmin(T)
            left, right = feature[:T_min], feature[T_min:]
            k_l, k_r = klass[:T_min], klass[T_min:]

            # set_trace()
            def stop(k, k_l, k_r):
                gain = E - T[T_min]

                def count(lst): return len(Counter(lst).keys())

                delta = np.log2(float(3 ** count(k) - 2)) - (
                    count(k) * measure(k) - count(k_l) * measure(k_l) - count(
                        k_r) * measure(k_r))

                return T_min == 0 or gain < (np.log2(N - 1) + delta) / N

            if stop(klass, k_l, k_r) and lvl >= atleast:
                if discrete:
                    splits.append(T_min)
                else:
                    splits.append(feature[T_min])

            else:
                _ = redo(feature=left, klass=k_l, lvl=lvl + 1)
                _ = redo(feature=right, klass=k_r, lvl=lvl + 1)

    # ------ main ------
    redo(feature, klass, lvl=0)
    # set_trace()
    return splits
