import os
import sys
import numpy as np
import pandas as pd
from pdb import set_trace
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter

root = os.path.join(os.getcwd().split(
    'src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from data.get_data import get_all_projects
from .fp_growth import find_frequent_itemsets
from tools.Discretize import discretize


class ItemSetLearner(BaseEstimator):
    def __init__(self, bins=3, support_min=90):
        """
        Frequent itemset learner based on FPgrowth algorithm

        Parameters
        ----------
        support_min: int (default 50)
            Minimum support for FPGrowth
        bins: int (default 3)
            Number of discrete bins
        """
        self.support_min = support_min
        self.bins = bins
        self._x_transformed = None

    @staticmethod
    def _get_transactions(Xt):
        cols = Xt.columns.tolist()
        transactions = []
        for i in range(len(Xt)):
            change_set = [cols[j]
                          for j, val in enumerate(Xt.iloc[i]) if val > 0]
            transactions.append(tuple(change_set))
        return transactions

    @staticmethod
    def discretize_dframe(X, y):
        """ Discretize a DataFrame

        Parameters
        ----------
        <pandas.core.frame.DataFrame>:
            Numeric Table

        Returns
        --------
        <pandas.core.frame.DataFrame>: 
            Discretized table
        """
        discrete_ndarray = [
            [0 for _ in range(len(X.columns))] for __ in range(len(X))]
        for col_id, name in enumerate(X.columns):
            indep = X[name].values
            depen = y.values
            splits = discretize(indep, depen)
            col_min, col_max = min(indep), max(indep)
            cutoffs = sorted(list(set(splits + [col_min, col_max])))
            for i, (range_lo, range_hi) in enumerate(zip(cutoffs[:-1], cutoffs[1:])):
                for row_id, val in enumerate(indep):
                    if range_lo <= val <= range_hi:
                        discrete_ndarray[row_id][col_id] = i
        discrete_ndarray = pd.DataFrame(discrete_ndarray, columns=X.columns)
        return discrete_ndarray

    def fit(self, X, y):
        """ 
        Fit data by binning into data into discrete intervals.

        Parameters
        ----------
        X: array_like (n x m)
            A list (or an array) of continuous attributes
        y: array_like (n X 1)
            A list (or a numpy array) of discrete class labels.    
        """

        # est = KBinsDiscretizer(
        #     n_bins=self.bins, encode='ordinal', strategy='kmeans')
        # Xt = est.fit_transform(X, y)
        # Xt = pd.DataFrame(Xt, columns=X.columns)

        Xt = self.discretize_dframe(X, y)
        self._x_transformed = Xt
        return self

    def transform(self):
        """ 
        Transform input data into a list of frequent items

        Returns
        -------
        List[tuple]:
            A list of frequent items.
        """

        transactions = self._get_transactions(self._x_transformed)
        self.frequent_items = [set(item) for item in find_frequent_itemsets(
            transactions, minimum_support=self.support_min) if len(item) > 1]
        return self.frequent_items

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X: array_like (n x m)
            A list (or an array) of continuous attributes
        y: array_like (n X 1)
            A list (or a numpy array) of discrete class labels.    

        Returns
        -------
        List[tuple]:
            A list of frequent items.
        """

        self.fit(X, y)
        return self.transform()
