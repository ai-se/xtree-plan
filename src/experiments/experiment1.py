"""
Experiment 1: Measure Overlap of XTREE with Developer changes
"""
import os
import sys
from pdb import set_trace
import pandas as pd
import unittest
import numpy
import warnings
warnings.simplefilter(action='ignore')
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))
if root not in sys.path:
    sys.path.append(str(root))

from data.get_data import get_all_projects
from planners.XTREE import XTREE


class Experiment1:
    def __init__(self):
        pass

    def main(self):
        projects = get_all_projects()
        ant = projects['ant']
        for train, test in zip(ant.data[:-1], ant.data[1:]):
            test_df = pd.read_csv(test)
            train_df = pd.read_csv(train)
            # Binarize training data labels
            train_df.loc[train_df[train_df.columns[-1]] >
                         0, train_df.columns[-1]] = 1
            xtree = XTREE()
            xtree = xtree.fit(train_df)
            patch = xtree.predict(test_df)
            set_trace()


if __name__ == "__main__":
    exp = Experiment1()
    exp.main()
