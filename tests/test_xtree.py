import os
import sys
import numpy
import unittest
import pandas as pd
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))

if root not in sys.path:
    sys.path.append(str(root))

from nose.tools import set_trace;
from data.get_data import get_all_projects
from planners.XTREE import XTREE

class TestXTREE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestXTREE, self).__init__(*args, **kwargs)
    
    def test_build_tree(self):
        projects = get_all_projects()
        ant = projects['ant']
        test_df = pd.read_csv(ant.data[-1])
        train_df = pd.concat([pd.read_csv(ant_file) for ant_file in ant.data[:-1]])
        train_df.loc[train_df[train_df.columns[-1]] > 0, train_df.columns[-1]] = 1
        X_train = train_df[train_df.columns[1:]]
        X_test = test_df[test_df.columns[1:]]
        xtree = XTREE()
        xtree.fit(X_train)
        xtree.predict(X_test)
        set_trace()

    