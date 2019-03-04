import os
import sys
import numpy
import unittest
import pandas as pd
from pathlib import Path
root = Path(os.path.abspath(os.path.join(os.getcwd().split("src")[0], 'src')))

if root not in sys.path:
    sys.path.append(str(root))

from nose.tools import set_trace
from data.get_data import get_all_projects
from frequent_items.item_sets import ItemSetLearner


class TestItemSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestItemSets, self).__init__(*args, **kwargs)

    def test_itemset_learner(self):
        projects = get_all_projects()
        ant = projects['camel']
        ant_df = pd.concat([pd.read_csv(ant_file)
                            for ant_file in ant.data], ignore_index=True)
        ant_df = ant_df[ant_df.columns[1:]]
        ant_df.loc[ant_df[ant_df.columns[-1]] > 0, ant_df.columns[-1]] = 1
        X = ant_df[ant_df.columns[:-1]]
        y = ant_df[ant_df.columns[-1]]
        isl = ItemSetLearner()
        isl = isl.fit(X, y)
        frequent_items = isl.transform()
        set_trace()
