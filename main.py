"""
Compare Bellwether XTREEs with other threshold based learners.
"""

from __future__ import print_function, division

import os
import sys
from pdb import set_trace

# Update path
root = os.path.join(os.getcwd().split('project')[0], 'project')
if root not in sys.path:
    sys.path.append(root)

import numpy as np
import pandas as pd
from data.get_data import get_all_projects
from Utils.FileUtil import list2dataframe
from commons.XTREE import xtree
from Utils.StatsUtils.CrossVal import TrainTestValidate

import warnings
warnings.filterwarnings("ignore")


def planning():
    data = get_all_projects()
    results = dict()
    for proj, paths in data.iteritems():
        print(proj)
        # heeded = []
        i = 0
        for train, test, validation in TrainTestValidate.split(paths.data):
            i+=1
            results = pd.DataFrame()
            "Convert to pandas type dataframe"
            train = list2dataframe(train).dropna(axis=0).reset_index(drop=True)
            test = list2dataframe(test).dropna(axis=0).reset_index(drop=True)
            validation = list2dataframe(validation).dropna(
                axis=0).reset_index(drop=True)


            "Recommend changes with XTREE"
            new = xtree(train[train.columns[1:]], test)

            common_modules = pd.merge(test, validation, how='inner', on=['Name'])['Name']

            improve_heeded = []
            overlap = []

            for module_name in common_modules:
                
                heeded = 0

                test_value = test.loc[test['Name'] == module_name]
                plan_value = new.loc[new['Name'] == module_name]
                validate_value = validation.loc[validation['Name']
                                                == module_name]
                
                for col in test_value.columns[1:-1]:
                    try:
                        if isinstance(plan_value[col].values[0], str):
                            if eval(plan_value[col].values[0])[0] <= validate_value[col].values[0] <= eval(plan_value[col].values[0])[1]:
                                heeded += 1
                        elif plan_value[col].values[0] == validate_value[col].values[0]:
                                heeded += 1

                    except IndexError:
                        pass

                overlap.append(int(heeded * 5))
                improve_heeded.append(test_value['bug'].values[0] - validate_value['bug'].values[0])

            results['Overlap'] = overlap
            results['Heeded'] = improve_heeded

            results.to_csv("results/{}/{}_{}.csv".format(proj, proj, i), index=False)


if __name__ == "__main__":
    planning()

